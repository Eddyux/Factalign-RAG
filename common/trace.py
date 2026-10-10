"""逐题 trace：把「我发的 prompt / 模型回的原文 / 每轮检索到的 chunks 和实体」按题落盘。

为什么需要：`logs/{method}/*.log` 只有 stdout 文本（建图/检索摘要 + 指标），
出问题时没法回答「这道题第 2 轮检索到底召回了哪几块、prompt 里 system 是哪个版本」。
本模块把**每一次 LLM 调用的完整请求与原始响应**、以及**每一轮检索的候选**结构化存下来。

存储形态（设计已定）：
  traces/{method}/{dataset}/v{n}r{m}/        每题一个 json（中断只丢当前题、可 cat）
  traces/{method}/{dataset}/v{n}r{m}.jsonl  逐题一行索引（qid → 文件 + 状态 + 指标）

挂钩方式（零侵入）：`common/llm.py` 的 chat()/chat_messages() 出口调 `record_llm_call()`；
它从**线程本地**取当前题的 trace。run.py 的 `answer_one` 在题目开始时 `start_trace()`，
结束时 `finish_trace()` 落盘。未开 trace（默认）时 record_llm_call 立即返回，开销≈0。

体积：一问的完整 prompt（含 few-shot system）+ 原始输出 ≈ 5–15KB；
500 题 × 12 方法 × 3 数据集全量 ≈ 200MB 级 —— 由 `--no-trace-prompt` 可只留 user 消息。
"""

import json
import os
import threading
import time
from pathlib import Path

from common.paths import (question_trace_path, trace_index_path, trace_run_dir,
                          rel_to_root, sha256_text, provenance_stamp, write_json)

# 线程本地：每个 worker 线程装自己那道题的 trace（run.py 是多线程 answer_one）
_local = threading.local()

# 每题 prompt 里的长文本截断上限（chunk 正文只留头部；全文按 sha256 可反查 merged/chunks.jsonl）
HEAD_CHARS = 200
# 单题最多记多少次 LLM 调用 / 多少轮检索（防止病态重试把文件写爆）
MAX_LLM_CALLS = 200
MAX_ROUNDS = 64
# 单条 prompt 消息超长时截断（默认不截：用户要的就是"原始记录"）
MAX_MESSAGE_CHARS = 0          # 0 = 不截（要截可设环境变量 TRACE_MAX_MESSAGE_CHARS）

# ── 重复 prompt 去重（IRCOT 那种 69k 字符的 29-shot，每题要塞进 10 次调用）──
# 同一段文本只在 prompts/{sha}.txt 存**一份**，trace 里留 content_sha256 指针。
# 实测：IRCOT 单数据集 500 题从 452MB 降到 ~67MB（省 85%），整表省 73%。
DEDUP_MIN_CHARS = int(os.environ.get("TRACE_DEDUP_MIN_CHARS", "512"))
_prompt_reg: dict[str, int] = {}          # sha → 出现次数
_prompt_lock = threading.Lock()
_prompt_dir = None                        # 由 set_trace_dir() 设定
_prompt_stats = {"files": 0, "chars_saved": 0, "chars_inline": 0}


def set_trace_dir(run_dir) -> None:
    """告诉 trace 去重层：这次 run 的 `prompts/` 共享目录在哪（写文件前必须先设）。"""
    global _prompt_dir, _prompt_reg
    _prompt_dir = Path(run_dir) / "prompts"
    with _prompt_lock:
        _prompt_reg = {}
        _prompt_stats.update({"files": 0, "chars_saved": 0, "chars_inline": 0})


def prompt_dedup_stats() -> dict:
    with _prompt_lock:
        return dict(_prompt_stats)


def _write_prompt_file(sha: str, content: str) -> Path:
    d = _prompt_dir
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{sha[:16]}.txt"
    if not p.exists():
        tmp = p.with_suffix(".txt.tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(p)
    return p


def dedup_message(rec: dict) -> None:
    """把「重复出现的长消息」改成 {sha 指针 + 头 200 字}，全文只在 prompts/ 存一次。

    对**所有角色**都做（不只是 system）：注意 `rag_prompt.txt` 这种统一作答模板是塞进
    **user** 消息的（2554 字符，每题都一样），IRCOT 的 69k 官方 prompt 也在 user 里 ——
    只对 system 去重会漏掉大头。

    没设共享目录（set_trace_dir 未调用）或内容太短 → 原样保留。
    任何异常都吞掉：trace 是留档，绝不能因为去重失败反而丢内容/影响实验。
    """
    if _prompt_dir is None:
        return
    content = rec.get("content")
    if not isinstance(content, str) or len(content) < DEDUP_MIN_CHARS:
        return
    try:
        sha = sha256_text(content)
        with _prompt_lock:
            seen = _prompt_reg.get(sha)
            _prompt_reg[sha] = (seen or 0) + 1
        if not seen:                       # 第一次出现 → 写共享文件，同时保留内联（可读性）
            _write_prompt_file(sha, content)
            with _prompt_lock:
                _prompt_stats["files"] += 1
                _prompt_stats["chars_inline"] += len(content)
            rec["prompt_file"] = f"prompts/{sha[:16]}.txt"
            return
        rec["content"] = content[:HEAD_CHARS]
        rec["prompt_file"] = f"prompts/{sha[:16]}.txt"
        rec["dedup"] = True
        with _prompt_lock:
            _prompt_stats["chars_saved"] += len(content) - HEAD_CHARS
    except Exception:                       # noqa: BLE001
        pass


class QuestionTrace:
    """一道题的 trace 累积器（线程内使用，不做跨线程共享）。"""

    def __init__(self, method: str, dataset: str, qid: str, sample: int = 0,
                 graph_version: int = 1, retrieval: int = 1, question: str = "",
                 gold: str = "", aliases=None, keep_prompt: bool = True):
        self.method = method
        self.dataset = dataset
        self.qid = str(qid)
        self.sample = sample
        self.graph_version = graph_version
        self.retrieval = retrieval
        self.question = question
        self.gold = gold
        self.aliases = list(aliases or [])
        self.keep_prompt = keep_prompt
        self.t0 = time.time()
        self.ts_start = time.strftime("%Y-%m-%d %H:%M:%S")
        self.status = "ok"
        self.retries = 0
        self.errors: list = []
        self.llm_calls: list = []
        self.rounds: list = []
        self.final: dict = {}
        self._lock = threading.Lock()

    # ── ① LLM 调用：prompt + 原始输出 ──
    def add_llm_call(self, messages, raw_output, pt=0, ct=0, seconds=0.0,
                     role="", phase="query", round_idx=None, parsed=None, error=None,
                     max_chars: int = MAX_MESSAGE_CHARS):
        if len(self.llm_calls) >= MAX_LLM_CALLS:
            return
        msgs = []
        for m in (messages or []):
            if not isinstance(m, dict):
                continue
            content = m.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            rec = {"role": m.get("role", ""), "content": content}
            if not self.keep_prompt:                      # 只留 user，system 用哈希代替
                if rec["role"] != "user":
                    rec = {"role": rec["role"], "content_sha256": sha256_text(content),
                           "chars": len(content)}
            elif max_chars and len(content) > max_chars:
                rec = {"role": rec["role"], "content": content[:max_chars],
                       "content_sha256": sha256_text(content), "chars": len(content),
                       "truncated": True}
            dedup_message(rec)                            # 重复的长 prompt → 存一次，其余留哈希
            msgs.append(rec)
        self.llm_calls.append({
            "i": len(self.llm_calls), "phase": phase, "round": round_idx, "role": role,
            "messages": msgs,
            "raw_output": raw_output,
            "output_sha256": sha256_text(raw_output if isinstance(raw_output, str)
                                         else json.dumps(raw_output, ensure_ascii=False)),
            "parsed": parsed,
            "usage": {"prompt_tokens": int(pt or 0), "completion_tokens": int(ct or 0),
                      "seconds": round(float(seconds or 0), 3)},
            "error": error,
        })

    # ── ② 每轮检索：chunks / 实体 ──
    def add_round(self, round_idx=None, query_used: str = "", chunks=None, entities=None,
                  paths=None, sub_questions=None, extra=None):
        if len(self.rounds) >= MAX_ROUNDS:
            return
        out_chunks = []
        for c in (chunks or []):
            if isinstance(c, str):
                out_chunks.append({"text_head": c[:HEAD_CHARS], "text_sha256": sha256_text(c),
                                   "chars": len(c)})
            elif isinstance(c, dict):
                text = c.get("text") or c.get("window_text") or ""
                rec = {k: v for k, v in c.items()
                       if k not in ("text", "window_text", "summary")}
                if text:
                    rec.setdefault("text_head", text[:HEAD_CHARS])
                    rec.setdefault("text_sha256", sha256_text(text))
                    rec.setdefault("chars", len(text))
                out_chunks.append(rec)
        self.rounds.append({
            "round": len(self.rounds) if round_idx is None else round_idx,
            "query_used": query_used,
            "chunks": out_chunks,
            "entities": list(entities or []),
            "paths": list(paths or []),
            "sub_questions": list(sub_questions or []),
            **(extra or {}),
        })

    # ── ③ 收尾 ──
    def set_final(self, answer="", evidence=None, evidence_locs=None, extra=None):
        """final 里**不重复存证据全文**：正文已在 retrieval_rounds 的 chunks 里
        （head+sha256），这里只留同样的 head/hash 对，避免一题里同一段文本存两遍
        （实测能省 30~40% trace 体积）。"""
        ev = []
        for e in (evidence or []):
            text = e if isinstance(e, str) else (e.get("text") or e.get("window_text") or "")
            ev.append({"text_head": text[:HEAD_CHARS], "text_sha256": sha256_text(text),
                       "chars": len(text)} if text else {"text_head": "", "chars": 0})
        self.final = {"answer": answer,
                      "evidence": ev,
                      "evidence_locs": [list(x) if isinstance(x, (list, tuple)) else x
                                        for x in (evidence_locs or [])],
                      **(extra or {})}

    def note_error(self, msg: str, retries: int = 0):
        self.errors.append(str(msg)[:500])
        if retries:
            self.retries = int(retries)

    def to_dict(self) -> dict:
        return {
            "qid": self.qid, "method": self.method, "dataset": self.dataset,
            "sample": self.sample, "graph_version": self.graph_version,
            "retrieval": self.retrieval,
            "ts_start": self.ts_start, "seconds": round(time.time() - self.t0, 3),
            "status": self.status, "retries": self.retries,
            "code": provenance_stamp(),
            "question": self.question, "gold": self.gold, "answer_aliases": self.aliases,
            "llm_calls": self.llm_calls,
            "retrieval_rounds": self.rounds,
            "final": self.final,
            "errors": self.errors,
        }


# ══════════════════════════════════════════════════════════════════════════════
# 线程本地挂载点（run.py 每题 start/finish；llm.py 出口 record）
# ══════════════════════════════════════════════════════════════════════════════
def start_trace(**kwargs) -> QuestionTrace:
    tr = QuestionTrace(**kwargs)
    _local.trace = tr
    return tr


def current_trace():
    return getattr(_local, "trace", None)


def clear_trace():
    _local.trace = None


def record_llm_call(messages, raw_output, pt=0, ct=0, seconds=0.0, **kwargs) -> None:
    """给 common/llm.py 用的出口钩子：没开 trace 时立即返回（热路径零开销）。"""
    tr = getattr(_local, "trace", None)
    if tr is None:
        return
    try:
        tr.add_llm_call(messages, raw_output, pt=pt, ct=ct, seconds=seconds, **kwargs)
    except Exception:                       # noqa: BLE001  trace 绝不能影响实验
        pass


# ══════════════════════════════════════════════════════════════════════════════
# 落盘 + 索引
# ══════════════════════════════════════════════════════════════════════════════
def finish_trace(tr: QuestionTrace, method: str, dataset: str, v: int, r: int,
                 sample: int = 0, em=None, f1=None, answer: str = "") -> dict:
    """写单题 json + 追加索引行，返回该行的 dict（供 run.py 汇总）。"""
    tr.status = "error" if tr.errors and not tr.final.get("answer") else (
        "retried" if tr.retries else tr.status)
    if not tr.final:
        tr.set_final(answer=answer)
    data = tr.to_dict()
    path = question_trace_path(method, dataset, v, r, tr.qid, sample=sample)
    write_json(path, data)
    row = {"qid": tr.qid, "trace": rel_to_root(path), "status": tr.status,
           "seconds": data["seconds"], "n_llm_calls": len(data["llm_calls"]),
           "n_rounds": len(data["retrieval_rounds"]), "em": em, "f1": f1,
           "retries": tr.retries, "answer": answer}
    p = trace_index_path(method, dataset, v, r, sample=sample)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def trace_run_header(method: str, dataset: str, v: int, r: int, sample: int = 0,
                     extra: dict | None = None) -> dict:
    """一次 run 的 trace 头（写进同目录 `_run.json`，说明这份 trace 是哪个配置跑的）。"""
    d = trace_run_dir(method, dataset, v, r, sample)
    d.mkdir(parents=True, exist_ok=True)
    hdr = {"method": method, "dataset": dataset, "sample": sample,
           "graph_version": v, "retrieval": r,
           "trace_dir": rel_to_root(d), **(provenance_stamp()), **(extra or {})}
    write_json(d / "_run.json", hdr)
    return hdr


# ══════════════════════════════════════════════════════════════════════════════
# 方法返回值 → trace 的「每轮检索」记录（通用兜底）
# ══════════════════════════════════════════════════════════════════════════════
def normalize_result(r: dict) -> dict:
    """把各方法 `query()` 的返回统一成 {answer, evidence, evidence_locs}。

    各 adapter 的字段名不完全一样（evidence_locs / locs、entities / …），
    trace 与结果行都只依赖这三个键，缺的补空。
    """
    r = r or {}
    return {
        "answer": str(r.get("answer", "") or ""),
        "evidence": list(r.get("evidence") or []),
        "evidence_locs": list(r.get("evidence_locs") or r.get("locs") or []),
        "entities": list(r.get("entities") or []),
        "_raw_keys": sorted(r.keys()),
    }


def rounds_from_result(r: dict, question: str = "") -> list[dict]:
    """从方法返回值里尽量还原「每轮检索」。

    优先用适配器自己给的轮次信息（IRCoT 的 thoughts/retrieval_rounds、
    S2G-RAG 的 sufficiency_trace、PropRAG 的 retrieval_paths、Self-Ask 的
    sub_questions…）；都没有时退化成**一轮**：最终的证据块 + 实体。
    这样"每轮得到哪些 chunks / 实体"在任何方法上都有东西可看。

    约定：适配器可以主动塞 `trace_rounds`（list[dict]）来精确表达每轮，
    键名与 QuestionTrace.add_round 的参数一致（query_used/chunks/entities/paths/
    sub_questions/extra）。有 trace_rounds 时优先用它。
    """
    r = r or {}
    if isinstance(r.get("trace_rounds"), list) and r["trace_rounds"]:
        return [dict(x) for x in r["trace_rounds"]]

    ev = list(r.get("evidence") or [])
    ents = list(r.get("entities") or [])
    # ⚠️ 两个字段名都要收：`retrieval_paths` 是 proprag 的老名，`paths` 是 hipporag
    #（图谱种子三元组）和 hypergraphrag（超边）现在返回的检索路径。只读前者会让
    # 新字段被**悄悄丢掉** —— 实测过。
    paths = r.get("retrieval_paths") or r.get("paths") or []
    subqs = r.get("sub_questions") or []
    thoughts = r.get("thoughts") or []
    dlog = r.get("decomp_log") or []

    rounds: list[dict] = []
    # ① factalign：`decomp_log` = [("SEARCH", 子查询, [{"text": 命题, "prop_id": …, "sem": …}, …]),
    #    ("ENOUGH"/"STOP", 原因, [])] —— 命中项是 **dict 形态的命题**（`factalign.py` 的
    #    `hits`），走下面的 dict 分支：`text` 被消费成 `text_head`/`sha256`/`chars`，
    #    `prop_id`/`sem` 原样保留。**别退回 tuple** —— 那样 id 无处安放、且上限只有
    #    两个值（史上是"命题前70字"，会让 `chars` 恒为假值 70）。**必须单独读它**：它被
    #    `_result()` 返回了，但本函数原先不认这个字段 → 落到 ④ 的单轮兜底，于是"哪一轮
    #    查了什么、命中什么、第几轮判定够了"在 trace 里全看不到（只能去翻 llm_calls.jsonl）。
    if dlog:
        for kind, sq, hits in dlog:
            if kind in ("ENOUGH", "STOP"):
                rounds.append({"round": len(rounds), "query_used": "", "chunks": [],
                               "entities": [], "paths": [], "sub_questions": [],
                               "note": (f"停止拆解：{kind}" + (f"（{sq}）" if sq else ""))})
                continue
            _chunks = []
            for rank, h in enumerate(hits or [], start=1):
                # hits 的元素是 (文本前 70 字, 相似度)，也可能已是 dict（兼容两种形态）
                if isinstance(h, (list, tuple)) and len(h) == 2:
                    _chunks.append({"text": str(h[0]), "sem": float(h[1]), "rank": rank})
                elif isinstance(h, dict):
                    _chunks.append({**h, "rank": rank})
                else:
                    _chunks.append({"text": str(h), "rank": rank})
            rounds.append({"round": len(rounds), "query_used": str(sq), "chunks": _chunks,
                           "entities": [], "paths": [], "sub_questions": [],
                           "note": "factalign 拆解轮"})
        return rounds
    # ② IRCoT：每轮一句 thought + 当时累积的段落
    if thoughts:
        for i, th in enumerate(thoughts):
            rounds.append({"round": i, "query_used": str(th)[:300],
                           "chunks": ev, "entities": [], "paths": [],
                           "sub_questions": [], "note": "IRCoT thought 轮"})
        return rounds
    # ③ Self-Ask / SubQRAG：每个子问题一轮
    if subqs:
        for i, sq in enumerate(subqs):
            rounds.append({"round": i, "query_used": str(sq), "chunks": ev,
                           "entities": [], "paths": [], "sub_questions": [str(sq)]})
        return rounds
    # ④ 单轮兜底：最终证据 + 实体（+ 路径）
    rounds.append({"round": 0, "query_used": question, "chunks": ev,
                   "entities": ents, "paths": paths, "sub_questions": subqs})
    return rounds
