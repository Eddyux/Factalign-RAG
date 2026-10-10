"""S2G-RAG 适配器（ACL 2026）——结构化充分性 + gap 判定的迭代检索。

论文: "S2G-RAG: Structured Sufficiency and Gap Judging for Iterative Retrieval-Augmented QA"
(Li, Zou, Lv, Zhang, Zhou; ACL 2026 long, doi 10.18653/v1/2026.acl-long.1185)
官方代码: https://github.com/nianaaa/S2G-RAG  →  `baselines/S2G-RAG-main/`

═══ 官方机制（逐函数核对过 inference/inference_bm25.py + utils/prompt_template.py）═══
  主循环（`main_batch`，`turn <= max_turns`，所以最多 max_turns+1 次判定）:
    1. **S2G-Judge**：系统 prompt = `SUFF_SYSTEM_PROMPT`，user = 「QUESTION + CONTEXT(累积证据)」，
       输出 JSON `{"sufficient": bool, "gap_items":[{category, target, slot, description}]}`
       （category ∈ bridge_entity | attribute | relation | evidence_span | other）
       —— judge **只看问题 + 已累积的句子级证据**，不看原始文档。
    2. **首轮防过度自信**：turn==0 且 verdict=True 但证据为空 → 强制判为需检索（`should_force_first_retrieval`）。
    3. **gap → 下一轮 query**（`build_query_from_missing`）：取前 3 个 gap，
       `target+slot` 有就拼 `"{target} {slot}"`，否则用 `description`，
       最终 `query = 原问题 + " " + 这些短语`（triviaqa 只取 1 个 gap）。
    4. **句子级 Evidence Selector**（`concat_and_pick_sentences_batch`）：把**本轮新检索到的文档**
       切成句（pysbd，失败退化正则），全局编号 `[i] (title | s#n) sentence`，user prompt 里带
       「ORIGINAL QUESTION / MISSING FACTS TO FILL（top3）/ NUMBERED SENTENCES / 最多选 K 句」，
       输出 `{"evidence_global_ids":[...]}` → 映射回 (doc, 句号)。
    5. **证据合并**（`merge_evidence_only` + `append_evidence_context`）：
       每个文档只留选中的句子 → `[title] EVIDENCE:\n- 句1\n- 句2`；追加时若新块已出现在旧上下文里则跳过（去重）。
    6. 判为 sufficient 或预算耗尽 → **answer reasoner** 用 `force_answer_prompt`（要求
       `Answer: … / Rationale: …`）作答，`extract_final_answer_and_rationale` 取答案。

═══ 本项目的适配取舍（路线 C，论文里必须写明）═══
  - **零下载、零训练**：官方 inference 脚本硬要求 `LLAMA_PATH`(Llama-3-8B) + `SUFF_BASE_PATH`
    (Llama-3.2-3B) + **`SUFF_LORA_PATH`（训练好的 judge adapter，官方未提供）**。
    本适配器把 judge/selector 都走**本地 vLLM**（`common.llm.chat`）+ 官方 prompt，
    即「**prompt-level 复现，未使用官方训练的 judge**」；表格里必须标注这一点。
  - **检索换成我们的底座**：官方用 pyserini BM25 + 全 wiki 语料；这里用 `CorpusIndex`
    （整段 passage + dense/BM25 RRF，同 standardrag）。`remove_repeat_docs` 语义保留
    （`exclude` 已取过的块）。
  - **句子级证据**：对我们的块切句（默认块 = 一个整段 passage；官方 pysbd；这里用等价正则 + 每块最多 40 句，
    与官方同参数），编号与选句逻辑逐字对齐。
  - **作答口径**：走本项目统一的 `generate_answer`（证据 = 选中的句子级证据），
    与其余 baseline 同生成口径。★ 官方 `force_answer_prompt` 那条对照路径**已删除**：
    全项目的最终作答只有两条生成器（`prompts/rag_prompt.txt` / `rag_prompt_triviaqa.txt`）。
  - 成本：每题 judge ≤ max_turns+1 次 + selector ≤ max_turns 次 + 作答 1 次
    = **≤ 10 次 LLM 调用**（max_turns=4 时：judge 5 + selector 4 + 作答 1），比 IRCoT(2~4) 贵。
    （`build_query` 是模板拼接，不调 LLM。提前判 sufficient 会同时省掉后面的 judge 与 selector。）

用法:
  python run.py --method s2grag --dataset hotpotqa --sample 300 --workers 8
  旋钮（环境变量，无需改代码）:
    S2GRAG_MAX_TURNS=4  S2GRAG_TOP_DOCS=6  S2GRAG_SELECT_TOP_K=6  S2GRAG_MAX_FACTS=3
    S2GRAG_JUDGE_JSON_MODE=1
"""

import json
import os
import re
from pathlib import Path

from methods.base import BaseMethod, generate_answer
from methods.corpus_index import CorpusIndex
from methods.scaffold import SourceNotReady, require_repo
from common.llm import chat as llm_chat, role as llm_role
from config import BASELINES_DIR

# ── 官方仓库与超参（默认取官方 inference 脚本的值）──
S2GRAG_REPO_CANDIDATES = [BASELINES_DIR / "S2G-RAG", BASELINES_DIR / "S2G-RAG-main"]
S2GRAG_MAX_TURNS = int(os.environ.get("S2GRAG_MAX_TURNS", "4"))          # 官方默认 4
S2GRAG_TOP_DOCS = int(os.environ.get("S2GRAG_TOP_DOCS", "6"))
# BM25 打分参数：官方 Lucene `BM25Similarity(k1=0.9, b=0.4)`（bm25_init_hotpotqa.py:8 等）。
# 我们全局默认是 1.5/0.75，这里按官方覆盖。
S2GRAG_BM25_K1 = float(os.environ.get("S2GRAG_BM25_K1", "0.9"))
S2GRAG_BM25_B = float(os.environ.get("S2GRAG_BM25_B", "0.4"))
# 检索方式：官方**只用纯 BM25**（inference/inference_bm25.py，无 dense、无融合）；
# 我们此前是 dense+BM25 的 RRF 融合 —— 这是与官方口径的实质偏离，故默认改为官方值。
# 设 S2GRAG_RETRIEVAL_MODE=rrf 可回到旧行为（与历史 run 对账用）。
S2GRAG_RETRIEVAL_MODE = os.environ.get("S2GRAG_RETRIEVAL_MODE", "bm25")
S2GRAG_SELECT_TOP_K = int(os.environ.get("S2GRAG_SELECT_TOP_K", "6"))    # selector 最多选几句
S2GRAG_MAX_FACTS = int(os.environ.get("S2GRAG_MAX_FACTS", "3"))          # gap→query 用前 3 个
S2GRAG_MAX_SENTS_PER_DOC = int(os.environ.get("S2GRAG_MAX_SENTS_PER_DOC", "40"))
S2GRAG_JUDGE_JSON_MODE = os.environ.get("S2GRAG_JUDGE_JSON_MODE", "1") == "1"
S2GRAG_JUDGE_MAX_TOKENS = int(os.environ.get("S2GRAG_JUDGE_MAX_TOKENS", "256"))
S2GRAG_SELECT_MAX_TOKENS = int(os.environ.get("S2GRAG_SELECT_MAX_TOKENS", "64"))

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|[\n]+")
_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _load_official_prompts(repo: Path) -> dict:
    """读官方 `utils/prompt_template.py` 的常量（**按路径 exec，不污染 sys.path 的 utils 包**）。

    该文件只有字符串常量、无 import 无副作用；exec 比 `import utils.prompt_template` 更安全
    ——后者会把 `utils` 这个名字塞进 sys.modules，污染其它 adapter 的同名包导入。
    """
    p = repo / "utils" / "prompt_template.py"
    if not p.exists():
        raise SourceNotReady(f"[s2grag] 官方 prompt 文件缺失：{p}")
    ns: dict = {}
    exec(compile(p.read_text(encoding="utf-8"), str(p), "exec"), ns)
    return ns


def _selector_system_prompt(P: dict, dataset: str) -> str:
    """逐字复刻官方 `generation/generation.py:71-75 get_selector_system_prompt`。

    ⚠️ **这是检索侧 prompt**（决定从检索到的文档里挑哪些句子进证据）→ 按本项目
    「官方 prompt 只在**检索侧**逐字复刻」的原则，必须走官方这套**按数据集分派**，
    不能硬用通用的 `SELECTOR_SYSTEM_PROMPT`。

    官方原文（`inference/inference_bm25.py:66`、`inference/inference_e5.py:106`
    还各有一份**逐行完全相同**的副本，因为官方拿同一套生成器跑 S2G-RAG 与基线）：
        if dataset_name.strip().lower() == "triviaqa":
            return TRIVIAQA_SELECTOR_SYSTEM_PROMPT
        return SELECTOR_SYSTEM_PROMPT
    """
    if str(dataset or "").strip().lower() == "triviaqa":
        return P.get("TRIVIAQA_SELECTOR_SYSTEM_PROMPT") or P.get("SELECTOR_SYSTEM_PROMPT", "")
    return P.get("SELECTOR_SYSTEM_PROMPT", "")


def _answer_system_prompt(P: dict, dataset: str) -> str:
    """官方 `generation/generation.py:78-82 get_answer_system_prompt` 的复刻（**仅作记录，不参与实验**）。

    官方按数据集分派（triviaqa 用 `TRIVIAQA_FORCE_ANSWER_PROMPT`）。本项目**生成侧统一**
    （`generate_answer` → `prompts/rag_prompt*.txt`），官方这条作答 prompt 不参与实验，
    保留此函数只为留下"官方原本怎么写"的出处。
    """
    if str(dataset or "").strip().lower() == "triviaqa":
        return P.get("TRIVIAQA_FORCE_ANSWER_PROMPT") or P.get("force_answer_prompt", "")
    return P.get("force_answer_prompt", "")


def _json_from_text(text: str, json_mode: bool):
    """等价官方 `safe_json_load`：先直接解析，再退化到抓第一个 {...}。"""
    if not text:
        return None
    if isinstance(text, dict):
        return text
    try:
        return json.loads(text)
    except Exception:
        pass
    m = _JSON_OBJ_RE.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def _split_sentences(text: str) -> list[str]:
    """等价官方 `split_wiki_sentences` 的**正则退化分支**（官方优先用 pysbd，未装则用这个）。"""
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []
    out = []
    for para in re.split(r"\n\s*\n+", text):
        para = para.strip()
        if not para:
            continue
        out.extend(s.strip() for s in _SENT_SPLIT_RE.split(para) if s and s.strip())
    return out


def _append_evidence(previous: str, new_text: str, sep: str = "\n\n---\n\n") -> str:
    """官方 `append_evidence_context`：新块已在旧上下文里就跳过（去重）。"""
    previous, new_text = (previous or "").strip(), (new_text or "").strip()
    if not new_text:
        return previous
    if not previous:
        return new_text
    if new_text in previous:
        return previous
    return previous + sep + new_text


class S2GRAG(BaseMethod):
    name = "s2grag"

    def __init__(self):
        self.dataset = ""
        self._index: CorpusIndex | None = None
        self._idx_by_level: dict = {}
        self._P: dict = {}          # 官方 prompt 常量

    # ── 语料级准备 ──
    def build(self, corpus_contexts: list[dict], working_dir: str = "") -> "S2GRAG":
        repo = self._require_official()
        self._P = _load_official_prompts(repo)
        self._index = CorpusIndex.from_contexts(corpus_contexts,
                                                dataset=getattr(self, "dataset", ""))
        # ★ 只在检索真要用向量时才建稠密向量。
        #   官方 S2G-RAG 的检索底座是 pyserini BM25（纯稀疏、不嵌语料），本项目默认
        #   `S2GRAG_RETRIEVAL_MODE=bm25` —— 无条件 build_dense() 会白花一次全语料 embedding
        #   （实测 hotpotqa/500 题 ≈ 70.6 万 embed token / 177 次调用 / 30s），
        #   而且把成本账面污染成"这个方法建图需要 embedding"，与官方口径不符。
        #   `mode="bm25"` 在 CorpusIndex.search 里只走 sparse_scores，完全不碰 vecs。
        if S2GRAG_RETRIEVAL_MODE in ("dense", "rrf"):
            self._index.build_dense()
        self._built_dense = S2GRAG_RETRIEVAL_MODE in ("dense", "rrf")
        print(f"[s2grag] 检索底座 {len(self._index)} 块（检索模式 {S2GRAG_RETRIEVAL_MODE}"
              f"{'，已建稠密向量' if self._built_dense else '，纯 BM25 不建向量'}）；"
              f"max_turns={S2GRAG_MAX_TURNS}、"
              f"top_docs={S2GRAG_TOP_DOCS}、selector≤{S2GRAG_SELECT_TOP_K} 句/轮；"
              f"作答=统一 generate_answer"
              f"（prompt-level 复现，未用官方训练的 judge）", flush=True)
        return self

    def _require_official(self) -> Path:
        return require_repo(S2GRAG_REPO_CANDIDATES, self.name,
                            "官方仓库（inference/ + utils/prompt_template.py）",
                            "把官方仓库整个目录放到 baselines/S2G-RAG-main/（含 .py）")

    # ── ① S2G-Judge：只看问题 + 累积证据 ──
    def judge(self, question: str, evidence_context: str) -> tuple[bool, list]:
        user = (f"QUESTION:\n{question}\n\nCONTEXT:\n{evidence_context}\n\n"
                f"Please ONLY use the above CONTEXT as evidence.\n"
                f"Decide whether it is sufficient, and if not, list the missing facts\n"
                f"in the required JSON format.")
        with llm_role("judge"):
            out = llm_chat(self._P["SUFF_SYSTEM_PROMPT"], user,
                           json_mode=S2GRAG_JUDGE_JSON_MODE, timeout=120,
                           max_tokens=S2GRAG_JUDGE_MAX_TOKENS)
        parsed = _json_from_text(out if isinstance(out, str) else json.dumps(out or {}),
                                 S2GRAG_JUDGE_JSON_MODE)
        if not isinstance(parsed, dict):
            return False, []          # 官方：解析失败 → verdict=False（保守：继续检索）
        sufficient = bool(parsed.get("sufficient", False))
        gaps = parsed.get("gap_items") or parsed.get("gap items") \
            or parsed.get("missing_facts") or []
        if sufficient or not isinstance(gaps, list):
            gaps = []
        return sufficient, gaps

    # ── ② gap → 下一轮 query（官方 build_query_from_missing，逐字对齐）──
    def build_query(self, question: str, gaps: list) -> str:
        max_facts = S2GRAG_MAX_FACTS
        if not gaps:
            return question
        phrases = []
        for item in gaps[:max_facts]:
            if not isinstance(item, dict):
                continue
            target = str(item.get("target") or "").strip()
            slot = str(item.get("slot") or "").strip()
            desc = str(item.get("description") or "").strip()
            if target and slot:
                phrases.append(f"{target} {slot}")
            elif desc:
                phrases.append(desc)
        return question if not phrases else question + " " + " ".join(phrases)

    # ── ③ 句子级 selector（官方 numbering + 映射 + 输出解析）──
    def select_sentences(self, question: str, gaps: list, hits: list[int]):
        """返回 [(hit_idx, [句号...])]，同时给出编号到 (hit_idx, 句号) 的映射。"""
        sents, id_map, per_doc = [], {}, {}
        for doc_idx, hi in enumerate(hits):
            parts = _split_sentences(self._index.units[hi]["text"])[:S2GRAG_MAX_SENTS_PER_DOC]
            per_doc[doc_idx] = []
            for local_sid, s in enumerate(parts, start=1):
                title = self._index.units[hi]["title"] or "N/A"
                sents.append((doc_idx, local_sid, title, s))
        if not sents:
            return [], {}
        numbered = []
        for gid, (doc_idx, local_sid, title, s) in enumerate(sents, start=1):
            numbered.append(f"[{gid}] ({title} | s#{local_sid}) {s}")
            id_map[gid] = (doc_idx, local_sid)

        hints = self._format_gaps(gaps)
        user = (f"ORIGINAL QUESTION:\n{question}\n\n"
                f"MISSING FACTS TO FILL:\n{hints}\n\n"
                f"NUMBERED SENTENCES FROM RETRIEVED DOCUMENTS:\n"
                + "\n".join(numbered)
                + f"\n\nYou may select up to {S2GRAG_SELECT_TOP_K} sentences.\n"
                + 'Return ONLY JSON with "evidence_global_ids".')
        with llm_role("selector"):
            # ★ 走官方按数据集分派（triviaqa → TRIVIAQA_SELECTOR_SYSTEM_PROMPT）。
            #   以前这里硬用通用 SELECTOR_SYSTEM_PROMPT，**把官方的分派逻辑整个绕过了**
            #   —— 而 selector 属检索侧，按本项目原则必须逐字复刻官方。
            out = llm_chat(_selector_system_prompt(self._P, self.dataset), user,
                           json_mode=True,
                           timeout=120, max_tokens=S2GRAG_SELECT_MAX_TOKENS)
        parsed = _json_from_text(out if isinstance(out, str) else json.dumps(out or {}), True)
        ids = []
        if isinstance(parsed, dict):
            vals = parsed.get("evidence_global_ids") or parsed.get("evidence global ids") or []
            for v in vals if isinstance(vals, list) else []:
                try:
                    v = int(v)
                    if v >= 1:
                        ids.append(v)
                except Exception:
                    pass
        for gid in ids:
            if gid in id_map:
                doc_idx, local_sid = id_map[gid]
                per_doc[doc_idx].append(local_sid)
        return [(i, sorted(set(v))) for i, v in per_doc.items() if v], id_map

    def _format_gaps(self, gaps: list, max_facts: int = 3) -> str:
        if not gaps:
            return "None."
        lines = []
        for item in gaps[:max_facts]:
            if not isinstance(item, dict):
                continue
            category = str(item.get("category") or "").strip()
            target = str(item.get("target") or "").strip()
            slot = str(item.get("slot") or "").strip()
            desc = str(item.get("description") or "").strip()
            core = f"{target} {slot}".strip() if target and slot else desc
            if core:
                lines.append(f"- [{category or 'other'}] {core}")
        return "\n".join(lines) if lines else "None."

    # ── ④ 证据合并（官方 merge_evidence_only）──
    def merge_evidence(self, selected, hits: list[int]) -> str:
        blocks = []
        for doc_idx, sids in selected:
            hi = hits[doc_idx]
            parts = _split_sentences(self._index.units[hi]["text"])[:S2GRAG_MAX_SENTS_PER_DOC]
            chosen = [parts[i - 1] for i in sids if 1 <= i <= len(parts)]
            if chosen:
                title = self._index.units[hi]["title"] or "N/A"
                blocks.append(f"[{title}] EVIDENCE:\n- " + "\n- ".join(chosen))
        return "\n\n---\n\n".join(blocks)

    # ── 主流程（官方 main_batch 的单题版）──
    def query(self, question: str) -> dict:
        if self._index is None:
            raise RuntimeError("请先 build()")
        evidence_ctx = ""       # judge 只看它
        past: set = set()
        picked_hits: list[int] = []
        trace = []
        # ★ 每轮单独一条（`common/trace.py:334` 里 `trace_rounds` 优先级最高）。
        #   以前只返回 `sufficiency_trace`（仅 turn/sufficient/n_gaps），而
        #   `rounds_from_result` **根本不认这个字段** → 每轮检索到什么全丢了。
        trace_rounds: list[dict] = []

        for turn in range(S2GRAG_MAX_TURNS + 1):
            sufficient, gaps = self.judge(question, evidence_ctx)
            # 官方：首轮空证据 + judge 过度自信 → 强制检索一次
            if turn == 0 and sufficient and not evidence_ctx.strip():
                sufficient, gaps = False, []
            trace.append({"turn": turn, "sufficient": bool(sufficient), "n_gaps": len(gaps)})
            if sufficient or turn == S2GRAG_MAX_TURNS:
                break

            q = self.build_query(question, gaps)
            hits = self._index.search(q, S2GRAG_TOP_DOCS, mode=S2GRAG_RETRIEVAL_MODE,
                                      exclude=past, bm25_k1=S2GRAG_BM25_K1,
                                      bm25_b=S2GRAG_BM25_B)
            if not hits:
                break
            past.update(hits)
            picked_hits.extend(hits)

            selected, _id_map = self.select_sentences(question, gaps, hits)
            block = self.merge_evidence(selected, hits)

            # 每轮记录：judge 给的 gaps（= 生成下轮 query 的缺失事实）、本轮 query、
            # 本轮检索到的块、本轮选进证据的句子
            trace_rounds.append({
                "round": turn, "query_used": q,
                "chunks": [{"text": self._index.units[i]["text"],
                            "title": self._index.units[i]["title"]} for i in hits],
                "entities": [], "paths": [],
                "sub_questions": [str(g) for g in gaps],
                "selected": selected if isinstance(selected, list) else [],
                "block": (block or "")[:2000],
            })
            new_ctx = _append_evidence(evidence_ctx, block)
            if new_ctx != evidence_ctx:
                evidence_ctx = new_ctx

        # 统一作答（唯一路径）：官方 `force_answer_prompt` 对照路径已删除
        answer = generate_answer(question, [{"text": evidence_ctx}] if evidence_ctx else [],
                                 dataset=getattr(self, "dataset", ""))

        return {"answer": answer,
                "evidence": [evidence_ctx] if evidence_ctx else [],
                "evidence_locs": self._index.locs(picked_hits),
                "turns_used": len(trace),
                "sufficiency_trace": trace,
                "trace_rounds": trace_rounds,
                "retrieved_blocks": len(picked_hits)}

    def run(self, question, contexts):
        raise NotImplementedError("S2G-RAG 走语料级：先 build(全部上下文) 再逐题 query")
