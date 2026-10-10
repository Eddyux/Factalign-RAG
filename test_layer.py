"""test_layer.py — 本方法的实验台（消融 / 因子分析），语料级复用已建语料图。

两类实验用 `--exp` 分目录，产物互不干扰：

  因子分析（--exp factor）：**只换查询形态**，其余全默认
      --decompose                          实例化疑问句（NEXT_PROMPT）
      --decompose --decompose_desc         描述式（the mother of X）
      --decompose --decompose_stmt         陈述式/开放槽（"X 的父亲是 ___"）← 主方法口径

  消融（--exp ablation，默认）
      --decompose --decompose_stmt --decompose_recall proposition   主方法 full
      --decompose --decompose_stmt --decompose_recall chunk         只换检索单元（原文窗口）
      --recall proposition                                          单轮原问题检索命题（不拆解/不改写）

主方法流程（= `methods/factalign.py` 的生产口径）：
  1. 拆解循环（≤ `--decomp_steps` 步）：子查询 → 全库命题 dense 召回 top `--decomp_top_k`（池只增不减）
  2. LLM 判定「是否已能直接回答」；不够 → 按当前查询形态生成下一个子查询 → 回到 1
  3. 池内精排：dense（`--rerank dense`）或 0.7×dense + 0.3×实体描述（`hybrid`）
  4. 跨跳融合：`final = max(final, normalize(子查询命中分))`
  5. 生成：top `FACTALIGN_EVIDENCE_PROPS`(20) 条命题 + 去重后前 `FACTALIGN_EVIDENCE_WINDOWS`(5) 个原文窗口

产物（`{exp}` = `--exp`，`{tag}` = `--tag`）：
  logs/ours/{exp}/{dataset}_{sample}[_{tag}]_{时间戳}.log     过程日志（终端同步双写）
  logs/ours/{exp}/…_{时间戳}.llm_calls.jsonl                   逐条 LLM 调用审计（本 run 独立一份）
  results/ours/{exp}/test_layer_{tag}_{dataset}.jsonl         逐题结果（em/f1/evidence/token 分解）
  results/ours/{exp}/ourssummary.jsonl                        每次 run 一行汇总
  traces/ours/{exp}[/{tag}]/{dataset}/v{n}r{m}_n{sample}/     逐题 trace（prompt + 原始输出 + 每轮检索）

用法:
  python test_layer.py --dataset 2wiki --sample 500 --workers 16 --rerank dense \\
      --exp factor --tag qf_stmt --decompose --decompose_stmt

★ 语料图：默认**只读加载** data/corpus/{dataset}/v{graph-version}，绝不重建/覆盖
  （加载失败即退出并打印原因）；确需重建才加 `--allow-build`。
"""

import argparse
import json
import math
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from common.data import iter_questions
from common.llm import (EMBED_DIM, phase, embed_batch, embed as llm_embed, chat as llm_chat,
                        snapshot, diff_snapshot, get_stats, reset_stats)
from common.metrics import exact_match_aliases, f1_score_aliases
from methods.base import generate_answer
from corpus import get_or_build, merged_contexts, _corpus_dir
from methods.standard_rag import build_chunks
from config import RESULTS_DIR
# ★ 兜底预算**镜像生产路径的 factalign**：命题条数用 `FACTALIGN_EVIDENCE_PROPS`、
#   窗口用 `FACTALIGN_EVIDENCE_WINDOWS`。原来这里兜底到 `SEARCH_TOP_K`(=20)，
#   那个旋钮已删除（生产路径无人消费）；直接 import 生产用的常量，工具与生产就不会漂。
from methods.factalign import (FACTALIGN_EVIDENCE_PROPS as _FA_PROPS,
                               FACTALIGN_EVIDENCE_WINDOWS as _FA_WINDOWS)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _safe_name(s: str) -> str:
    """文件/目录名安全化：消融 tag 会进 trace 目录名，带 `/` 会造出子目录、带空格会造怪名。

    与 `common/paths.question_trace_path` 对 qid 的处理同一口径（只用字母数字和 `-_.`）。
    """
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(s or "")) or "run"


def _bm25(query: str, docs: list[list[str]], k1=1.5, b=0.75) -> list[float]:
    N = len(docs)
    if N == 0:
        return []
    avgdl = sum(len(d) for d in docs) / N or 1.0
    df = {}
    for d in docs:
        for t in set(d):
            df[t] = df.get(t, 0) + 1
    q_terms = set(_tokenize(query))
    scores = []
    for d in docs:
        tf = {}
        for t in d:
            tf[t] = tf.get(t, 0) + 1
        s = 0.0
        for t in q_terms:
            if t in tf:
                idf = math.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * (tf[t] * (k1 + 1)) / (tf[t] + k1 * (1 - b + b * len(d) / avgdl))
        scores.append(s)
    return scores


def cosine(a, b) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


NEXT_SYSTEM = "You plan the next retrieval step for a multi-hop QA system."
NEXT_PROMPT = """Question: {question}

Evidence gathered so far:
{evidence}

Judge strictly, step by step:
1) Does any evidence fact DIRECTLY state the answer to the question?
   Example: for "who is X's maternal grandfather", ONLY a fact like
   "Y is X's maternal grandfather" counts as a direct answer.
   A fact like "X was the son of Z" does NOT answer it — it only gives an intermediate link.
2) If NOT directly answered:
   a. Read the INTERMEDIATE ENTITY'S NAME from the evidence.
      (e.g. evidence says "X was the son of Antiochus IX and his wife Cleopatra IV"
       → the intermediate entity is "Cleopatra IV".)
   b. Write the next sub-question USING THAT NAME, e.g. "Who is the father of Cleopatra IV?"
      NEVER write a description like "the mother of X" when the evidence gives the name —
      such a description retrieves only generic facts about mothers, not facts about THIS entity.
   c. Only if the evidence truly does not give the name, describe it.

Return ONLY JSON: {{"direct_answer": false, "next_question": "Who is the father of Cleopatra IV?"}}"""

# 消融用：描述式子问题变体（对照"实例化子问题"，证明中间实体名的必要性）
NEXT_PROMPT_DESC = """Question: {question}

Evidence gathered so far:
{evidence}

Judge strictly:
1) Does any evidence fact DIRECTLY state the answer to the question? Only a fact that
   itself names the answer counts.
2) If NOT directly answered, what is the SINGLE next sub-question to look up?
   Describe the intermediate entity WITHOUT naming it, e.g. "the mother of Antiochus X".

Return ONLY JSON: {{"direct_answer": false, "next_question": "the mother of Antiochus X"}}"""


# 变体 3：陈述式——把下一步查询写成**事实单元的样子**（自包含事实句、未知处留空），
# 与检索单元（命题=自包含事实句）形态对齐，而不是疑问句/名词短语
NEXT_PROMPT_STMT = """Question: {question}

Evidence gathered so far:
{evidence}

Judge strictly:
1) Does any evidence fact DIRECTLY state the answer to the question? Only a fact that
   itself names the answer counts.
2) If NOT directly answered: the knowledge base stores SELF-CONTAINED FACT sentences, so
   write what you need to look up AS A FACT SENTENCE — the sentence you hope to find,
   with the unknown part left as ___. Reuse the names that appear in the evidence.
   Example: evidence says "X was the son of Antiochus IX and his wife Cleopatra IV" and
   you need Cleopatra IV's father → write "Cleopatra IV's father is ___".
   If the evidence does not name the intermediate entity, put a short description in its
   place, e.g. "the father of Antiochus X's mother is ___".

Return ONLY JSON: {{"direct_answer": false, "next_question": "Cleopatra IV's father is ___"}}"""


def _load_corpus_only(dataset: str, sample: int, graph_version: int):
    """只加载已建好的语料图（test_layer 默认口径）：**绝不重建、绝不覆盖**。

    `corpus.get_or_build` 的语义是"读不到就在同一个 v 目录上重建并覆盖"——meta 里的
    题数 / 切块口径 / embed 指纹 / EXTRACT_VERSION / chat 模型任一不符都会触发
    （corpus.py:161-176）。消融台跑得频繁、又常换 --tag，参数一旦写错就会把手上那张图
    原地换掉且没有备份，所以这里改成"加载失败就带原因退出"，重建必须显式 --allow-build。
    """
    from corpus import load_corpus
    cdir = _corpus_dir(dataset, sample, graph_version)
    try:
        out = load_corpus(dataset, sample, graph_version)
    except FileNotFoundError as e:
        raise SystemExit(
            f"\n[corpus] ✗ 没有可用的语料图，且默认**不重建**（只有 --allow-build 才允许）：\n"
            f"  目录: {cdir}\n"
            f"  原因: {e}\n"
            f"  处理：\n"
            f"    · 想用已建好的图 → 让 --dataset/--sample/--graph-version 与建图时一致\n"
            f"      （建图参数记在该目录的 meta.json 里：sample / 切块口径 / embed 指纹）\n"
            f"    · 确认要重建 → 加 --allow-build（会**覆盖**同一个 v 目录里的图）\n"
            f"    · 想保留旧图再建新的 → 换 --graph-version（如 2）")
    print(f"[corpus] 载入已建语料图（只读，不重建）: {cdir}", flush=True)
    return out


def main():
    _wall0 = time.time()          # 本次 run 的墙钟起点（进 results/ours/{exp}/ourssummary.jsonl）
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="只测前 N 题（0=全部）")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--graph-version", type=int, default=1,
                    help="语料图版本（data/corpus/{dataset}/v{n}）：换 chat 模型/建图参数时开新 v 保留旧图")
    ap.add_argument("--allow-build", action="store_true",
                    help="[默认关] 允许语料图缺失/校验不过时**重建并覆盖** data/corpus/{ds}/v{n}。"
                         "默认只读加载已建好的图，加载失败即退出（防手滑把图覆盖掉）")
    ap.add_argument("--retrieval", type=int, default=1,
                    help="同一张图上的第几次查询（只进 trace/results 的目录名 v{n}r{m}，默认 1；"
                         "同一 tag 换查询参数想另存一份 trace 就换 r）")
    ap.add_argument("--no-trace", dest="trace", action="store_false", default=True,
                    help="关掉逐题 trace（**默认开**，同 run.py）："
                         "traces/ours/{exp}[/{tag}]/{dataset}/v{n}r{m}_n{sample}/{qid}.json"
                         " —— 存每次 LLM 的"
                         "真实 prompt + 模型原始输出，以及每轮拆解检索的查询与命中")
    ap.add_argument("--no-trace-prompt", dest="trace_prompt", action="store_false", default=True,
                    help="trace 里 system/assistant 消息只存 sha256（默认存原文）；体积约降到 1/4")
    ap.add_argument("--recall_chunks", type=int, default=20, help="窗口模式：RRF/BM25/dense 召回的窗口数")
    ap.add_argument("--hyper_recall", type=int, default=60,
                    help="proposition 模式：全量命题语义召回的候选数（给精排留空间）")
    ap.add_argument("--debug", type=int, default=0, help="打印前 N 题的候选命题明细（建议配 --workers 1）")
    ap.add_argument("--gen_top", type=int, default=0,
                    help="喂给生成的命题数（0=用 FACTALIGN_EVIDENCE_PROPS=20）。加大能带上排序靠后的桥命题"
                         "——命题多对一映射到窗口，生成端去重后仍只取前 10 窗口")
    ap.add_argument("--decompose", action="store_true",
                    help="拆解子问题检索（IRCoT 风格）：检索 → 判据不足则生成下一个子问题"
                         "（自包含、可引用中间实体）→ 再检索 → 合并证据 → 精排生成")
    ap.add_argument("--decomp_top_k", type=int, default=5, help="每个子问题检索的命题数")
    ap.add_argument("--decomp_steps", type=int, default=4, help="最多拆解/检索步数")
    ap.add_argument("--decompose_desc", action="store_true",
                    help="[消融] 子问题用「描述式」（the mother of X）而非「实例化」（用证据里的实体名）")
    ap.add_argument("--decompose_stmt", action="store_true",
                    help="[消融] 子问题用「陈述式」：写成事实单元的样子（'X 的父亲是 ___'），"
                         "与检索单元（命题=自包含事实句）形态对齐；优先于 --decompose_desc")
    ap.add_argument("--decompose_recall", default="proposition", choices=["proposition", "chunk"],
                    help="[消融] 拆解时检索什么：proposition=命题（默认）；chunk=原文窗口（证明事实级检索的必要性）")
    ap.add_argument("--tag", default="",
                    help="结果文件名标签（消融用，避免不同配置互相覆盖）")
    ap.add_argument("--exp", default="ablation",
                    help="实验类别（只进目录名，用来把不同实验分开）："
                         "results/ours/{exp}/… 与 traces/ours/{exp}/…；"
                         "默认 ablation；查询形态（因子）那组用 --exp factor")
    ap.add_argument("--recall", type=str, default="rrf", choices=["rrf", "bm25", "dense", "proposition"],
                    help="rrf=chunk dense+BM25 RRF（同 standardrag）；bm25=纯 BM25；"
                         "dense=纯 chunk dense；proposition=全量命题语义召回（单命题 dense，无窗口层）")
    ap.add_argument("--rerank", type=str, default="hybrid",
                    choices=["none", "dense", "hybrid"],
                    help="none=不精排直接按召回序；dense=纯语义精排；"
                         "hybrid=dense+实体描述侧面")
    args = ap.parse_args()

    # 分方法日志：logs/ours/{exp}/{dataset}_{sample}[_{tag}]_{时间戳}.log（终端同步双写）
    # ★ 实验类别进目录（--exp）：消融 与 因子分析 的日志也分开。
    _exp = _safe_name(args.exp)
    from common.log_setup import setup_log
    _log_path = setup_log(f"ours/{_exp}", args.dataset, args.sample,
                          extra=args.tag or f"{args.recall}_{args.rerank}")

    # ★ 逐条 LLM 调用审计：**本 run 独立一个文件**（与 run.py 同款命名），不再全塞进共享的
    #   `logs/llm_calls.jsonl`——文件名 = 过程日志同名 + `.llm_calls.jsonl`，自带
    #   数据集/题数/tag，所以「按数据集、按每个配置」天然分开。
    #   只有 `LLM_CALL_LOG=0`（false/off/no）能关掉；**其它取值一律被本 run 的文件取代**
    #   ——包括历史上 export 过的 `LLM_CALL_LOG=<路径>`：留着它会静默让所有 run 又写回同一个
    #   jsonl（真踩过），所以这里不再"尊重"路径，只在打印里说明覆盖了谁。
    try:
        from common.llm import LLM_CALL_LOG_ON, set_call_log_path
        _clv = os.environ.get("LLM_CALL_LOG", "1").strip()
        if LLM_CALL_LOG_ON:
            _call_log = _log_path.with_suffix(".llm_calls.jsonl")
            set_call_log_path(_call_log)
            _note_old = ("" if _clv.lower() in ("", "1", "true", "yes", "on")
                         else f"（已覆盖 LLM_CALL_LOG={_clv!r}）")
            print(f"[llm] 逐条调用审计（本 run 独立文件）→ {_call_log}{_note_old}", flush=True)
        else:
            print("[llm] 逐条调用审计已关闭（LLM_CALL_LOG=0）", flush=True)
    except Exception as e:                                   # noqa: BLE001
        print(f"[llm] ⚠️ 未切换调用审计路径（不影响实验）：{type(e).__name__}: {e}",
              flush=True)

    items = list(iter_questions(args.dataset, args.sample))
    if args.limit > 0:
        items = items[: args.limit]

    # 结果**路径**先算出来，但文件等语料图载入成功后再动（旧顺序：先 unlink 再建图，
    # 图校验不过退出时上一次的结果文件已被删掉）。
    # ★ 全部落到 `results/ours/{exp}/`（与 `logs/ours/{exp}/` 对齐）：
    #   results/ours/{exp}/test_layer_{tag}_{dataset}.jsonl   —— 逐题
    #   results/ours/{exp}/ourssummary.jsonl                  —— 每次 run 一行
    #   连带影响：`tools/paired_significance.py` 默认在 `results/` 下找，需要
    #   `--results-dir results/ours/{exp}`；`tools/compare_runs.py` 吃显式路径，不受影响。
    _ours_dir = RESULTS_DIR / "ours" / _exp
    _stem = (f"test_layer_{args.tag}_{args.dataset}" if args.tag
             else f"test_layer_{args.recall}_{args.rerank}_{args.dataset}")
    results_path = _ours_dir / f"{_stem}.jsonl"

    reset_stats()
    with phase("build"):
        # ★ 默认**只读加载**已建语料图，绝不重建：`get_or_build` 在 meta 校验不过时
        #   会**在同一个 v 目录上重建覆盖**（题数/切块口径/embed 指纹/EXTRACT_VERSION/
        #   chat 模型任一不符）——手滑把 --sample 写错就把手上的图换掉且没有备份。
        #   要重建必须显式 `--allow-build`。
        _load = get_or_build if args.allow_build else _load_corpus_only
        entities, propositions, chunks, pvecs = _load(args.dataset, args.sample,
                                                           args.graph_version)
        # 语料图句子索引：(title, sent_idx) -> proposition_ids
        sent_props: dict[tuple, list[str]] = {}
        for c in chunks:
            sent_props.setdefault((c["title"], c["sent_idx"]), []).extend(c.get("proposition_ids", []))
        # 命题 id -> 记录
        prop_by_id = {h["id"]: h for h in propositions}
        # 实体描述 embedding（语料级一次；缓存在语料图目录，复用不重算）
        ent_vecs = {}
        cdir = _corpus_dir(args.dataset, args.sample, args.graph_version)
        ent_cache = cdir / "ent_desc_vecs.npy"
        ent_ids_cache = cdir / "ent_desc_ids.json"
        if ent_cache.exists() and ent_ids_cache.exists():
            _ids = json.loads(ent_ids_cache.read_text(encoding="utf-8"))
            _v = np.load(str(ent_cache))
            for i, eid in enumerate(_ids):
                ent_vecs[eid] = _v[i]
        else:
            _texts = []
            _ids = []
            for e in entities:
                desc = (e.get("description") or "").strip()
                if desc:
                    _texts.append(f"{e.get('name','')}: {desc}")
                    _ids.append(e["id"])
            if _texts:
                _v = embed_batch(_texts)
                for i, eid in enumerate(_ids):
                    ent_vecs[eid] = _v[i]
                np.save(str(ent_cache), _v.astype(np.float32))
                ent_ids_cache.write_text(json.dumps(_ids), encoding="utf-8")
                print(f"[build] 实体描述 embedding 已算并缓存: {len(_ids)} 条", flush=True)
            else:
                print("[build] 无实体描述", flush=True)
        # 窗口（与建图同口径 = 整段）：BM25 文档 = 窗口全文
        all_ctx = merged_contexts(args.dataset, args.sample)
        windows = build_chunks(all_ctx, dataset=args.dataset)
        # 窗口向量（召回 dense 用；缓存在语料图目录，复用不重算）
        wcache = cdir / "window_vecs.npy"
        if wcache.exists():
            win_vecs = np.load(str(wcache))
        else:
            win_vecs = embed_batch([w[2] for w in windows]) if windows else np.zeros((0, EMBED_DIM))
            np.save(str(wcache), win_vecs.astype(np.float32))
            print(f"[build] 窗口向量已算并缓存: {len(windows)} 条", flush=True)

    # 语料图载入成功后才动结果文件（每次全新运行覆盖旧文件 = 同 tag 重跑即替换）
    _ours_dir.mkdir(parents=True, exist_ok=True)
    results_path.unlink(missing_ok=True)
    fout = open(results_path, "w", encoding="utf-8")

    eidx_ok = len(ent_vecs)
    print(f"语料图: {len(entities)}e {len(propositions)}h | BM25 窗口: {len(windows)} | "
          f"实体描述数: {eidx_ok} | 测试 {len(items)} 题 | rerank={args.rerank}", flush=True)

    # BM25 词项化一次（窗口文档；窗口召回 bm25/rrf 用）
    win_toks = [_tokenize(w[2]) for w in windows]
    dbg_printed = 0

    def _answer_core(it, qv_pre=None):
        nonlocal dbg_printed
        q = it["question"]
        # 首跳 query 向量：批量预计算（见下面 qvecs）时直接用，否则退回单条 embed
        qv = llm_embed(q) if qv_pre is None else qv_pre

        cand_props: dict[str, dict] = {}
        decomp_log = []
        decomp_score: dict[str, float] = {}   # 命题 id -> 被子问题检索到的最高分
        decomp_wins: list[int] = []           # [消融] decompose_recall=chunk：累积的窗口下标
        if args.decompose:
            # ── 拆解子问题检索（IRCoT 风格）：检索 → 判不足 → 生成下一个子问题 → 再检索 ──
            _use_chunk = (args.decompose_recall == "chunk")
            _next_prompt = (NEXT_PROMPT_STMT if args.decompose_stmt else
                            NEXT_PROMPT_DESC if args.decompose_desc else NEXT_PROMPT)
            sub_q = q
            seen_subs = {sub_q.lower()}
            for step in range(1, args.decomp_steps + 1):
                # 第 1 轮的 sub_q 就是原问题 → 直接复用首跳向量，省一次 embed HTTP
                qv_s = qv if step == 1 else llm_embed(sub_q)
                added = 0
                hits = []
                if _use_chunk:
                    # [消融] 检索原文窗口（不做命题映射），生成时直接用窗口文本
                    wsims = win_vecs @ qv_s
                    for i in np.argsort(-wsims):
                        wi = int(i)
                        if wi in decomp_wins:
                            continue
                        decomp_wins.append(wi)
                        hits.append((windows[wi][2][:70], float(wsims[wi])))
                        added += 1
                        if added >= args.decomp_top_k:
                            break
                else:
                    sims = pvecs @ qv_s
                    for i in np.argsort(-sims):
                        h = propositions[int(i)]
                        if h["id"] in cand_props:
                            decomp_score[h["id"]] = max(decomp_score.get(h["id"], 0.0),
                                                        float(sims[int(i)]))
                            continue
                        cand_props[h["id"]] = h
                        decomp_score[h["id"]] = float(sims[int(i)])
                        hits.append((h["summary"][:70], float(sims[int(i)])))
                        added += 1
                        if added >= args.decomp_top_k:
                            break
                decomp_log.append(("SEARCH", sub_q, hits))
                if step == args.decomp_steps:
                    break
                if _use_chunk:
                    ev_lines = "\n".join(f"- {windows[wi][2][:200]}" for wi in decomp_wins[:15])
                else:
                    ev_lines = "\n".join(f"- {h['summary']}"
                                         for h in list(cand_props.values())[:15])
                nxt_out = llm_chat(NEXT_SYSTEM,
                                   _next_prompt.format(question=q, evidence=ev_lines),
                                   json_mode=True, timeout=90)
                if not nxt_out:
                    decomp_log.append(("STOP", "LLM 失败", []))
                    break
                if nxt_out.get("direct_answer"):
                    decomp_log.append(("ENOUGH", "", []))
                    break
                nxt = str(nxt_out.get("next_question") or "").strip()
                if not nxt:
                    decomp_log.append(("STOP", "没给出下一问", []))
                    break
                if nxt.lower() in seen_subs:
                    decomp_log.append(("STOP", f"子问题重复: {nxt[:50]}", []))
                    break
                seen_subs.add(nxt.lower())
                sub_q = nxt
        elif args.recall == "proposition":
            # 单命题 dense：全量命题语义召回 top-N（无窗口层；pvecs/qv 已归一化，内积=余弦）
            dense_all = pvecs @ qv
            for i in np.argsort(-dense_all)[: args.hyper_recall]:
                cand_props[propositions[i]["id"]] = propositions[i]
        else:
            # ── 召回层：窗口 dense / BM25 / RRF（同 standardrag query_chunks 口径）──
            bm25_scores = np.array(_bm25(q, win_toks), dtype=np.float32)
            dense = np.array([cosine(qv, v) for v in win_vecs], dtype=np.float32)

            if args.recall == "bm25":
                top_wins = sorted(range(len(windows)), key=lambda i: -bm25_scores[i])[: args.recall_chunks]
            elif args.recall == "dense":
                top_wins = sorted(range(len(windows)), key=lambda i: -dense[i])[: args.recall_chunks]
            else:  # rrf：dense + BM25 排名倒数融合（同 standardrag）
                kk = min(len(windows), args.recall_chunks)
                dense_rank = {i: r for r, i in
                              enumerate(np.argsort(-dense)[:len(windows)])}
                sparse_rank = {i: r for r, i in
                               enumerate(np.argsort(-bm25_scores)[:len(windows)])}
                fused = [(1.0 / (60 + dense_rank.get(i, len(windows)))
                          + 1.0 / (60 + sparse_rank.get(i, len(windows))), i)
                         for i in range(len(windows))]
                fused.sort(reverse=True)
                top_wins = [i for _, i in fused[:kk]]

            # 窗口 -> 关联命题
            for wi in top_wins:
                for title, si in windows[wi][3]:
                    for pid in sent_props.get((title, si), []):
                        if pid in prop_by_id:
                            cand_props[pid] = prop_by_id[pid]

        # [消融] decompose_recall=chunk：拆解检索的是原文窗口，直接用窗口文本生成
        if args.decompose and args.decompose_recall == "chunk":
            if not decomp_wins:
                _a = generate_answer(q, [], dataset=args.dataset)
                return it, _a, {"answer": _a, "evidence": [], "evidence_locs": [],
                                "decomp_log": decomp_log, "entities": []}
            # 精排（与命题模式对齐，保证单变量：只换检索单元）：按原问题语义分对累积窗口排序。
            # 取**生产路径同一把窗口旋钮**（`FACTALIGN_EVIDENCE_WINDOWS`，默认 5）——与命题
            # 模式生成端实际喂的窗口数对齐，保证「只换检索单元」这个单变量成立
            _w_idx = np.array(decomp_wins, dtype=np.int64)
            _wsem = win_vecs[_w_idx] @ qv
            _order = np.argsort(-_wsem)
            _ntop = args.gen_top if args.gen_top > 0 else _FA_WINDOWS
            _ev = [{"text": windows[decomp_wins[int(p)]][2]} for p in _order[:_ntop]]
            _ans = generate_answer(q, _ev, dataset=args.dataset)
            if args.debug > 0:
                with lock:
                    if dbg_printed < args.debug:
                        dbg_printed += 1
                        print(f"\n===== DEBUG 题 {dbg_printed} (chunk 消融) =====", flush=True)
                        print(f"Q: {q}", flush=True)
                        print(f"gold: {it['answer']!r}", flush=True)
                        print(f"证据: {len(decomp_wins)} 个原文窗口", flush=True)
                        for kind, sq, hits in decomp_log:
                            if kind in ("ENOUGH", "STOP"):
                                print(f"     → 停止拆解（{kind}）", flush=True)
                                continue
                            print(f"     检索: {sq}", flush=True)
                            for r, (sm, sc) in enumerate(hits, start=1):
                                print(f"       #{r} sem={sc:.3f} | {sm}", flush=True)
                        print(f"pred: {_ans!r}", flush=True)
            return it, _ans, {"answer": _ans,
                              "evidence": [e["text"] for e in _ev],
                              "evidence_locs": [], "decomp_log": decomp_log,
                              "entities": []}

        cand_list = list(cand_props.values())
        if not cand_list:
            _a = generate_answer(q, [], dataset=args.dataset)
            return it, _a, {"answer": _a, "evidence": [], "evidence_locs": [],
                            "decomp_log": decomp_log, "entities": []}

        # 候选池内精排：每个候选命题的 summary 语义分
        def _norm(x):
            m = float(x.max()) if len(x) else 0.0
            return x / m if m > 0 else np.zeros_like(x)

        idx_of = {h["id"]: i for i, h in enumerate(propositions)}
        sem = np.array([cosine(qv, pvecs[idx_of[h["id"]]]) for h in cand_list],
                       dtype=np.float32)
        final = _norm(sem)
        if args.rerank == "hybrid":
            ent_scores = np.zeros(len(cand_list), dtype=np.float32)
            for ci, h in enumerate(cand_list):
                best = 0.0
                for eid in h.get("entity_ids", []):
                    v = ent_vecs.get(eid)
                    if v is not None:
                        s = cosine(qv, v)
                        if s > best:
                            best = s
                ent_scores[ci] = best
            final = 0.7 * final + 0.3 * _norm(ent_scores)

        if args.decompose and decomp_score:
            # 拆解模式（必须放在所有 rerank 融合之后）：被「实例化子问题」强匹配到的桥命题，
            # 用它的子问题检索分兜底 —— 桥命题讲的是中间实体，对原问题 sem 低、实体分也低，
            # 不兜底就会被压到排序后面。
            dsc = np.array([decomp_score.get(h["id"], 0.0) for h in cand_list],
                           dtype=np.float32)
            final = np.maximum(final, _norm(dsc))

        if args.rerank == "none":
            # 不精排：候选按召回顺序（cand_list 插入序 = top_wins 序）
            order = np.arange(len(cand_list))
        else:
            order = np.argsort(-final)
        # 取前 gen_top 命题喂生成（0=用 FACTALIGN_EVIDENCE_PROPS；generate_answer 对 window_text
        # 去重后取前 FACTALIGN_EVIDENCE_WINDOWS 窗口 —— 命题多对一窗口，多喂命题能让排序
        # 靠后的桥窗口进入窗口预算内）
        _ntop = args.gen_top if args.gen_top > 0 else _FA_PROPS
        top = [cand_list[i] for i in order[: _ntop]]
        # ★ 窗口预算显式传 `FACTALIGN_EVIDENCE_WINDOWS`（= `factalign._query` 里 `generate_answer`
        #   传的同一把旋钮）：不写就走 `base.EVIDENCE_WINDOWS`——两个环境变量默认都是 5，
        #   但**各自可被改**，改一个就会出现"test_layer 与 run.py --method factalign 喂的
        #   窗口数不同"的静默漂移（结果不可比且没有任何报错）。
        ans = generate_answer(q, top, windows=_FA_WINDOWS, dataset=args.dataset)

        # debug：打印前 N 题检索明细（建议 --workers 1 保证顺序）
        if args.debug > 0:
            with lock:
                if dbg_printed < args.debug:
                    dbg_printed += 1
                    print(f"\n===== DEBUG 题 {dbg_printed} =====", flush=True)
                    print(f"Q: {q}", flush=True)
                    print(f"gold: {it['answer']!r}", flush=True)
                    print(f"候选命题池: {len(cand_list)} 条 | recall={args.recall} "
                          f"rerank={args.rerank}", flush=True)
                    if decomp_log:
                        print("  ── 拆解+检索步骤 ──", flush=True)
                        for kind, sq, hits in decomp_log:
                            if kind in ("ENOUGH", "STOP"):
                                print(f"     → 停止拆解（{kind}{': ' + sq if sq else ''}）",
                                      flush=True)
                                continue
                            print(f"     检索: {sq}", flush=True)
                            for r, (sm, sc) in enumerate(hits, start=1):
                                print(f"       #{r} sem={sc:.3f} | {sm}", flush=True)
                    for rank, r in enumerate(order[: _FA_PROPS], start=1):
                        i = int(r)
                        h = cand_list[i]
                        print(f"  #{rank:<2} final={float(final[i]):.3f} sem={float(sem[i]):.3f}"
                              f" | {h.get('summary','')[:95]}", flush=True)
                    print(f"pred: {ans!r}", flush=True)
        # ★ 与 `factalign._result()` 同口径返回：trace 的 rounds_from_result 靠
        #   `decomp_log` 还原"每轮拆解检索了什么、命中哪几条"，靠 evidence 记最终证据。
        return it, ans, {"answer": ans,
                         "evidence": [h.get("window_text") or h.get("summary", "") for h in top],
                         "evidence_locs": [], "decomp_log": decomp_log, "entities": []}

    # —— 逐题打分 / 成本统计的容器（并发骨架沿用旧 test_ppr.py，该脚本已删）——
    # ★ 必须在**批量预计算之前**定义：预计算那段要往 `tok["embed"]` 累加 build/query
    #   两桶的 embed token。原来这几行定义在预计算**之后** → 那句累加必然抛
    #   UnboundLocalError，被 except 吞掉后打印成"批量预计算失败"（误导成网络问题），
    #   同时这批 token 永远进不了成本报表。`qvecs` 本身在抛错前已赋值成功，
    #   所以实际走的仍是批量快路径（只是被误报了）。
    stats = {"em": 0.0, "f1": 0.0, "n": 0}
    tok = {"prompt": 0, "completion": 0, "embed": 0, "seconds": 0.0, "calls": 0}
    lock = threading.Lock()
    total = len(items)
    done = 0

    # ── 逐题 trace（**默认开**，同 run.py）：把「每次 LLM 的真实 prompt + 模型原始输出」与
    #    「每一轮拆解检索的查询/命中」按题落盘。出口钩子在 `common/llm.py:_trace_out`
    #    （从**线程本地**取当前题），所以每题必须在**同一个 worker 线程**里 start/finish。
    # ★ 目录：`traces/ours/{exp}[/{tag}]/{dataset}/v{n}r{m}_n{sample}/` —— 实验类别分开、
    #   tag 再分开；**绝不能**用 `factalign` 当方法名：`trace_run_dir` 只按
    #   method/dataset/v{r} 分目录（common/paths.py:105），同名 qid 会直接**覆盖
    #   run.py 那份 factalign 的 trace**。
    _trace_method = f"ours/{_exp}" + (f"/{_safe_name(args.tag)}" if args.tag else "")
    _trace_dir = None
    if args.trace:
        try:
            from common.paths import trace_run_dir
            from common.trace import set_trace_dir, trace_run_header
            _trace_dir = trace_run_dir(_trace_method, args.dataset, args.graph_version,
                                       args.retrieval, sample=args.sample)
            trace_run_header(_trace_method, args.dataset, args.graph_version, args.retrieval,
                             sample=args.sample,
                             extra={"exp": args.exp, "tag": args.tag, "recall": args.recall,
                                    "rerank": args.rerank, "decompose": bool(args.decompose),
                                    "decompose_desc": bool(args.decompose_desc),
                                    "decompose_stmt": bool(args.decompose_stmt),
                                    "decompose_recall": args.decompose_recall,
                                    "workers": args.workers, "limit": args.limit,
                                    "argv": sys.argv})
            set_trace_dir(_trace_dir)      # 重复的长 prompt 只在 prompts/ 存一份
            print(f"[trace] 逐题留档（含推理原文）→ {_trace_dir}", flush=True)
        except Exception as e:                                   # noqa: BLE001
            print(f"[trace] ⚠️ 初始化失败（不影响实验）：{type(e).__name__}: {e}", flush=True)
            args.trace = False
            _trace_dir = None

    # —— 首跳 query 向量：**批量预计算**（别让每个线程各发一次小 HTTP）——
    # 每题现在只发 1 次 embed；不预计算的话，+拆解第 1 轮还会再发一次同样的向量。
    from common.llm import query_prefix, EMBED_DOC_PREFIX
    qvecs: list = [None] * len(items)
    try:
        _before_pre = snapshot()
        _qtexts = [query_prefix(it["question"]) for it in items]
        # ★ 对齐 factalign：`embed()` 是**查询侧**（只加 query_prefix），而 `embed_batch()`
        #   是**文档侧**接口、会再叠一层 `EMBED_DOC_PREFIX`。两者默认等价（该前缀为空），
        #   一旦非空，批量预计算出来的首跳 query 向量就与 factalign 不一致 → 明确放弃批量，
        #   退回落单条 `embed()`（下面 except 会打印原因）。
        if EMBED_DOC_PREFIX:
            raise RuntimeError(f"EMBED_DOC_PREFIX={EMBED_DOC_PREFIX!r} 非空 → 不做批量预计算")
        _qv = embed_batch(_qtexts) if _qtexts else np.zeros((0, 1), dtype=np.float32)
        qvecs = [np.asarray(v, dtype=np.float32) for v in _qv]
        _pre = diff_snapshot(_before_pre, snapshot())      # 别让这部分 token 掉出统计
        tok["embed"] += _pre["build"]["embed_tokens"] + _pre["query"]["embed_tokens"]
        print(f"[test_layer] 首跳 query 向量已批量预计算：{len(_qtexts)} 条"
              f"（省掉每题一次 embed HTTP；拆解第 1 轮直接复用）", flush=True)
    except Exception as e:                                   # noqa: BLE001
        print(f"[test_layer] 批量预计算失败（{type(e).__name__}: {e}）→ 退回每题单条 embed",
              flush=True)

    # 检索开始前报一次引擎状态：用来判断"到底谁在限流"
    try:
        from common.llm import vllm_engine_stats
        _st = vllm_engine_stats()
        if _st:
            _kv, _run, _wait = _st
            print(f"[test_layer] 检索开始：--workers={args.workers} | vLLM "
                  f"running={_run} waiting={_wait} KV={_kv:.1%}"
                  f"（waiting>0 = 引擎打满，加 --workers 无用；"
                  f"waiting=0 且 running<workers = 客户端没喂饱，可加 --workers）", flush=True)
    except Exception:                                        # noqa: BLE001
        pass

    def answer_one(idx, it):
        """包一层统计：单题 token/时间 = 线程内桶差值（build+query 合计）。

        逐题 trace 也在这里 start（与 run.py 同款）：本线程内所有 chat 调用会自动挂到这道题
        上（`common/llm.py:_trace_out` 的出口钩子），embed 不记。trace 出错绝不影响实验。
        """
        tr = None
        if args.trace:
            try:
                from common.trace import start_trace
                tr = start_trace(method=_trace_method, dataset=args.dataset, qid=it["id"],
                                 sample=args.sample, graph_version=args.graph_version,
                                 retrieval=args.retrieval, question=it["question"],
                                 gold=it.get("answer", ""),
                                 aliases=it.get("answer_aliases", []),
                                 keep_prompt=args.trace_prompt)
            except Exception as e:                               # noqa: BLE001
                print(f"[trace] ⚠️ start 失败（{it.get('id')}，不影响实验）：{e}", flush=True)
                tr = None
        before = snapshot()
        it2, ans, res = _answer_core(it, qvecs[idx] if idx < len(qvecs) else None)
        usage = diff_snapshot(before, snapshot())
        with lock:
            tok["prompt"] += usage["build"]["prompt_tokens"] + usage["query"]["prompt_tokens"]
            tok["completion"] += (usage["build"]["completion_tokens"]
                                  + usage["query"]["completion_tokens"])
            tok["embed"] += usage["build"]["embed_tokens"] + usage["query"]["embed_tokens"]
            tok["seconds"] += (usage["build"]["chat_seconds"]
                               + usage["query"]["chat_seconds"])
            tok["calls"] += usage["build"]["chat_calls"] + usage["query"]["chat_calls"]
        if tr is not None:
            _emit_trace(tr, it2, res, ans)
        return it2, ans, usage, res

    def _emit_trace(tr, it, res, ans):
        """把这道题的 trace 落盘（含每轮拆解检索）并写索引行；trace 出错不影响实验。

        `rounds_from_result` 认 factalign/test_layer 的 `decomp_log`
        （`[("SEARCH", 子查询, [(命题前70字, 相似度), …]), ("ENOUGH"/"STOP", …)]`），
        所以「哪一轮查了什么、命中哪几条、第几轮判定够了」会像 factalign 一样进 trace。
        """
        try:
            from common.trace import (normalize_result, rounds_from_result, finish_trace,
                                      clear_trace)
            nr = normalize_result(res)
            for rd in rounds_from_result(res, question=it["question"]):
                tr.add_round(round_idx=rd.get("round"), query_used=rd.get("query_used", ""),
                             chunks=rd.get("chunks"), entities=rd.get("entities"),
                             paths=rd.get("paths"), sub_questions=rd.get("sub_questions"),
                             extra={k: v for k, v in rd.items()
                                    if k not in ("round", "query_used", "chunks", "entities",
                                                 "paths", "sub_questions")})
            aliases = it.get("answer_aliases", [])
            em = exact_match_aliases(ans, it.get("answer", ""), aliases)
            f1 = f1_score_aliases(ans, it.get("answer", ""), aliases)
            tr.set_final(answer=nr["answer"], evidence=nr["evidence"],
                         evidence_locs=nr["evidence_locs"], extra={"raw_keys": nr["_raw_keys"]})
            finish_trace(tr, _trace_method, args.dataset, args.graph_version, args.retrieval,
                         sample=args.sample, em=em, f1=f1, answer=nr["answer"])
            clear_trace()
        except Exception as e:                                   # noqa: BLE001
            print(f"[trace] ⚠️ 写 {it.get('id')} 的 trace 失败（不影响实验）：{e}", flush=True)
            try:
                from common.trace import clear_trace
                clear_trace()
            except Exception:                                    # noqa: BLE001
                pass

    def report(it, ans, usage=None, res=None):
        nonlocal done
        done += 1
        gold = it["answer"]
        aliases = it.get("answer_aliases", [])
        e = exact_match_aliases(ans, gold, aliases)
        f = f1_score_aliases(ans, gold, aliases)
        # ★ 行内补上 factalign/run.py 同款字段（evidence / token / 时间分解）：原有
        #   `id/question/gold/pred/em/f1` 一字不动 → `tools/compare_runs.py` 的命令照用。
        ub = (usage or {}).get("build", {}) or {}
        uq = (usage or {}).get("query", {}) or {}
        res = res or {}
        row = {"id": it["id"], "question": it["question"], "gold": gold,
               "pred": ans, "em": e, "f1": f,
               "entities": res.get("entities", []),
               "evidence": res.get("evidence", []),
               "evidence_locs": res.get("evidence_locs", []),
               "prompt_tokens": ub.get("prompt_tokens", 0) + uq.get("prompt_tokens", 0),
               "completion_tokens": ub.get("completion_tokens", 0) + uq.get("completion_tokens", 0),
               "embed_tokens": ub.get("embed_tokens", 0) + uq.get("embed_tokens", 0),
               "chat_seconds": round(ub.get("chat_seconds", 0) + uq.get("chat_seconds", 0), 3),
               "embed_seconds": round(ub.get("embed_seconds", 0) + uq.get("embed_seconds", 0), 3),
               "llm_seconds": round(ub.get("chat_seconds", 0) + uq.get("chat_seconds", 0)
                                    + ub.get("embed_seconds", 0) + uq.get("embed_seconds", 0), 2),
               "llm_calls": ub.get("chat_calls", 0) + uq.get("chat_calls", 0),
               "embed_calls": ub.get("embed_calls", 0) + uq.get("embed_calls", 0)}
        with lock:
            stats["em"] += e
            stats["f1"] += f
            stats["n"] += 1
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
        print(f"[{done}/{total}] EM={'√' if e else '×'} | gold={gold!r} pred={ans[:50]!r}",
              flush=True)

    if args.workers <= 1:
        for idx, it in enumerate(items):
            it2, ans, usage, res = answer_one(idx, it)
            report(it2, ans, usage, res)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(answer_one, idx, it) for idx, it in enumerate(items)]
            for f in as_completed(futs):
                it2, ans, usage, res = f.result()
                report(it2, ans, usage, res)

    n = stats["n"]
    fout.close()
    print(f"\n=== test_layer ({args.rerank}) n={n} ===", flush=True)
    print(f"EM={stats['em']/n:.4f}  F1={stats['f1']/n:.4f}", flush=True)
    print(f"已存: {results_path}", flush=True)

    # ── token / 时间成本（逐题差值合计 + build/query 分桶）──
    if n:
        tot = tok["prompt"] + tok["completion"] + tok["embed"]
        print(f"\n=== token 消耗 (n={n}) ===", flush=True)
        print(f"合计: prompt={tok['prompt']:,} completion={tok['completion']:,} "
              f"embed={tok['embed']:,} total={tot:,}", flush=True)
        print(f"每题平均: {tot/n:,.0f} tokens "
              f"(prompt {tok['prompt']/n:,.0f} / completion {tok['completion']/n:,.0f} "
              f"/ embed {tok['embed']/n:,.0f}), "
              f"{tok['seconds']/n:.2f}s, {tok['calls']/n:.1f} calls", flush=True)
        try:
            s = get_stats()
            print(f"分桶: build {s['build']['total_seconds']:.1f}s / "
                  f"{s['build']['total_tokens']:,} token | "
                  f"query {s['query']['total_seconds']:.1f}s / "
                  f"{s['query']['total_tokens']:,} token", flush=True)
        except Exception:
            pass

    # ── 汇总**另起一份**：`results/ours/{exp}/ourssummary.jsonl`（不动 run.py 的
    #    `results/summary.jsonl`，免得和它的「同 v+r 已跑完」判据、以及其它方法的记录混在一起）──
    try:
        _tot = tok["prompt"] + tok["completion"] + tok["embed"]
        _row = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "method": _trace_method,
                "dataset": args.dataset, "sample": args.sample, "tag": args.tag,
                "graph_version": args.graph_version, "retrieval": args.retrieval,
                "config": {"recall": args.recall, "rerank": args.rerank,
                           "decompose": bool(args.decompose),
                           "decompose_desc": bool(args.decompose_desc),
                           "decompose_stmt": bool(args.decompose_stmt),
                           "decompose_recall": args.decompose_recall,
                           "decomp_top_k": args.decomp_top_k, "decomp_steps": args.decomp_steps,
                           "gen_top": args.gen_top, "workers": args.workers,
                           "limit": args.limit, "trace": bool(args.trace)},
                "n": n, "em": round(stats["em"] / n, 4) if n else 0.0,
                "f1": round(stats["f1"] / n, 4) if n else 0.0,
                "prompt_tokens": tok["prompt"], "completion_tokens": tok["completion"],
                "embed_tokens": tok["embed"], "total_tokens": _tot,
                "per_question": {"tokens": round(_tot / n, 1) if n else 0.0,
                                 "prompt_tokens": round(tok["prompt"] / n, 1) if n else 0.0,
                                 "completion_tokens": round(tok["completion"] / n, 1) if n else 0.0,
                                 "embed_tokens": round(tok["embed"] / n, 1) if n else 0.0,
                                 "seconds": round(tok["seconds"] / n, 2) if n else 0.0,
                                 "calls": round(tok["calls"] / n, 2) if n else 0.0},
                "chat_seconds": round(tok["seconds"], 1), "chat_calls": tok["calls"],
                "wall_seconds": round(time.time() - _wall0, 1),
                "results": str(results_path),
                "trace_dir": str(_trace_dir) if _trace_dir else None,
                "argv": sys.argv}
        _sp = _ours_dir / "ourssummary.jsonl"
        with _sp.open("a", encoding="utf-8") as _f:
            _f.write(json.dumps(_row, ensure_ascii=False) + "\n")
        print(f"[summary] 汇总已追加: {_sp}", flush=True)
    except Exception as e:                                       # noqa: BLE001
        print(f"[summary] ⚠️ 写 ourssummary 失败（不影响实验）：{e}", flush=True)


if __name__ == "__main__":
    main()
