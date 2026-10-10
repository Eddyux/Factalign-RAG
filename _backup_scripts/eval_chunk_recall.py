"""eval_chunk_recall.py — 检索层验证：chunk 筛选特征对「含答案 chunk」召回率的影响。

不跑 LLM 生成（只 embed query），几分钟出表。核心问题：
  用「chunk 内超边的语义分」等细粒度特征指导 chunk 筛选，是否比纯 chunk 特征（dense/BM25）召回更好？

配置:
  A dense            : cos(q, chunk)（基线）
  B rrf(dense,bm25)  : chunk dense + BM25 排名融合（= standardrag 口径）
  C rrf(dense,he)    : dense + 「chunk 内超边最大语义分」RRF（细粒度特征）
  D C+ent+ppr        : 再加实体命中数 + 关联超边 PPR 聚合
  E hyperedge-only   : 全量超边 dense 直接检索，再映射回 chunk（上界参照）

指标: Recall@10/@20（any=任一 gold chunk 命中；all=全部命中）、MRR
统计: 配对命中差异（C vs A：多命中/少命中题数 + 二项检验 p 值）

用法:
  python eval_chunk_recall.py --dataset hotpotqa_time --sample 100 --limit 100
  python eval_chunk_recall.py --dataset 2wiki --sample 300
"""

import argparse
import math
import re

import numpy as np

from common.data import iter_questions
from common.llm import embed_batch, embed as llm_embed, phase, reset_stats
from methods.standard_rag import build_chunks
from corpus import get_or_build, merged_contexts


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _bm25(query: str, docs: list[list[str]], k1=1.5, b=0.75) -> np.ndarray:
    N = len(docs)
    if N == 0:
        return np.zeros(0, dtype=np.float32)
    avgdl = sum(len(d) for d in docs) / N or 1.0
    df = {}
    for d in docs:
        for t in set(d):
            df[t] = df.get(t, 0) + 1
    q_terms = set(_tokenize(query))
    out = np.zeros(N, dtype=np.float32)
    for i, d in enumerate(docs):
        tf = {}
        for t in d:
            tf[t] = tf.get(t, 0) + 1
        s = 0.0
        for t in q_terms:
            if t in tf:
                idf = math.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * (tf[t] * (k1 + 1)) / (tf[t] + k1 * (1 - b + b * len(d) / avgdl))
        out[i] = s
    return out


def _rrf(*score_lists, k: int = 60) -> np.ndarray:
    """多路分数的排名倒数融合（无需权重、无需归一化）。"""
    n = len(score_lists[0])
    fused = np.zeros(n, dtype=np.float64)
    for s in score_lists:
        order = np.argsort(-s)
        ranks = np.empty(n, dtype=np.int64)
        ranks[order] = np.arange(n)
        fused += 1.0 / (k + ranks)
    return fused


def _recall_at(gold: set, order: list, k: int, require_all: bool) -> float:
    topk = set(order[:k])
    if not gold:
        return 0.0
    hit = len(gold & topk)
    return 1.0 if (hit == len(gold) if require_all else hit > 0) else 0.0


def _mrr(gold: set, order: list) -> float:
    for r, i in enumerate(order, start=1):
        if i in gold:
            return 1.0 / r
    return 0.0


def _binom_p(a: int, b: int) -> float:
    """配对符号检验的双侧精确 p 值（a=赢的题数，b=输的题数）。"""
    n = a + b
    if n == 0:
        return 1.0
    k = min(a, b)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--configs", default="A,B,C,D,E")
    args = ap.parse_args()

    items = list(iter_questions(args.dataset, args.sample))
    if args.limit > 0:
        items = items[: args.limit]
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]

    reset_stats()
    with phase("build"):
        entities, hyperedges, _hchunks, hvecs = get_or_build(args.dataset, args.sample)
        chunks = build_chunks(merged_contexts(args.dataset, args.sample))
        chunk_vecs = embed_batch([c[2] for c in chunks]) if chunks else np.zeros((0, 1024))
        # 窗口文本 -> 超边下标（超边 window_text 与 build_chunks 同切法同输入）
        wt_to_he: dict[str, list[int]] = {}
        for hi, h in enumerate(hyperedges):
            wt = (h.get("window_text") or "").strip()
            if wt:
                wt_to_he.setdefault(wt, []).append(hi)
        # chunk -> 关联超边下标
        chunk_hes: list[list[int]] = [
            wt_to_he.get((c[2] or "").strip(), []) for c in chunks
        ]
        n_with_he = sum(1 for x in chunk_hes if x)
        print(f"chunk 数: {len(chunks)} | 其中有超边的: {n_with_he} "
              f"| 超边数: {len(hyperedges)} | 测试 {len(items)} 题", flush=True)
        # 实体 -> 超边（PPR/实体特征用）
        ent_he: dict[str, list[int]] = {}
        for hi, h in enumerate(hyperedges):
            for eid in h.get("entity_ids", []):
                ent_he.setdefault(eid, []).append(hi)
        ent_name = {e["id"]: e["name"] for e in entities}

    # 每题：算各配置的 chunk 排序
    per_cfg_hits = {c: {"any10": 0.0, "any20": 0.0, "all10": 0.0, "all20": 0.0,
                        "mrr": 0.0, "n": 0} for c in configs}
    pair_flags = {}  # 配置 -> 每题的 any20 命中(1/0)，用于配对检验
    win_toks = [_tokenize(c[2]) for c in chunks]

    for qi, it in enumerate(items):
        q = it["question"]
        qv = llm_embed(q)
        sf = {tuple(x) for x in it.get("supporting_facts", [])}
        gold = {i for i, c in enumerate(chunks)
                if any((t, s) in sf for t, s in c[3])}
        if not gold:
            continue  # 该题 gold 支撑句不在任何 chunk（数据问题），跳过

        dense = (chunk_vecs @ qv).astype(np.float32)  # 归一化向量内积 = 余弦
        bm25 = _bm25(q, win_toks)
        he_max = np.zeros(len(chunks), dtype=np.float32)
        he_ppr_like = np.zeros(len(chunks), dtype=np.float32)
        ent_hits = np.zeros(len(chunks), dtype=np.float32)
        for ci, hes in enumerate(chunk_hes):
            if not hes:
                continue
            sims = hvecs[hes] @ qv
            he_max[ci] = float(sims.max())
            he_ppr_like[ci] = float(sims.mean())
            cnt = 0
            for hi in hes:
                cnt += len(hyperedges[hi].get("entity_ids", []))
            ent_hits[ci] = cnt
        # E: 全量超边 dense → 映射回 chunk（该 chunk 的最佳超边分）
        he_only = he_max.copy()

        score_map = {
            "A": dense,
            "B": _rrf(dense, bm25),
            "C": _rrf(dense, he_max),
            "D": _rrf(dense, he_max, he_ppr_like, ent_hits),
            "E": he_only,
        }
        for c in configs:
            sc = score_map[c]
            order = list(np.argsort(-sc))
            st = per_cfg_hits[c]
            st["any10"] += _recall_at(gold, order, 10, False)
            st["any20"] += _recall_at(gold, order, 20, False)
            st["all10"] += _recall_at(gold, order, 10, True)
            st["all20"] += _recall_at(gold, order, 20, True)
            st["mrr"] += _mrr(gold, order)
            st["n"] += 1
            pair_flags.setdefault(c, []).append(
                _recall_at(gold, order, 20, False))

    n = per_cfg_hits[configs[0]]["n"] if configs else 0
    print(f"\n=== chunk 筛选特征对比 (有效题数 {n}) ===")
    print(f"{'cfg':<5}{'R@10(any)':>11}{'R@20(any)':>11}{'R@10(all)':>11}{'R@20(all)':>11}{'MRR':>9}")
    for c in configs:
        st = per_cfg_hits[c]
        nn = st["n"] or 1
        print(f"{c:<5}{st['any10']/nn:>11.4f}{st['any20']/nn:>11.4f}"
              f"{st['all10']/nn:>11.4f}{st['all20']/nn:>11.4f}{st['mrr']/nn:>9.4f}")

    # 配对检验：C vs A（any@20）
    if "A" in pair_flags and "C" in pair_flags:
        a = pair_flags["A"]
        c = pair_flags["C"]
        win = sum(1 for x, y in zip(a, c) if y > x)
        lose = sum(1 for x, y in zip(a, c) if x > y)
        print(f"\n配对检验 C vs A（R@20-any）: C 赢 {win} 题 / A 赢 {lose} 题 "
              f"/ 平 {len(a)-win-lose} 题, 双侧符号检验 p={_binom_p(win, lose):.4f}")
    for other in ("B", "D", "E"):
        if other in pair_flags and "C" in pair_flags:
            a, c = pair_flags[other], pair_flags["C"]
            win = sum(1 for x, y in zip(a, c) if y > x)
            lose = sum(1 for x, y in zip(a, c) if x > y)
            print(f"配对检验 C vs {other}: C 赢 {win} / 输 {lose}, "
                  f"p={_binom_p(win, lose):.4f}")


if __name__ == "__main__":
    main()
