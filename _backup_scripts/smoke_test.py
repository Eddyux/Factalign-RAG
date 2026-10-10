"""smoke_test.py — 逐方法冒烟测试（用最小规模把每个方法跑通一遍）。

流程:
  1. 环境: vLLM chat + embedding 端点、样本数据
  2. 自带: standardrag / no_head / full（K 题上下文建小图 + 查 1 题）
  3. 外部: lightrag / graphrag / hypergraphrag / hipporag（K 题上下文建小索引 + 查 1 题；未安装记 SKIP）

图/索引统一放 data/smoke/{method}/（每次冒烟重建，不污染正式实验）；
逐题结果存 results/smoke/{method}.jsonl（与正式结果同字段）。

用法:
  python smoke_test.py                       # 全部方法，默认前 5 题
  python smoke_test.py --methods full,lightrag
  python smoke_test.py --dataset hotpotqa --n 5

退出码: 0 = 无 FAIL；1 = 有 FAIL（SKIP 不算失败）。
"""

import argparse
import json
import shutil
import sys
import time

from common.data import iter_questions
from common.llm import chat, embed
from common.metrics import exact_match_aliases, f1_score_aliases
from config import (DATA_DIR, RESULTS_DIR, SAMPLES_DIR, SEARCH_TOP_K,
                    CHAT_MODEL, EMBED_MODEL_SERVED)

OK, FAIL, SKIP = "OK", "FAIL", "SKIP"

ALL_METHODS = ["standardrag", "no_head", "full", "lightrag", "graphrag",
               "hypergraphrag", "hipporag"]
STUB_METHODS = set()  # 所有方法都有适配器了；保留机制以备后续

_results = []  # (step, status, seconds, note)


def report(step, status, seconds, note=""):
    _results.append((step, status, seconds, note))
    print(f"[{status}] {step} ({seconds:.0f}s)" + (f" — {note}" if note else ""), flush=True)


def check_env():
    # chat
    out = chat("Reply with exactly: OK", "reply", json_mode=False)
    report("vLLM chat", OK if out and out.strip() else FAIL, 0,
           CHAT_MODEL if out else "无响应，检查 8000 端口")
    # embedding
    try:
        v = embed("hello world")
        ok = v.shape == (1024,) and abs(float(v @ v) - 1.0) < 1e-3
        report("vLLM embedding", OK if ok else FAIL, 0,
               EMBED_MODEL_SERVED if ok else f"向量异常 shape={v.shape}")
    except Exception as e:
        report("vLLM embedding", FAIL, 0, str(e)[:100])
    # 数据
    files = list(SAMPLES_DIR.glob(f"{args.dataset}_*.json"))
    report("样本数据", OK if files else FAIL, 0,
           f"{len(files)} 个文件" if files else f"{SAMPLES_DIR} 里没有 {args.dataset}_*.json，先跑 sample_datasets.py")


def load_items():
    items = list(iter_questions(args.dataset, 0))[:args.n]  # n=0 时取最大样本文件
    if not items:
        raise RuntimeError("没有可用题目")
    merged = []
    for it in items:
        for c in it["contexts"]:
            c2 = dict(c)
            c2["qid"] = it["id"]
            merged.append(c2)
    return items, merged


def save_result(name, item, r):
    """逐题结果存 results/smoke/{method}.jsonl（与正式结果同字段）。返回 (em, f1)。"""
    outdir = RESULTS_DIR / "smoke"
    outdir.mkdir(parents=True, exist_ok=True)
    aliases = item.get("answer_aliases", [])
    em = exact_match_aliases(r["answer"], item["answer"], aliases)
    f1 = f1_score_aliases(r["answer"], item["answer"], aliases)
    row = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"),
           "method": name, "dataset": args.dataset,
           "id": item["id"], "question": item["question"], "gold": item["answer"],
           "pred": r["answer"], "em": em, "f1": f1}
    with open(outdir / f"{name}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return em, f1


def smoke_standardrag(items, merged):
    from methods.standard_rag import build_chunks, query_chunks
    from common.llm import embed_batch
    from search import build_faiss_index
    chunks = build_chunks(merged)
    vecs = embed_batch([c[2] for c in chunks])
    faiss = build_faiss_index(vecs) if chunks else None
    r = query_chunks(items[0]["question"], chunks, SEARCH_TOP_K,
                     vecs=vecs, faiss_index=faiss)
    assert isinstance(r.get("answer"), str) and r["answer"].strip(), "answer 为空"
    em, f1 = save_result("standardrag", items[0], r)
    return f"{len(chunks)} 块, pred={r['answer'][:40]!r} EM={em} F1={f1:.3f}"


def smoke_hyperfact(items, merged, use_answer_type):
    from extract import build_graph_for_question
    from methods.hyperfactrag import HyperfactRAG
    graph = build_graph_for_question(f"smoke_{use_answer_type}", merged)  # 按题缓存，重跑快
    m = HyperfactRAG(use_answer_type=use_answer_type)
    m.prepare(*graph)
    r = m.query(items[0]["question"], *graph)
    assert isinstance(r.get("answer"), str) and r["answer"].strip(), "answer 为空"
    em, f1 = save_result("full" if use_answer_type else "no_head", items[0], r)
    return f"{len(graph[1])} 超边, pred={r['answer'][:40]!r} EM={em} F1={f1:.3f}"


def smoke_external(name, merged, question, item):
    from methods import get_method
    wdir = DATA_DIR / "smoke" / name
    shutil.rmtree(wdir, ignore_errors=True)  # 冒烟每次重建，不污染真实索引
    m = get_method(name)
    m.build(merged, working_dir=str(wdir))
    r = m.query(question)
    assert isinstance(r.get("answer"), str) and r["answer"].strip(), "answer 为空"
    em, f1 = save_result(name, item, r)
    return f"pred={r['answer'][:40]!r} EM={em} F1={f1:.3f}"


def smoke_stub(name, item):
    from methods import get_method
    m = get_method(name)
    try:
        m.run(item["question"], item["contexts"])
        return "run() 正常返回（适配器已实现？）"
    except NotImplementedError as e:
        raise RuntimeError(f"适配器未实现: {str(e)[:80]}")  # 当前预期状态 → SKIP
    except RuntimeError as e:
        raise RuntimeError(f"未就绪: {str(e)[:80]}")  # 未 clone 等 → SKIP


def main():
    check_env()
    items, merged = load_items()
    q = items[0]["question"]
    print(f"\n冒烟: {args.dataset} 前 {args.n} 题, {len(merged)} 条上下文, 查询 '{q[:50]}...'", flush=True)

    for name in args.methods:
        t0 = time.time()
        try:
            if name == "standardrag":
                note = smoke_standardrag(items, merged)
            elif name in ("no_head", "full"):
                note = smoke_hyperfact(items, merged, use_answer_type=(name == "full"))
            elif name in ("lightrag", "graphrag", "hypergraphrag", "hipporag"):
                note = smoke_external(name, merged, q, items[0])
            elif name in STUB_METHODS:
                try:
                    note = smoke_stub(name, items[0])
                except RuntimeError as e:
                    report(name, SKIP, time.time() - t0, str(e)[:100])
                    continue
            else:
                report(name, FAIL, 0, "未知方法")
                continue
            report(name, OK, time.time() - t0, note)
        except RuntimeError as e:
            # 未安装（pip install ...）/ 未 clone（未找到 ...）都是预期内 SKIP
            if "pip install" in str(e) or "未找到" in str(e):
                report(name, SKIP, time.time() - t0, str(e)[:100])
            else:
                report(name, FAIL, time.time() - t0, str(e)[:150])
        except Exception as e:
            report(name, FAIL, time.time() - t0, f"{type(e).__name__}: {str(e)[:150]}")

    # 汇总
    print("\n=== 汇总 ===")
    for step, status, sec, note in _results:
        print(f"  {status:4s} {step:14s} {sec:6.0f}s  {note}")
    fails = [r for r in _results if r[1] == FAIL]
    oks = [r for r in _results if r[1] == OK]
    print(f"\nOK {len(oks)} / FAIL {len(fails)} / SKIP {len(_results) - len(oks) - len(fails)}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", type=str, default="",
                    help="逗号分隔，默认全部")
    ap.add_argument("--dataset", type=str, default="hotpotqa")
    ap.add_argument("--n", type=int, default=20, help="用前 N 题上下文（默认 20，试试水）")
    ap.add_argument("--workers", type=int, default=32,
                    help="查询并发（与 run.py 同口径；冒烟只查 1 题暂未用到，"
                         "建图并发由各 adapter 内部参数控制）")
    args = ap.parse_args()
    if args.methods:
        args.methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    else:
        args.methods = ALL_METHODS
    main()
