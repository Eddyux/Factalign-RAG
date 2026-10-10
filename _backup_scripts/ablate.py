"""消融实验：复用已有语料图（data/corpus/），跑全部消融配置，输出 EM/F1 + 成本对比。

每个配置只改「检索逻辑」、共用同一张图（不碰 extract.py / EXTRACT_VERSION，图缓存有效）。
断点续：每个配置的结果写 results/ablate/{config}_{dataset}.jsonl，重跑跳过已完成题。

用法:
  python ablate.py --dataset hotpotqa --sample 300 --workers 8          # 跑全部 8 个
  python ablate.py --dataset hotpotqa --sample 300 --configs full,no_constraint,no_expand
"""

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from common.data import iter_questions
from common.metrics import exact_match_aliases, f1_score_aliases
from common.llm import snapshot, diff_snapshot
from methods.hyperfactrag import HyperfactRAG
from corpus import get_or_build
from config import RESULTS_DIR

# 每个配置只关一个组件（单变量消融）
ABLATIONS = {
    "full":          ("完整方法", {}),
    "no_head":       ("-答案指向", {"use_answer_type": False}),
    "no_constraint": ("-时空约束", {"use_constraint": False}),
    "no_entity":     ("-实体维度", {"use_entity": False}),
    "no_hybrid":     ("-BM25", {"hybrid": False}),
    "no_expand":     ("-2hop扩展", {"use_expansion": False}),
    "single_round":  ("-多轮judge", {"multi_round": False}),
    "fixed_weights": ("固定权重", {"fixed_weights": True}),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--sample", type=int, default=300)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--configs", type=str, default="",
                    help="逗号分隔，默认跑全部 8 个")
    ap.add_argument("--limit", type=int, default=0,
                    help="只跑前 N 题（0=全部），先小规模看趋势用")
    ap.add_argument("--graph-version", type=int, default=1,
                    help="复用第几次建图的语料图（v 号，默认 1）")
    args = ap.parse_args()

    configs = ([c.strip() for c in args.configs.split(",") if c.strip()]
               if args.configs else list(ABLATIONS))
    for c in configs:
        if c not in ABLATIONS:
            print(f"未知配置 {c!r}，可选: {list(ABLATIONS)}")
            return

    items = list(iter_questions(args.dataset, args.sample))
    if not items:
        print(f"没找到数据 data/samples/{args.dataset}_*.json，先跑 sample_datasets.py")
        return
    if args.limit > 0:
        items = items[:args.limit]

    # 复用已有图（不重建；没有则现场建一次）
    graph = get_or_build(args.dataset, args.sample, args.graph_version)
    print(f"复用语料图: {len(graph[0])} 实体 / {len(graph[1])} 超边\n", flush=True)

    outdir = RESULTS_DIR / "ablate"
    outdir.mkdir(parents=True, exist_ok=True)

    summary = []
    for key in configs:
        desc, kwargs = ABLATIONS[key]
        method = HyperfactRAG(**kwargs)
        method.prepare(*graph)  # faiss + 索引一次建好，各题复用

        path = outdir / f"{key}_{args.dataset}.jsonl"
        # 一次性读入已有行（断点续），之后在内存里累加，不再重读文件
        all_rows = []
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        all_rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        done_ids = {r["id"] for r in all_rows}
        pending = [it for it in items if it["id"] not in done_ids]
        print(f"[{key}] {desc}: 已完成 {len(done_ids)}，待跑 {len(pending)}", flush=True)

        lock = threading.Lock()
        fout = open(path, "a", encoding="utf-8")

        def answer_one(item):
            for attempt in range(3):
                before = snapshot()
                try:
                    r = method.query(item["question"], *graph)
                except Exception as e:
                    if attempt == 2:
                        raise
                    continue
                usage = diff_snapshot(before, snapshot())
                return item, r, usage

        if pending:
            rows = []
            total = len(pending)
            done = 0
            if args.workers <= 1:
                for item in pending:
                    rows.append(answer_one(item))
                    done += 1
                    if done % 5 == 0 or done == total:
                        print(f"  [{key}] {done}/{total}", flush=True)
            else:
                with ThreadPoolExecutor(max_workers=args.workers) as ex:
                    futs = [ex.submit(answer_one, it) for it in pending]
                    for f in as_completed(futs):
                        rows.append(f.result())
                        done += 1
                        if done % 5 == 0 or done == total:
                            print(f"  [{key}] {done}/{total}", flush=True)

            for it, r, u in rows:
                row = {"id": it["id"], "question": it["question"],
                       "gold": it["answer"], "pred": r["answer"],
                       "em": exact_match_aliases(r["answer"], it["answer"], it.get("answer_aliases", [])),
                       "f1": f1_score_aliases(r["answer"], it["answer"], it.get("answer_aliases", [])),
                       "prompt_tokens": u["query"]["prompt_tokens"] + u["build"]["prompt_tokens"],
                       "completion_tokens": u["query"]["completion_tokens"] + u["build"]["completion_tokens"],
                       "llm_seconds": round(u["query"]["chat_seconds"] + u["build"]["chat_seconds"], 2),
                       "llm_calls": u["query"]["chat_calls"] + u["build"]["chat_calls"]}
                with lock:
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fout.flush()
                all_rows.append(row)
        fout.close()

        # 汇总（直接算内存里的 all_rows，不再重读文件）
        n = len(all_rows)
        em = sum(r["em"] for r in all_rows) / n if n else 0.0
        f1 = sum(r["f1"] for r in all_rows) / n if n else 0.0
        pt = sum(r.get("prompt_tokens", 0) for r in all_rows)
        ct = sum(r.get("completion_tokens", 0) for r in all_rows)
        sec = sum(r.get("llm_seconds", 0) for r in all_rows)
        calls = sum(r.get("llm_calls", 0) for r in all_rows)
        summary.append({"config": key, "desc": desc, "n": n, "em": round(em, 4),
                        "f1": round(f1, 4), "prompt_tokens": pt, "completion_tokens": ct,
                        "llm_seconds": round(sec, 1), "llm_calls": calls})

    # 打印对比表
    print(f"\n=== {args.dataset} 消融对比 (n={args.sample}) ===", flush=True)
    print(f"{'config':<15}{'说明':<12}{'EM':>8}{'F1':>8}{'ΔEM(vs full)':>14}{'calls/题':>9}{'秒/题':>8}", flush=True)
    base_em = next((s["em"] for s in summary if s["config"] == "full"), None)
    for s in summary:
        d = "" if (base_em is None or s["config"] == "full") else f"{s['em'] - base_em:+.4f}"
        per_call = s["llm_calls"] / s["n"] if s["n"] else 0.0
        per_sec = s["llm_seconds"] / s["n"] if s["n"] else 0.0
        print(f"{s['config']:<15}{s['desc']:<12}{s['em']:>8.4f}{s['f1']:>8.4f}{d:>14}"
              f"{per_call:>9.1f}{per_sec:>8.1f}", flush=True)

    # 落盘汇总
    with open(outdir / f"summary_{args.dataset}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    print(f"\n汇总已写: {outdir}/summary_{args.dataset}.jsonl", flush=True)


if __name__ == "__main__":
    main()
