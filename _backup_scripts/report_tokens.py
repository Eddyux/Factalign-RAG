"""从 results/summary.jsonl 生成 token 消耗对比表（markdown / latex）。

数据来源：每次正常跑完 run.py 都会追加一行 summary（含 build/query 分桶 token、秒、EM/F1）。
本脚本按「方法 × 数据集」取最新的 complete 行，聚合出两张表：
  表1 总成本；表2 每题平均推理成本。

用法:
  python report_tokens.py                     # 全部数据集取平均，markdown
  python report_tokens.py --dataset hotpotqa_gr1
  python report_tokens.py --format latex
"""

import argparse
import json
from collections import defaultdict

from config import RESULTS_DIR

METHOD_ORDER = ["standardrag", "no_head", "full", "lightrag", "graphrag",
                "hypergraphrag", "hipporag"]


def load_rows():
    rows = []
    path = RESULTS_DIR / "summary.jsonl"
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def latest_complete(rows, dataset=None, graph_version=None, retrieval=None):
    """(method, dataset) -> 最新一条 complete=True 的行。

    可按图版本/检索次数过滤：--graph-version 2 --retrieval 3 只看 v2r3；
    不传则取每个 (method, dataset) 的最新行。
    """
    best = {}
    for r in rows:
        if dataset and r.get("dataset") != dataset:
            continue
        if graph_version is not None and r.get("graph_version") != graph_version:
            continue
        if retrieval is not None and r.get("retrieval") != retrieval:
            continue
        if not r.get("complete", False):
            continue
        key = (r["method"], r["dataset"])
        old = best.get(key)
        if old is None or r["ts"] > old["ts"]:
            best[key] = r
    return best


def _k(x):
    return round(x / 1000, 1)


def aggregate(best):
    """按方法聚合（多数据集求和 token/时间，平均 EM/F1）。"""
    out = defaultdict(lambda: {"em": [], "f1": [], "build_p": 0, "build_c": 0,
                               "build_e": 0, "q_p": 0, "q_c": 0, "q_e": 0,
                               "build_s": 0.0, "query_s": 0.0, "n": 0, "calls": 0})
    for (method, _ds), r in best.items():
        a = out[method]
        a["em"].append(r["em"]); a["f1"].append(r["f1"])
        a["build_p"] += r.get("build_prompt_tokens", 0)
        a["build_c"] += r.get("build_completion_tokens", 0)
        a["build_e"] += r.get("build_embed_tokens", 0)
        a["q_p"] += r["prompt_tokens"]; a["q_c"] += r["completion_tokens"]
        a["q_e"] += r.get("query_embed_tokens", 0)
        a["build_s"] += r.get("build_seconds", 0)
        a["query_s"] += r.get("query_seconds", 0)
        a["n"] += r["n"]; a["calls"] += r.get("chat_calls", 0)
    return out


def render_markdown(out, shared_build):
    print("| Method | EM | F1 | Build LLM (k) | Build Emb (k) | Query LLM (k) | "
          "Query Emb (k) | Total (k) | Build (h) | Query (h) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for m in METHOD_ORDER:
        a = out.get(m)
        if not a:
            continue
        em = sum(a["em"]) / len(a["em"])
        f1 = sum(a["f1"]) / len(a["f1"])
        total = a["build_p"] + a["build_c"] + a["build_e"] + a["q_p"] + a["q_c"] + a["q_e"]
        build_str = f"{_k(a['build_p']+a['build_c'])}" if m not in shared_build else "共享"
        print(f"| {m} | {em:.4f} | {f1:.4f} | {build_str} | {_k(a['build_e'])} | "
              f"{_k(a['q_p'])} | {_k(a['q_c'])} | {_k(a['q_e'])} | {_k(total)} | "
              f"{a['build_s']/3600:.1f} | {a['query_s']/3600:.1f} |")
    print()
    print("| Method | Prompt/题 | Completion/题 | 秒/题 | LLM调用/题 |")
    print("|---|---|---|---|---|")
    for m in METHOD_ORDER:
        a = out.get(m)
        if not a:
            continue
        n = a["n"] or 1
        print(f"| {m} | {a['q_p']/n:,.0f} | {a['q_c']/n:,.0f} | "
              f"{a['query_s']/n:.1f} | {a['calls']/n:.1f} |")


def render_latex(out, shared_build):
    print("\\begin{table}[t]")
    print("\\centering\\small")
    print("\\begin{tabular}{lccccccccc}")
    print("\\toprule")
    print("Method & EM & F1 & Build LLM (k) & Build Emb (k) & Query P (k) & "
          "Query C (k) & Query Emb (k) & Build (h) & Query (h) \\\\")
    print("\\midrule")
    for m in METHOD_ORDER:
        a = out.get(m)
        if not a:
            continue
        em = sum(a["em"]) / len(a["em"])
        f1 = sum(a["f1"]) / len(a["f1"])
        build_str = f"{_k(a['build_p']+a['build_c'])}" if m not in shared_build else "shared"
        print(f"{m} & {em:.3f} & {f1:.3f} & {build_str} & {_k(a['build_e'])} & "
              f"{_k(a['q_p'])} & {_k(a['q_c'])} & {_k(a['q_e'])} & "
              f"{a['build_s']/3600:.1f} & {a['query_s']/3600:.1f} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{Token and time costs per method.}")
    print("\\end{table}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="",
                    help="只看某个数据集；默认全部聚合")
    ap.add_argument("--graph-version", type=int, default=None,
                    help="只看某个图版本（v 号）；默认每个方法取最新 complete 行")
    ap.add_argument("--retrieval", type=int, default=None,
                    help="只看某次检索（r 号）；与 --graph-version 配合精确取一行")
    ap.add_argument("--format", choices=["markdown", "latex"], default="markdown")
    args = ap.parse_args()

    rows = load_rows()
    if not rows:
        print("没有 summary.jsonl 数据（先正常跑完至少一次 run.py）")
        return
    best = latest_complete(rows, args.dataset or None,
                           args.graph_version, args.retrieval)
    if not best:
        print("没有 complete 的行（全部被中断？）")
        return
    out = aggregate(best)
    # no_head 与 full 共享语料图，建图成本只记在 full
    shared_build = {"no_head"}
    if args.format == "latex":
        render_latex(out, shared_build)
    else:
        render_markdown(out, shared_build)


if __name__ == "__main__":
    main()
