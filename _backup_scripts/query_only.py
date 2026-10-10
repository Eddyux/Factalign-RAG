"""只查不建：直接加载 data/ 里已建好的索引，逐题检索 + 生成答案。

与 run.py 的区别：完全没有 build / 断点 / 重建 / marker 判断——
索引不存在就报错退出，绝不触发任何抽取。适合「索引已建好」后的重复跑题
（换生成 prompt、调参数、补跑）。

结果文件与 run.py 完全同格式（results/{method}_{dataset}.jsonl，断点续 +
summary.jsonl），report_tokens.py 可直接用。

用法:
  python query_only.py --method lightrag --dataset musique --sample 300 --workers 16
  python query_only.py --method graphrag  --dataset hotpotqa --sample 300 --workers 8
"""

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from common.data import iter_questions
from common.metrics import exact_match_aliases, f1_score_aliases
from common.llm import reset_stats, snapshot, diff_snapshot
from common.result import graph_dir, result_path
from methods import get_method
from config import (RESULTS_DIR, SAMPLES_DIR)

# 语料级 baseline（索引建一次、逐题 query）
QUERY_METHODS = ["lightrag", "graphrag", "hipporag", "hypergraphrag"]

_EMPTY_USAGE = {"prompt_tokens": 0, "completion_tokens": 0,
                "chat_seconds": 0, "chat_calls": 0}


def _result_complete(args, results_path) -> bool:
    """该 (method, dataset, v, r) 的结果是否已完整跑过（summary 有 complete=True 行）。

    完整 → 提示已有并跳过（不覆盖不重跑）；不完整（中断）→ 断点续。
    """
    if not results_path.exists():
        return False
    sp = RESULTS_DIR / "summary.jsonl"
    if not sp.exists():
        return False
    for line in sp.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if (r.get("method") == args.method and r.get("dataset") == args.dataset
                and r.get("graph_version", 1) == args.graph_version
                and r.get("retrieval", 1) == args.retrieval
                and r.get("complete")):
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=QUERY_METHODS)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--sample", type=int, default=300)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--native-generator", action="store_true")
    ap.add_argument("--graph-version", type=int, default=1,
                    help="第几次建图（v 号），与 run.py 的目录一致")
    ap.add_argument("--retrieval", type=int, default=1,
                    help="同一张图上的第几次验证检索（r 号）：换查询参数开新 r，互不覆盖")
    args = ap.parse_args()

    if not list(SAMPLES_DIR.glob(f"{args.dataset}_*.json")):
        print(f"没找到数据 {SAMPLES_DIR}/{args.dataset}_*.json，先跑: "
              f"python sample_datasets.py --sample {args.sample}")
        return

    items = list(iter_questions(args.dataset, args.sample))
    results_path = result_path(args.method, args.dataset, args.graph_version, args.retrieval)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # 结果已存在且完整（summary 里有该 v+r 的 complete=True 行）→ 提示已有，不覆盖不重跑
    if _result_complete(args, results_path):
        print(f"[{args.method}] v{args.graph_version}r{args.retrieval} 结果已存在且完整："
              f"{results_path}\n（不覆盖；要重跑请换 --retrieval 开新 r 或删掉该文件）")
        return

    # 索引目录与 run.py 完全一致：data/{method}/{dataset}/v{n}/
    gr_dir = str(graph_dir(args.method, args.dataset, args.graph_version))

    # 断点续：读已完成题 id
    done_ids = set()
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    done_ids.add(json.loads(line)["id"])
                except Exception:
                    pass
    pending = [it for it in items if it["id"] not in done_ids]
    print(f"{args.method} @ {args.dataset}（只查不建）: "
          f"已完成 {len(done_ids)}，待跑 {len(pending)}，{args.workers} 并发")

    # ── 加载索引（存在即用，不存在报错，绝不重建）──
    reset_stats()
    method_obj = get_method(args.method)
    if args.native_generator:
        method_obj.native_generator = True
    method_obj.load(gr_dir)

    def answer_one(item):
        """每题最多 3 次重试；线程内统计快照差值算单题 token/LLM 时间。"""
        for attempt in range(3):
            before = snapshot()
            try:
                r = method_obj.query(item["question"])
            except Exception as e:
                if attempt == 2:
                    raise
                print(f"  重试 {attempt + 1}/3: {e}", flush=True)
                time.sleep(1)
                continue
            usage = diff_snapshot(before, snapshot())
            return item, r, usage

    lock = threading.Lock()
    fout = open(results_path, "a", encoding="utf-8")

    def write_row(item, r, usage):
        aliases = item.get("answer_aliases", [])
        row = {"id": item["id"], "question": item["question"], "gold": item["answer"],
               "pred": r["answer"],
               "em": exact_match_aliases(r["answer"], item["answer"], aliases),
               "f1": f1_score_aliases(r["answer"], item["answer"], aliases),
               "entities": r.get("entities", []),
               "evidence": r.get("evidence", []),
               "evidence_locs": r.get("evidence_locs", []),
               "prompt_tokens": usage["build"]["prompt_tokens"] + usage["query"]["prompt_tokens"],
               "completion_tokens": usage["build"]["completion_tokens"] + usage["query"]["completion_tokens"],
               "llm_seconds": round(usage["build"]["chat_seconds"] + usage["query"]["chat_seconds"], 2),
               "llm_calls": usage["build"]["chat_calls"] + usage["query"]["chat_calls"]}
        with lock:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()

    done = len(done_ids)
    total = len(items)

    def run_pending():
        nonlocal done
        ex = ThreadPoolExecutor(max_workers=args.workers)
        futures = {ex.submit(answer_one, it): it for it in pending}
        try:
            for f in as_completed(futures):
                _item = futures[f]
                try:
                    _item, r, u = f.result()
                except Exception as e:
                    # 3 次重试仍失败：记 0 分行，别让一道题杀死整批
                    r = {"answer": f"<ERROR: {e}>", "entities": [],
                         "evidence": [], "evidence_locs": []}
                    u = {"build": dict(_EMPTY_USAGE), "query": dict(_EMPTY_USAGE)}
                    print(f"[{args.method}] 题目 {_item['id']} 3 次重试失败：{e}",
                          flush=True)
                write_row(_item, r, u)
                done += 1
                em = exact_match_aliases(
                    r["answer"], _item["answer"], _item.get("answer_aliases", []))
                print(f"[{args.method}][{done}/{total}] EM={em} | "
                      f"gold={_item['answer']!r} pred={r['answer'][:40]!r}", flush=True)
        except KeyboardInterrupt:
            # 中断：取消未开始的题，只等在途的（≤workers 道），不再等全部
            ex.shutdown(wait=False, cancel_futures=True)
            print("\n已中断，重跑同命令断点续")
            raise
        ex.shutdown(wait=True)

    try:
        run_pending()
    except KeyboardInterrupt:
        pass
    finally:
        fout.close()

    # 汇总（与 run.py 同格式，容忍中断残留的半行）
    rows = []
    for l in results_path.read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        try:
            rows.append(json.loads(l))
        except json.JSONDecodeError:
            pass
    n = len(rows)
    if n == 0:
        print("没有结果")
        return
    em = sum(r["em"] for r in rows) / n
    f1 = sum(r["f1"] for r in rows) / n
    pt = sum(r.get("prompt_tokens", 0) for r in rows)
    ct = sum(r.get("completion_tokens", 0) for r in rows)
    llm_s = sum(r.get("llm_seconds", 0) for r in rows)
    calls = sum(r.get("llm_calls", 0) for r in rows)
    print(f"\n{args.method} @ {args.dataset}: n={n} EM={em:.4f} F1={f1:.4f}")
    print(f"LLM(逐题合计): prompt={pt:,} completion={ct:,} tokens, "
          f"{llm_s:.1f}s, calls={calls}")
    with open(RESULTS_DIR / "summary.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "method": args.method, "dataset": args.dataset, "sample": args.sample,
            "graph_version": args.graph_version, "retrieval": args.retrieval,
            "n": n, "complete": n >= len(items), "em": round(em, 4), "f1": round(f1, 4),
            "prompt_tokens": pt, "completion_tokens": ct, "embed_tokens": 0,
            "llm_seconds": round(llm_s, 1),
            "build_seconds": 0, "build_prompt_tokens": 0, "build_completion_tokens": 0,
            "build_embed_tokens": 0,
            "query_seconds": 0, "query_embed_tokens": 0,
            "chat_calls": calls,
        }, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
