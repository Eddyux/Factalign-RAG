"""filter_time.py — 用 vLLM 判 hotpotqa 验证集全量（7405 题）是否「时间相关」。

喂「题目 + 答案」给 LLM，判「回答此题是否需要时序推理」（年份/时间/年龄/先后/继任等），
即使题目没显式提时间也能判。断点续 + 并发。输出：
  results/time_filter_hotpotqa.jsonl    每题一行：{id, question, answer, is_time, raw}
  data/samples/hotpotqa_time_{N}.json   筛出的时间题（统一格式，可直接跑 run.py）

用法:
  python filter_time.py --workers 32
  python filter_time.py --limit 100 --workers 8
"""

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from common.llm import chat as llm_chat
from config import SAMPLES_DIR, RESULTS_DIR

JUDGE_SYSTEM = "You are a question classifier."
JUDGE_USER = """Is this question TIME-RELATED?

A question is time-related if answering it requires TEMPORAL reasoning: dates, years,
time periods, centuries, decades, ages, durations, or the ordering/succession of events
(e.g. "when", "what year", "how old", "who was born earlier", "who succeeded whom",
"which happened first", "how long"). Even if the question does NOT explicitly mention
time words, judge whether the underlying reasoning is about time.

Question: {question}
Answer: {answer}

Answer ONLY "yes" or "no"."""


def load_full_hotpotqa() -> list[dict]:
    """下载 hotpotqa 验证集全量（distractor，7405 题），转统一格式。"""
    from datasets import load_dataset
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    items = []
    for item in ds:
        ctx = item["context"]
        contexts = [{"title": t.strip(), "text": " ".join(s).strip(),
                     "sentences": [x.strip() for x in s]}
                    for t, s in zip(ctx["title"], ctx["sentences"])]
        sf = item.get("supporting_facts", {})
        facts = [[t.strip(), int(s)] for t, s in
                 zip(sf.get("title", []), sf.get("sent_id", []))]
        items.append({"id": str(item["id"]), "question": item["question"].strip(),
                      "answer": item["answer"].strip(), "answer_aliases": [],
                      "contexts": contexts, "supporting_facts": facts})
    return items


def judge_time(question: str, answer: str):
    """喂题目+答案给 LLM，判是否时间相关。返回 (is_time, raw)。"""
    out = llm_chat(JUDGE_SYSTEM, JUDGE_USER.format(question=question, answer=answer),
                   json_mode=False, timeout=60, thinking=False)
    if not out:
        return False, ""
    raw = out.strip() if isinstance(out, str) else str(out).strip()
    low = raw.lower()
    is_time = low.startswith("yes")
    return is_time, raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="只判前 N 题（0=全部）")
    args = ap.parse_args()

    items = load_full_hotpotqa()
    if args.limit > 0:
        items = items[: args.limit]
    by_id = {it["id"]: it for it in items}
    print(f"hotpotqa 验证集全量: {len(items)} 题", flush=True)

    # 断点续
    out_path = RESULTS_DIR / "time_filter_hotpotqa.jsonl"
    done_ids = set()
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    done_ids.add(json.loads(line)["id"])
                except Exception:
                    pass
    pending = [it for it in items if it["id"] not in done_ids]
    print(f"已判 {len(done_ids)}，待判 {len(pending)}，{args.workers} 并发", flush=True)

    lock = threading.Lock()
    fout = open(out_path, "a", encoding="utf-8")

    def run_one(item):
        for attempt in range(3):
            try:
                is_time, raw = judge_time(item["question"], item["answer"])
                break
            except Exception:
                is_time, raw = False, ""
        row = {"id": item["id"], "question": item["question"],
               "answer": item["answer"], "is_time": is_time, "raw": raw}
        with lock:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
        return is_time

    if pending:
        if args.workers <= 1:
            for it in pending:
                run_one(it)
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futs = [ex.submit(run_one, it) for it in pending]
                for f in as_completed(futs):
                    f.result()
    fout.close()

    # 汇总 + 存时间题样本
    time_rows = []
    for line in out_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("is_time"):
            time_rows.append(d)

    total = len(items)
    print(f"\n时间题: {len(time_rows)}/{total}", flush=True)

    if time_rows:
        out_items = []
        for r in time_rows:
            it = by_id[r["id"]]
            out_items.append({"id": r["id"], "question": r["question"],
                              "answer": r["answer"],
                              "answer_aliases": it.get("answer_aliases", []),
                              "contexts": it["contexts"],
                              "supporting_facts": it["supporting_facts"]})
        out_sample = SAMPLES_DIR / f"hotpotqa_time_{len(out_items)}.json"
        out_sample.write_text(json.dumps(
            {"dataset": "hotpotqa_time", "sample_size": len(out_items),
             "random_seed": 42, "items": out_items},
            ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"时间题样本已存: {out_sample}", flush=True)


if __name__ == "__main__":
    main()
