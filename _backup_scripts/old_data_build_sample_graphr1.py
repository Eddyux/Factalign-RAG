"""把 Graph-R1 官方数据（TeraBox 下载的 raw/qa_*.json）转成我们的统一样本格式。

Graph-R1 六个数据集: 2WikiMultiHopQA / HotpotQA / Musique / NQ / PopQA / TriviaQA。
qa json 每项必有 question + golden_answers；上下文字段各数据集叫法不同
（context / contexts / paragraphs / docs），转换时做兼容探测。

输出 data/samples/{name}_gr1_{N}.json，用 _gr1 后缀区分 HF 版样本，
避免和已有样本/结果冲突。对照论文用 qa_test.json（没有则用 qa_dev.json）。

用法:
  python sample_graphr1.py --data_dir baselines/Graph-R1-main/datasets --sample 300
  python sample_graphr1.py --data_dir ... --datasets HotpotQA,Musique --sample 100
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

from config import SAMPLES_DIR, RANDOM_SEED

# Graph-R1 数据源名 -> 我们的样本名（_gr1 后缀防与 HF 版样本冲突）
NAME_MAP = {
    "2WikiMultiHopQA": "2wiki_gr1",
    "HotpotQA": "hotpotqa_gr1",
    "Musique": "musique_gr1",
    "NQ": "nq_gr1",
    "PopQA": "popqa_gr1",
    "TriviaQA": "triviaqa_gr1",
}

CONTEXT_KEYS = ("context", "contexts", "paragraphs", "docs", "passages")
TITLE_KEYS = ("title", "id", "doc_id", "article_title")
TEXT_KEYS = ("text", "content", "contents", "paragraph_text", "passage_text")
SENT_KEYS = ("sentences", "sents")


def _split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _norm_context(ctx) -> dict | None:
    """各种形态的上下文条目 -> {"title", "text", "sentences"}。"""
    if isinstance(ctx, str):
        text = ctx.strip()
        return {"title": "", "text": text, "sentences": _split_sentences(text)} if text else None
    if not isinstance(ctx, dict):
        return None
    if "title" in ctx and "sentences" in ctx:  # hotpotqa 风格
        sents = [str(s).strip() for s in ctx["sentences"]]
        return {"title": str(ctx.get("title", "")).strip(),
                "text": " ".join(sents), "sentences": sents}
    text = ""
    for k in TEXT_KEYS:
        if ctx.get(k):
            text = str(ctx[k]).strip()
            break
    if not text and ctx.get("sentences"):
        text = " ".join(str(s).strip() for s in ctx["sentences"])
    if not text:
        return None
    title = ""
    for k in TITLE_KEYS:
        if ctx.get(k):
            title = str(ctx[k]).strip()
            break
    sents = ctx.get("sentences") or _split_sentences(text)
    return {"title": title, "text": text,
            "sentences": [str(s).strip() for s in sents if str(s).strip()]}


def extract_contexts(item: dict) -> list[dict]:
    """探测 item 里的上下文字段，返回统一格式列表。"""
    for key in CONTEXT_KEYS:
        v = item.get(key)
        if v is None:
            continue
        if isinstance(v, dict) and "title" in v and "sentences" in v:
            c = _norm_context(v)
            return [c] if c else []
        if isinstance(v, (list, tuple)):
            out = []
            for c in v:
                nc = _norm_context(c)
                if nc:
                    out.append(nc)
            if out:
                return out
    return []


def load_dataset(data_source: str, data_dir: Path, sample: int, seed: int):
    """读 raw/qa_test.json（缺则 qa_dev.json）→ 统一格式 items。"""
    raw_dir = data_dir / data_source / "raw"
    src_file = None
    for fname in ("qa_test.json", "qa_dev.json"):
        p = raw_dir / fname
        if p.exists():
            src_file = p
            break
    if src_file is None:
        print(f"  [FAIL] {data_source}: 没找到 {raw_dir}/qa_test.json 或 qa_dev.json", file=sys.stderr)
        return []
    raw = json.loads(src_file.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("data", raw.get("items", []))
    if not isinstance(raw, list) or not raw:
        print(f"  [FAIL] {data_source}: {src_file} 不是数组", file=sys.stderr)
        return []

    rng = random.Random(seed)
    idxs = sorted(rng.sample(range(len(raw)), min(sample, len(raw))))

    items, no_ctx = [], 0
    for i in idxs:
        item = raw[i]
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        if not question:
            continue
        gold = item.get("golden_answers") or item.get("answer") or []
        if isinstance(gold, str):
            gold = [gold]
        answers = [str(a).strip() for a in gold if str(a).strip()]
        contexts = extract_contexts(item)
        if not contexts:
            no_ctx += 1
        items.append({
            "id": f"{data_source}_{i}",
            "question": question,
            "answer": answers[0] if answers else "",
            "answer_aliases": answers[1:] if len(answers) > 1 else [],
            "contexts": contexts,
            "supporting_facts": [],  # 他们数据不一定带支持事实标注
        })
    if no_ctx:
        print(f"  [WARN] {data_source}: {no_ctx} 条没探测到上下文字段（字段名不在已知列表）",
              file=sys.stderr)
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True,
                    help="Graph-R1 的 datasets/ 目录（TeraBox 解压后）")
    ap.add_argument("--datasets", type=str, default="",
                    help="逗号分隔 Graph-R1 数据源名（默认全部 6 个）")
    ap.add_argument("--sample", type=int, default=300)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    names = [n.strip() for n in args.datasets.split(",") if n.strip()] if args.datasets else list(NAME_MAP)

    for ds in names:
        out_name = NAME_MAP[ds]
        items = load_dataset(ds, data_dir, args.sample, args.seed)
        if not items:
            continue
        out = SAMPLES_DIR / f"{out_name}_{args.sample}.json"
        out.write_text(json.dumps({"dataset": out_name, "sample_size": args.sample,
                                   "random_seed": args.seed, "items": items},
                                  ensure_ascii=False, indent=2), encoding="utf-8")
        avg = sum(len(it["contexts"]) for it in items) / max(len(items), 1)
        print(f"  [OK] {ds}: {len(items)} 题 -> {out}（平均上下文 {avg:.1f}，来源 {data_dir/ds}/raw/）")

    print(f"\n完成，文件在 {SAMPLES_DIR}/。跑法: python run.py --method full --dataset {out_name} ...")


if __name__ == "__main__":
    main()
