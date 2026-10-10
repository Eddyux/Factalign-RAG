"""下载并随机抽样 4 个 benchmark（hotpotqa / 2wiki / musique / popqa）各 N 题。

统一 item 格式，存到 data/samples/{name}_{N}.json。

用法:
  python sample_datasets.py                # 4 个各 500
  python sample_datasets.py --sample 100   # 先各 100 试跑
  python sample_datasets.py --datasets hotpotqa,musique
  python sample_datasets.py --seed 42
"""

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from config import SAMPLES_DIR, SAMPLE_SIZE, RANDOM_SEED

SPECS = {
    "hotpotqa": {"path": "hotpotqa/hotpot_qa", "config": "distractor", "split": "validation"},
    "musique":  {"path": "bdsaglam/musique", "config": "answerable", "split": "validation"},
    "popqa":    {"path": "Atipico1/popQA", "config": None, "split": "test"},
    "2wiki":    {"path": "ohjoonhee/2WikiMultihopQA", "config": None, "split": "validation"},
}


def _sample_indices(n, k, seed):
    rng = random.Random(seed)
    return sorted(rng.sample(range(n), min(k, n)))


def _split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def load_hotpotqa(spec, k, seed):
    from datasets import load_dataset
    ds = load_dataset(spec["path"], spec["config"], split=spec["split"])
    items = []
    for i in _sample_indices(len(ds), k, seed):
        item = ds[int(i)]
        ctx = item["context"]
        contexts = [{"title": t.strip(), "text": " ".join(s).strip(), "sentences": [x.strip() for x in s]}
                    for t, s in zip(ctx["title"], ctx["sentences"])]
        sf = item.get("supporting_facts", {})
        facts = [[t.strip(), int(s)] for t, s in zip(sf.get("title", []), sf.get("sent_id", []))]
        items.append({"id": str(item["id"]), "question": item["question"].strip(),
                      "answer": item["answer"].strip(), "answer_aliases": [],
                      "contexts": contexts, "supporting_facts": facts})
    return items


def load_musique(spec, k, seed):
    from datasets import load_dataset
    ds = load_dataset(spec["path"], spec["config"], split=spec["split"])
    items = []
    for i in _sample_indices(len(ds), k, seed):
        item = ds[int(i)]
        contexts, idx_to_pos = [], {}
        for pos, para in enumerate(item.get("paragraphs", [])):
            if isinstance(para, str):
                text, title, sents = para.strip(), f"Doc_{pos}", _split_sentences(para)
            elif isinstance(para, dict):
                text = (para.get("text") or para.get("paragraph_text") or "").strip()
                title = para.get("title") or f"Doc_{pos}"
                sents = para.get("sentences") or (_split_sentences(text) if text else [])
                if "idx" in para:
                    idx_to_pos[para["idx"]] = pos
            else:
                continue
            if text:
                contexts.append({"title": title, "text": text, "sentences": [s.strip() for s in sents if s.strip()]})
        facts = []
        for sidx in item.get("paragraph_support_idx", []):
            pos = idx_to_pos.get(sidx, sidx)
            title = contexts[pos]["title"] if pos < len(contexts) else f"Doc_{pos}"
            facts.append([title, 0])
        answer = (item.get("answer") or "").strip()
        aliases = [a.strip() for a in (item.get("answer_aliases") or []) if a.strip()]
        if answer and answer not in aliases:
            aliases = [answer] + aliases
        elif not answer and aliases:
            answer = aliases[0]
        items.append({"id": str(item["id"]), "question": item["question"].strip(),
                      "answer": answer, "answer_aliases": aliases, "contexts": contexts, "supporting_facts": facts})
    return items


def load_popqa(spec, k, seed):
    from datasets import load_dataset
    ds = load_dataset(spec["path"], split=spec["split"])
    items, missing = [], 0
    for i in _sample_indices(len(ds), k, seed):
        item = ds[int(i)]
        ctxs = item.get("ctxs", [])
        if not ctxs:
            missing += 1
        contexts, hasanswer = [], []
        for pos, para in enumerate(ctxs):
            if isinstance(para, dict):
                text = (para.get("text") or para.get("content") or "").strip()
                title = (para.get("title") or f"Passage_{pos}").strip()
            elif isinstance(para, str):
                text, title = para.strip(), f"Passage_{pos}"
            else:
                continue
            if not text:
                continue
            contexts.append({"title": title, "text": text, "sentences": _split_sentences(text)})
            if isinstance(para, dict) and para.get("hasanswer"):
                hasanswer.append(title)
        answer = (item.get("obj") or "").strip()
        aliases = [a.strip() for a in (item.get("possible_answers") or []) if a.strip()]
        if answer and answer not in aliases:
            aliases = [answer] + aliases
        elif not answer and aliases:
            answer = aliases[0]
        items.append({"id": str(item.get("id", i)), "question": item["question"].strip(),
                      "answer": answer, "answer_aliases": aliases,
                      "contexts": contexts, "supporting_facts": [[t, 0] for t in hasanswer]})
    if missing:
        print(f"  [WARN] popqa: {missing}/{k} 条无 ctxs —— Atipico1/popQA 无预检索上下文，"
              f"请改用 HippoRAG 预检索版", file=sys.stderr)
    return items


def load_2wiki(spec, k, seed):
    from datasets import load_dataset
    ds = load_dataset(spec["path"], split=spec["split"])
    items = []
    for i in _sample_indices(len(ds), k, seed):
        item = ds[int(i)]
        ctx = item["context"]
        contexts = []
        if isinstance(ctx, dict):
            for t, s in zip(ctx["title"], ctx.get("content", ctx.get("sentences", []))):
                contexts.append({"title": t.strip(), "text": " ".join(s).strip(), "sentences": [x.strip() for x in s]})
        else:
            for para in ctx:
                if isinstance(para, (list, tuple)) and len(para) >= 2:
                    contexts.append({"title": para[0].strip(), "text": " ".join(para[1]).strip(),
                                     "sentences": [x.strip() for x in para[1]]})
        sf = item.get("supporting_facts", {})
        facts = []
        if isinstance(sf, dict):
            for t, sid in zip(sf.get("title", []), sf.get("sent_id", [])):
                facts.append([t.strip(), int(sid)])
        items.append({"id": str(item.get("_id", i)), "question": item["question"].strip(),
                      "answer": (item.get("answer") or "").strip(), "answer_aliases": [],
                      "contexts": contexts, "supporting_facts": facts})
    return items


LOADERS = {"hotpotqa": load_hotpotqa, "musique": load_musique, "popqa": load_popqa, "2wiki": load_2wiki}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=SAMPLE_SIZE)
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--datasets", type=str, default="")
    args = ap.parse_args()

    names = [n.strip() for n in args.datasets.split(",") if n.strip()] if args.datasets else list(SPECS)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    for name in names:
        spec = SPECS[name]
        print(f"[{name}] 下载 {spec['path']} ({spec['split']}) 抽样 {args.sample} ...", flush=True)
        try:
            items = LOADERS[name](spec, args.sample, args.seed)
        except Exception as e:
            print(f"  [FAIL] {name}: {e}", file=sys.stderr)
            continue
        out = SAMPLES_DIR / f"{name}_{args.sample}.json"
        out.write_text(json.dumps({"dataset": name, "sample_size": args.sample,
                                   "random_seed": args.seed, "items": items},
                                  ensure_ascii=False, indent=2), encoding="utf-8")
        avg = sum(len(it["contexts"]) for it in items) / max(len(items), 1)
        print(f"  [OK] {len(items)} 题 -> {out}  (平均上下文 {avg:.1f})")

    print(f"\n完成，文件在 {SAMPLES_DIR}/")


if __name__ == "__main__":
    main()
