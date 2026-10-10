#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ircot 的 few-shot 示例段落 vs 我们的语料：**标题重合度自检**（gold-leak 体检）。

为什么需要这个检查
------------------
官方 IRCoT 的 prompt 文件（`baselines/ircot-main/prompts/hotpotqa/gold_with_2_distractors_
context_cot_qa_codex.txt`）把**示例问题的检索段落原文**直接写进了 prompt：

    # METADATA: {"qid": "5a8ed9f355429917b4a5bddd"}
    Wikipedia Title: Nobody Loves You (When You're Down and Out)
    "Nobody Loves You..." is a song written by John Lennon released on ...

我们在 `IRCOT_PROMPT_FILTER=1`（默认）下只保留官方 jsonnet 里那 **20 个 valid_qid**
对应的示例（这是官方自己 `filter_by_key_values={"qid": valid_qids}` 的口径，对齐没错）。
但这 20 条示例**一共提到 80 个 Wikipedia 标题**，而这些标题也都在我们的 BM25 索引里
（索引 = 本题 gold + distractors，见 `run.py` 组装语料的方式）。

后果：如果某个**测试题**的语料里正好含这 80 个标题之一，那么
  ① 该段落的内容会**同时**出现在 prompt 的示例里 → 模型可以"照抄示例"；
  ② 它还会被 BM25 检索到 → 一轮就命中，
于是这一题的分数**不可比**（有便宜可占）。这不是我们引入的偏离，是**官方 prompt 文件
自带的**性质；但复现官方时必须知道它有多大、命中哪几题，才能在论文里说清楚。

用法（服务器，语料在那边）：
    python tools/check_ircot_demo_leak.py hotpotqa
    python tools/check_ircot_demo_leak.py hotpotqa --corpus data/merged/hotpotqa_500/corpus.jsonl
    python tools/check_ircot_demo_leak.py hotpotqa --questions data/samples/hotpotqa_500.json

退出码：0 = 无重合；3 = 有重合（**不一定是错**，但必须知情）。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from methods.ircot_adapter import (  # noqa: E402
    _metadata_examples, _official_valid_qids, IRCOT_PROMPT_ROOT, _dataset_dir)


def _demo_titles(dataset: str, n_distractors: int = 2) -> set[str]:
    sub = _dataset_dir.get(dataset, dataset)
    p = IRCOT_PROMPT_ROOT / sub / f"gold_with_{n_distractors}_distractors_context_cot_qa_codex.txt"
    if not p.exists():
        print(f"[x] 找不到官方 prompt：{p}")
        return set()
    # ★ `_metadata_examples(text)` 收的是**文本**，不是路径（传路径会静默返回 1 条废示例）
    ex = _metadata_examples(p.read_text(encoding="utf-8"))
    qids = _official_valid_qids(dataset)
    kept = [e for e in ex if e.get("qid") in qids]
    kept.sort(key=lambda e: qids.index(e["qid"]))
    titles: set[str] = set()
    for e in kept:
        for ln in e.get("lines") or []:
            m = re.match(r"Wikipedia Title:\s*(.+)", ln.strip())
            if m:
                titles.add(m.group(1).strip())
    print(f"[demo] 官方 prompt = {p.name}")
    print(f"[demo] 文件里 {len(ex)} 条 → 按官方 valid_qid 过滤后 {len(kept)} 条"
          f"（qid 表 {len(qids)} 个）→ 提到 {len(titles)} 个不同 Wikipedia 标题")
    return titles


def _corpus_titles(paths: list[Path]) -> dict[str, int]:
    """corpus.jsonl → {title: 出现次数}。兼容 dict / list 两种顶层形状。"""
    out: dict[str, int] = {}
    for p in paths:
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                items = rec if isinstance(rec, list) else [rec]
                for it in items:
                    if not isinstance(it, dict):
                        continue
                    t = (it.get("title") or "").strip()
                    if t:
                        out[t] = out.get(t, 0) + 1
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?", default="hotpotqa")
    ap.add_argument("--corpus", default="", help="corpus.jsonl 路径；默认自动找 data/merged/<ds>_*/")
    ap.add_argument("--questions", default="", help="samples json；给了就按题报哪些题的语料命中")
    ap.add_argument("--distractors", type=int, default=2)
    args = ap.parse_args(argv[1:])

    demo = _demo_titles(args.dataset, args.distractors)
    if not demo:
        return 1

    if args.corpus:
        cps = [Path(args.corpus)]
    else:
        cps = sorted((ROOT / "data" / "merged").glob(f"{args.dataset}_*/corpus.jsonl"))
    if not cps:
        print(f"[x] 没找到语料（data/merged/{args.dataset}_*/corpus.jsonl）—— 语料在服务器上，"
              f"用 --corpus 指路径。")
        print(f"\n[demo] 这 {len(demo)} 个标题（前 20 个）：")
        for t in sorted(demo)[:20]:
            print(f"   · {t}")
        return 2

    ct = _corpus_titles(cps)
    print(f"[corpus] {len(cps)} 个文件 → {len(ct)} 个不同标题")
    hit = sorted(demo & set(ct))
    print(f"\n[命中] 示例标题 ∩ 语料标题 = {len(hit)} / {len(demo)} "
          f"= {100.0 * len(hit) / len(demo):.1f}%")
    for t in hit:
        print(f"   · {t}   （语料中出现 {ct[t]} 次）")

    if args.questions:
        qp = Path(args.questions)
        try:
            qs = json.loads(qp.read_text(encoding="utf-8"))
        except Exception as e:                       # noqa: BLE001
            print(f"[x] 读不了 {qp}：{e}")
            return 1
        if isinstance(qs, dict):
            qs = qs.get("data") or qs.get("questions") or list(qs.values())
        bad = []
        for q in qs if isinstance(qs, list) else []:
            if not isinstance(q, dict):
                continue
            titles = set()
            for c in (q.get("contexts") or q.get("context") or []):
                if isinstance(c, dict) and c.get("title"):
                    titles.add(str(c["title"]).strip())
                elif isinstance(c, (list, tuple)) and c:
                    titles.add(str(c[0]).strip())
            h = titles & demo
            if h:
                bad.append((q.get("_id") or q.get("qid") or q.get("id"), sorted(h)))
        print(f"\n[逐题] 语料里含示例段落的题 = {len(bad)} / {len(qs)} "
              f"= {100.0 * len(bad) / max(1, len(qs)):.1f}%   ← 这些题有'照抄示例'的便宜")
        for qid, h in bad[:30]:
            print(f"   · {qid}: {h}")

    return 3 if hit else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
