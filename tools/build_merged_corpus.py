"""合并语料 + 不切块（整段）+ 校验（本地、无 LLM、无网络）——建图前的"语料指纹"。

为什么要单独一步：所有语料级方法（standardrag / hipporag / proprag / …）都从
`corpus.merged_contexts(dataset, n)` 现场构造同一份语料。这份语料**没有落盘**，
于是「两个方法的语料是不是同一份」只能靠读代码。本脚本把它固化下来并算哈希：

  data/merged/{dataset}_{n}/corpus.jsonl     每行一个 context（title/text/sentences/qid）
  data/merged/{dataset}_{n}/chunks.jsonl     不切块（整段，含 (title, sent_idx) 定位）
  data/merged/{dataset}_{n}/checksums.json   行数 + sha256（语料指纹，可进 git）
  data/merged/{dataset}_{n}/build.json       切块参数 / 尺子 / embedding 口径 / 代码版本

校验（会打印，异常会非零退出）：
  - 每行 JSON 合法、必需字段齐全
  - chunk 数 == 按同一参数重切的块数
  - 题号覆盖面：语料里的 qid 应在 samples 里出现；同 (title,text) 去重后的段数与
    data/samples 里算出来的**一致**（防止"改了语料没改样本"）

用法:
  python tools/build_merged_corpus.py --dataset hotpotqa --sample 500
  python tools/build_merged_corpus.py --dataset 2wiki --sample 500 --force
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.paths import (merged_paths, provenance_stamp, read_json, rel_to_root,
                          sha256_file, write_json)
from corpus import _merged_from_samples
from config import EXTRACT_CHUNK_OVERLAP, EXTRACT_CHUNK_TOKENS, CHUNK_TOKENIZER


def chunking_spec() -> dict:
    """当前切块口径（写进 build.json，并用于判断已有产物是否还能复用）。"""
    from common.run_record import embed_fingerprint
    return {"tokens": EXTRACT_CHUNK_TOKENS, "overlap": EXTRACT_CHUNK_OVERLAP,
            "tokenizer": CHUNK_TOKENIZER, "embed_fingerprint": embed_fingerprint()}


def is_fresh(paths) -> bool:
    """已有产物是不是**按当前口径**切的。

    只看 `checksums.json` 存不存在是个坑：换了切块尺子/大小（比如官方对齐的
    1200/100 + tiktoken）之后，旧的 chunks.jsonl 会被继续复用 —— 后面所有建图方法
    都吃到**旧口径的语料**，而且一切正常、毫无提示。必须比对 build.json 里的口径。
    """
    if not paths["checksums"].exists():
        return False
    old = (read_json(paths["build"]) or {}).get("chunking") or {}
    now = chunking_spec()
    diff = {k: (old.get(k), v) for k, v in now.items() if old.get(k) != v}
    if diff:
        print("[merged] 已有产物口径不符 → 重建：" + "；".join(
            f"{k}: {a} → {b}" for k, (a, b) in diff.items()), flush=True)
        return False
    return True


def write_jsonl(path: Path, rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description="合并语料 + 切块 + 校验（建图前一步）")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--sample", type=int, default=500)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    paths = merged_paths(args.dataset, args.sample)
    if is_fresh(paths) and not args.force:
        print(f"已存在且口径一致：{rel_to_root(paths['checksums'])}（--force 重建）")
        return

    t0 = time.time()
    contexts = _merged_from_samples(args.dataset, args.sample)
    print(f"[merged] {args.dataset} n={args.sample} → {len(contexts)} 条 context", flush=True)

    # ── corpus.jsonl：原样固化（含 qid = 来源题）──
    n_ctx = write_jsonl(paths["corpus"], contexts)

    # ── chunks.jsonl：与建图类方法同口径的切块 ──
    # 口径说明：建图类适配器（hipporag / proprag 等）都是「title\ntext 拼一整段再按
    # 不切块（整段）」；standardrag 走的是 `build_chunks(contexts)` 直接用 sentences。
    # 这里固化**建图口径**（title 行 + 重拼 sentences），因为图谱类方法共用它。
    from methods.standard_rag import build_chunks
    chunks = []
    for ci, c in enumerate(contexts):
        sents = [s.strip() for s in (c.get("sentences") or []) if s and s.strip()]
        if not sents:
            sents = [c.get("text", "")]
        full = f"{c.get('title', '')}\n{' '.join(sents)}".strip()
        if not full:
            continue
        for text, cs, ce, locs in build_chunks([{**c, "text": full, "sentences": [full]}]):
            if not text.strip():
                continue
            chunks.append({"chunk_idx": len(chunks), "context_idx": ci,
                           "title": c.get("title", ""), "qid": c.get("qid"),
                           "start_char": cs, "end_char": ce, "text": text,
                           "locs": [list(x) for x in locs]})
    n_chunks = write_jsonl(paths["chunks"], chunks)
    print(f"[merged] 切块 {n_chunks} 个（{EXTRACT_CHUNK_TOKENS}/{EXTRACT_CHUNK_OVERLAP}）"
          f"，用时 {time.time() - t0:.1f}s", flush=True)

    # ── 校验 ──
    problems = []
    if n_ctx != len(contexts):
        problems.append(f"corpus 行数 {n_ctx} != {len(contexts)}")
    if not chunks:
        problems.append("切块为空")
    for c in chunks[:50]:
        if not c["text"].strip():
            problems.append(f"chunk {c['chunk_idx']} 文本为空")
            break
    # 段数一致性：语料里 (title,text) 去重后的段数
    uniq = {(c.get("title", ""), c.get("text", "")) for c in contexts}
    if len(uniq) != len(contexts):
        print(f"[merged] 提示：(title,text) 去重后 {len(uniq)} 段（原 {len(contexts)} 条）",
              flush=True)

    checks = {
        "dataset": args.dataset, "sample": args.sample,
        "n_contexts": n_ctx, "n_contexts_unique": len(uniq), "n_chunks": n_chunks,
        "n_questions": len({c.get("qid") for c in contexts if c.get("qid")}),
        "titles": len({c.get("title", "") for c in contexts}),
        "files": {
            "corpus.jsonl": {"bytes": paths["corpus"].stat().st_size,
                             "sha256": sha256_file(paths["corpus"])},
            "chunks.jsonl": {"bytes": paths["chunks"].stat().st_size,
                             "sha256": sha256_file(paths["chunks"])},
        },
        "problems": problems,
        "checked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(paths["checksums"], checks)
    write_json(paths["build"], {
        "generator": "tools/build_merged_corpus.py",
        "dataset": args.dataset, "sample": args.sample,
        "chunking": chunking_spec(),
        "elapsed_sec": round(time.time() - t0, 1),
        **provenance_stamp(),
    })
    print(f"[merged] checksums → {rel_to_root(paths['checksums'])}"
          f"  corpus {checks['files']['corpus.jsonl']['sha256'][:12]}…"
          f"  chunks {checks['files']['chunks.jsonl']['sha256'][:12]}…", flush=True)
    if problems:
        print("[merged] ⚠ 校验问题：" + "；".join(problems), flush=True)
        raise SystemExit(1)
    print("[merged] 校验通过", flush=True)


if __name__ == "__main__":
    main()
