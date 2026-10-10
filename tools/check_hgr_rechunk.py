"""体检：HyperGraphRAG 的切块尺子是否和我们一致（会不会被「再切一刀」）。**

背景：我们**不切块**（整段 passage；triviaqa 例外，按数据集切成 1200/100），官方
`operate.chunking_by_token_size` 原本用 **tiktoken**（默认 gpt-4o-mini → o200k_base）
再量一遍；一旦某块超过 `HGR_CHUNK_TOKENS`（默认跟随全局口径 **0 → 2000 的安全阀**）
就被再切（旧口径是 400/40，现已改）→ 索引块数 > 喂入 docs 数、两套重叠叠加，那部分
检索单元变成"半块"，与其它方法的语料单元窗口不同粒度。

现在 `methods/hypergraphrag_adapter.py::_patch_hgr_tokenizer()` 已把官方那把尺子
**换成 bge**（`HGR_TOKENIZER=bge`，默认），两边同尺子 → 官方内部 1 doc = 1 chunk。
设 `HGR_TOKENIZER=tiktoken` 可回到官方原版。

本脚本在这两种模式下都能量，回答两个问题：
  [2] 若走官方原版（tiktoken），我们有多少块会被再切？（= patch 掉了多少风险）
  [3] 走 patch 后（bge 尺子），官方会切出多少块？（期望 == 我们喂入的块数）

用法（服务器上跑）:
  python tools/check_hgr_rechunk.py --dataset 2wiki --sample 300
  python tools/check_hgr_rechunk.py --dataset hotpotqa --sample 300 --limit 500 --show 3
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="2wiki")
    ap.add_argument("--sample", type=int, default=300)
    ap.add_argument("--limit", type=int, default=0, help="只查前 N 块（0=全部）")
    ap.add_argument("--show", type=int, default=3, help="打印前 N 个会超阈的块样例")
    a = ap.parse_args()

    from corpus import merged_contexts
    from common.llm import split_by_tokens, _get_tokenizer
    from config import EXTRACT_CHUNK_TOKENS, EXTRACT_CHUNK_OVERLAP
    from methods.hypergraphrag_adapter import (HGR_CHUNK_TOKENS, HGR_CHUNK_OVERLAP,
                                               hgr_tokenizer_mode, _tiktoken_encoding)

    step = HGR_CHUNK_TOKENS - HGR_CHUNK_OVERLAP

    # [1] 我们的预切块（= methods/hypergraphrag_adapter.py::prechunk_corpus 同一段逻辑）
    docs: list[str] = []
    for c in merged_contexts(a.dataset, a.sample):
        full = f"{c.get('title', '')}\n{c.get('text', '')}".strip()
        for chunk_text, _cs, _ce in split_by_tokens(full, EXTRACT_CHUNK_TOKENS,
                                                    EXTRACT_CHUNK_OVERLAP):
            if chunk_text.strip():
                docs.append(chunk_text.strip())
    total = len(docs)
    if a.limit:
        docs = docs[: a.limit]
    print(f"[1] 我们的块（bge {EXTRACT_CHUNK_TOKENS}/{EXTRACT_CHUNK_OVERLAP}）: {total} 块"
          f"（本次检查 {len(docs)}）")

    # [2] 官方原版尺子（tiktoken）会再切多少 —— 即 patch 掉的真实风险
    enc = _tiktoken_encoding()
    if enc is None:
        print("[2] 跳过：本机没装 tiktoken（服务器上 HyperGraphRAG 自带；"
              "patch 后官方也不用它了）")
    else:
        over = extra = 0
        samples = []
        for i, d in enumerate(docs):
            n = len(enc.encode(d))
            if n > HGR_CHUNK_TOKENS:
                over += 1
                extra += -(-n // step) - 1
                if len(samples) < a.show:
                    samples.append((i, n, d[:90]))
        print(f"[2] 官方原版尺子（tiktoken {enc.name}，阈值 {HGR_CHUNK_TOKENS}）："
              f"超阈 {over}/{len(docs)} 块（{100.0*over/max(1,len(docs)):.1f}%），"
              f"若不 patch 官方会多切出 {extra} 块")
        for i, n, head in samples:
            print(f"      · 块#{i}: {n} tiktoken | {head}...")
        if over:
            print("    → 正是这些块会让检索单元变成「半块」；patch 成 bge 后消失")

    # [3] patch 后的口径（bge 尺子）：官方内部块数必须 == 我们喂入的块数
    mode = hgr_tokenizer_mode()
    tok = _get_tokenizer()
    official_chunks = 0
    over_bge = 0
    for d in docs:
        n = len(tok.encode(d, add_special_tokens=False))
        if n > HGR_CHUNK_TOKENS:
            over_bge += 1
        official_chunks += max(1, -(-n // step))      # 官方 range(0,n,step) 的块数
    print(f"[3] 当前模式 HGR_TOKENIZER={mode}" +
          ("（默认；官方尺子=bge）" if mode == "bge" else
           "（官方原版；官方尺子=tiktoken）"))
    if mode == "bge":
        print(f"    官方内部块数 = {official_chunks}，我们喂入 = {len(docs)}"
              f" → {'一致 ✓（1 doc=1 chunk）' if official_chunks == len(docs) else '不一致 ✗'}"
              f"；超 {HGR_CHUNK_TOKENS} bge 的块 = {over_bge}")
    else:
        print("    ⚠ 你回到了官方原版尺子：官方内部会用自己的 tokenizer 再切，"
              "上面的 [2] 就是后果")
    return 0


if __name__ == "__main__":
    sys.exit(main())
