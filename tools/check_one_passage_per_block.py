#!/usr/bin/env python
"""核对：当前 Qwen3-Embedding-0.6B 的切块口径下，每个数据集**一篇 passage 是不是一块**。

为什么需要（要被问到的口径问题）：语料单元口径决定了"检索粒度"，而它是全项目单变量对照的
前提。`config.EXTRACT_CHUNK_TOKENS=0` → 归一成 `CHUNK_NO_SPLIT`=2000 的**安全阀**，
所以"不切块"并不是"一定不切"，而是"超过 2000 token 才切"。这个脚本把这层说清楚：

  · token 数 ≤ 阈值 → **1 篇 = 1 块**（这是要核对的结论）
  · token 数 > 阈值 → 会被切成多块（安全阀生效）

判据用的是**真正的切块入口**（`methods.standard_rag.build_chunks`），不是另写一套，
所以结论与 `run.py` 跑出来的一致。前置条件：`transformers` 可用且 `EMBED_MODEL` 指向
含 tokenizer 的目录 —— 否则切块会退化成字符级（脚本会检测并显著警告，那样测的就不是
token 口径了）。

用法::

    python tools/check_one_passage_per_block.py --dataset hotpotqa --sample 500
    python tools/check_one_passage_per_block.py --all            # 5 个数据集各扫一遍
    python tools/check_one_passage_per_block.py --all --limit 50 # 只扫前 50 行（快）
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _count_tokens(texts: list[str]) -> list[int] | None:
    """用切块尺子数 token；拿不到就返回 None（调用方据此警告）。"""
    try:
        from common.llm import _count_tokens_each, tokenizer_failure, CHUNK_TOKENIZER
    except Exception:                                             # noqa: BLE001
        return None
    if tokenizer_failure(CHUNK_TOKENIZER):
        return None
    try:
        return _count_tokens_each(texts, CHUNK_TOKENIZER)
    except Exception:                                             # noqa: BLE001
        return None


def check_dataset(dataset: str, sample: int, limit: int = 0) -> int:
    from config import EXTRACT_CHUNK_TOKENS, CHUNK_NO_SPLIT, dataset_chunking
    from corpus import _merged_from_samples
    from methods.standard_rag import build_chunks

    rows = _merged_from_samples(dataset, sample)
    if limit:
        rows = rows[:limit]
    if not rows:
        print(f"  [{dataset}] ⚠ 没有语料（先 build_eval_samples / build_merged_corpus）")
        return 1

    ct, co = dataset_chunking(dataset)
    # ★ 数 token 必须用**与 `build_chunks` 逐字相同的文本**，否则边界判断会差一个 token：
    #   `build_chunks` 拼的是 `f"{title}\n" + " ".join(sentences)` —— 句间换行被压成**空格**。
    #   若这里用 `title + "\n" + text`（保留原换行），计数结果会与切块路径不同，
    #   于是出现"我数的 max > 阈值、但它没被切"这种自相矛盾（实测踩过，别改回去）。
    def _idx_text(r: dict) -> str:
        title = (r.get("title") or "").strip()
        sents = r.get("sentences") or re.split(r"(?<=[.!?])\s+", r.get("text", ""))
        sents = [s.strip() for s in sents if s.strip()]
        return (f"{title}\n" if title else "") + " ".join(sents)

    texts = [(_idx_text(r), r) for r in rows]
    toks = _count_tokens([t for t, _ in texts])

    n_blocks = sum(len(build_chunks([r], dataset=dataset)) for r in rows)
    _nb = [len(build_chunks([r], dataset=dataset)) for r in rows]
    print(f"\n  ── {dataset}（{len(rows)} 篇 passage；口径 {ct}/{co} token）──")
    if toks is None:
        # ★ 退化路径也要给块数统计**并且不能编造 token 数**。之前这里直接 return，
        #   导致"看起来一切正常"的数字被当成证据（实测踩过：输出写着 max 2076 却 0 篇被切，
        #   而那 2076 根本不是 token 数）。所以这里只报块数、并明确标注口径不可信。
        print("     ⚠ tokenizer 不可用 → 切块**已退化为字符级**，token 数无法给出，"
              "下面的「1 篇 = 1 块」**不能当作 token 口径的证据**。")
        print("       请先确认 EMBED_MODEL 指向含 tokenizer_config.json 的目录（服务器上跑）。")
        _one = sum(1 for k in _nb if k == 1)
        print(f"     块数统计（字符级退化下的结果，仅供参考）：1 篇 = 1 块的有 "
              f"{_one}/{len(rows)} 篇；总块数 {n_blocks}（{n_blocks/len(rows):.2f} 块/篇）")
        return 2

    mult = [i for i, t in enumerate(toks) if _nb[i] > 1]
    one = len(rows) - len(mult)
    print(f"     token 分布：min {min(toks)} / 中位 {sorted(toks)[len(toks)//2]} / max {max(toks)}")
    # ★ 交叉验证（防判据漏判）：`build_chunks` 内部是 `len(ids) <= max_tokens → 1 块`，
    #   所以"被切开的行"里**必须**存在 token > 阈值的行，且 token 最大的行**必须**被切。
    #   两者不一致 = 我这里的 token 计数或阈值口径与切块路径不同，结论不能采信。
    _thr = ct
    _maxi = max(range(len(toks)), key=lambda i: toks[i])
    _incons = []
    if mult:
        if max(toks[i] for i in mult) <= _thr:
            _incons.append(f"有行被切开但它的 token({max(toks[i] for i in mult)}) ≤ 阈值 {_thr}")
    else:
        if toks[_maxi] > _thr:
            _incons.append(f"0 行被切开，但最大 token = {toks[_maxi]} > 阈值 {_thr}"
                           f"（「{str(rows[_maxi].get('title'))[:40]}」）——判据与切块路径不一致")
    if _incons:
        print("     ✗✗ **自检不通过**：" + "；".join(_incons))
        print("        → 本次「1 篇 = 1 块」的结论**不可采信**，请连同输出一起反馈。")
        return 1
    print(f"     （自检通过：阈值 {_thr}，最大 token {toks[_maxi]}，切开 {len(mult)} 篇，"
          f"三者一致）")
    print(f"     1 篇 = 1 块：**{one}/{len(rows)}** 篇（{one/len(rows)*100:.1f}%）")
    print(f"     被切成多块：{len(mult)} 篇（{len(mult)/len(rows)*100:.1f}%）；总块数 {n_blocks}"
          f"（{n_blocks/len(rows):.2f} 块/篇）")
    if mult:
        worst = sorted(mult, key=lambda i: -toks[i])[:3]
        for i in worst:
            print(f"       · {toks[i]:6d} token → {_nb[i]} 块"
                  f"  「{str(rows[i].get('title'))[:50]}」")
    # 结论行：口径是否真的等价于"一篇一块"
    if ct == CHUNK_NO_SPLIT and co == 0:
        verdict = ("是（安全阀 2000 未触发 → 本质是**不切块**）" if not mult
                   else f"**否**：有 {len(mult)} 篇超过 2000 token 被切开")
    else:
        verdict = (f"是（数据集专属口径 {ct}/{co}，且无一篇超过 {ct} token）" if not mult
                   else f"**否**：{len(mult)} 篇超过 {ct} token 被切开（这是 triviaqa 的预期行为）")
    print(f"     → 「1 篇 = 1 块」？{verdict}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="核对一篇 passage 是否就是一块")
    ap.add_argument("--dataset", default="hotpotqa")
    ap.add_argument("--sample", type=int, default=500)
    ap.add_argument("--limit", type=int, default=0, help="只扫前 N 行（0=全部）")
    ap.add_argument("--all", action="store_true", help="5 个数据集各扫一遍")
    ap.add_argument("--require-tokenizer", action="store_true",
                    help="tokenizer 不可用时直接失败（退出码 2）。★ 防止「字符级退化」下"
                         "误判成「一篇一块」——那时判据已不是 token 口径")
    args = ap.parse_args()

    from config import EXTRACT_CHUNK_TOKENS, EXTRACT_CHUNK_OVERLAP, CHUNK_NO_SPLIT, dataset_chunking
    from common.llm import tokenizer_failure, tokenizer_kind, CHUNK_TOKENIZER
    _tok_fail = tokenizer_failure(CHUNK_TOKENIZER)
    print("══ 语料单元口径核对：一篇 passage 是不是一块 ══")
    # ★ 原值 vs 归一后：配置里写 0（=不切块），`config.py` 在 import 时就归一成 2000。
    #   只打"2000"会让人以为"按 2000 切"，所以两个数都打。
    import os as _os
    _raw = _os.environ.get("EXTRACT_CHUNK_TOKENS", "0")
    print(f"  EXTRACT_CHUNK_TOKENS：配置/env 原值 = {_raw}"
          f" → 归一后 = {EXTRACT_CHUNK_TOKENS}（`0` 表示**不切块**，统一归一成安全阀 "
          f"CHUNK_NO_SPLIT={CHUNK_NO_SPLIT}；不是「按 2000 切」）")
    print(f"  全局 overlap = {EXTRACT_CHUNK_OVERLAP}；切块尺子 = {tokenizer_kind()}")
    if _tok_fail:
        print(f"  ⚠️⚠️ 尺子不可用（{_tok_fail}）→ 切块已**退化为字符级**，"
              f"下面的「1 篇 = 1 块」**不能当作 token 口径的证据**。请在有 tokenizer 的机器上重跑。")
    ds_list = ["hotpotqa", "2wiki", "musique", "popqa", "triviaqa"] if args.all else [args.dataset]
    for ds in ds_list:
        check_dataset(ds, args.sample, args.limit)
    print("\n说明：每个数据集的口径由 `config.dataset_chunking(ds)` 决定（triviaqa=1200/100，其余=不切块）。")
    if _tok_fail and args.require_tokenizer:
        print("\n✗ --require-tokenizer：尺子不可用，结论不成立，按失败退出。")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())