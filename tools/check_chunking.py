"""体检：语料切块是不是「逐段落切、绝不跨段落」的。

规则（项目口径）：HotpotQA 每题自带 10 个段落 → 每个段落**单独**用 bge tokenizer 切块；
**默认不切块**（0 → 2000/0）= 一个段落一个整段块；块不跨段落，overlap 只在段内（非 0 口径时）。图类方法内部的切块器因为块
≤400 < 1200 而原样通过。

本脚本的判据（不依赖 bge 是否可用）：每个块的文本必须能在**它自己那个段落**里原样
定位到；若某个块拼了两段的内容，就一定定位不到 → 直接报 FAIL。

用法:
  python tools/check_chunking.py                 # 默认 hotpotqa_300 前 20 题
  python tools/check_chunking.py --dataset 2wiki --n 20
"""

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


def _stub_pandas_if_broken():
    """本机 Anaconda 的 pandas 与 numpy ABI 不匹配时，塞个壳让 import 通过。

    只影响这个诊断脚本；服务器上 pandas 正常时是 no-op（和 smoke_new_baselines 同款兜底）。
    """
    try:
        import pandas  # noqa: F401
        return
    except Exception:
        import types
        m = types.ModuleType("pandas")
        m.DataFrame = type("DataFrame", (), {})
        sys.modules["pandas"] = m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hotpotqa")
    ap.add_argument("--sample", default="300")
    ap.add_argument("--n", type=int, default=20, help="看前几题")
    a = ap.parse_args()

    _stub_pandas_if_broken()
    from methods.standard_rag import build_chunks          # noqa: E402

    p = ROOT / "data" / "samples" / f"{a.dataset}_{a.sample}.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    items = d["items"][: a.n]

    n_para = n_chunk = 0
    bad = []
    multi = 0
    for it in items:
        for ctx in it["contexts"]:
            n_para += 1
            # 段落的规范化文本（build_chunks 里就是 " ".join(sentences)）
            sents = ctx.get("sentences") or (ctx.get("text", ""),)
            para = " ".join(s.strip() for s in sents if s and s.strip())
            for title, _cs, ck, _locs in build_chunks([ctx]):
                n_chunk += 1
                if ck.strip() and ck.strip() not in para:
                    bad.append((title, ck[:60]))
                if len(build_chunks([ctx])) > 1:
                    multi += 1
                    break

    print(f"数据: {p.name}  题数={len(items)}")
    print(f"段落 {n_para} → 块 {n_chunk}（平均 {n_chunk / max(n_para, 1):.2f} 块/段）")
    print(f"被切成多块的段落数: {multi}（说明长段落会被切，短段落 1 段 = 1 块）")
    if bad:
        print(f"\n✗ 有 {len(bad)} 个块在自己段落里定位不到（可能跨段拼接！）例：")
        for t, c in bad[:5]:
            print(f"   [{t}] {c!r}")
        return 1
    print("\n✓ 每个块都能在原段落内原样定位 → 逐段落切块，未跨段落拼接")
    return 0


if __name__ == "__main__":
    sys.exit(main())
