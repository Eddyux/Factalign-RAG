"""对比两组 test_layer 结果文件：EM/F1 + McNemar 配对精确检验。

为什么需要它：消融的两个配置跑出来是两个 jsonl（每题一行 {id, question, gold,
pred, em, f1}），单看 EM 差值说明不了显著性——论文里要的是**配对**检验：同一道题
两边一对一错才算"分歧题"，用精确二项检验看分歧是否偏向一边。

用法（可给多份（多数据集）合并成一组，按 id 配对）：

  python tools/compare_runs.py \
    --mine  results/test_layer_ablate_desc_2wiki.jsonl \
            results/test_layer_ablate_desc_hotpotqa.jsonl \
            results/test_layer_ablate_desc_musique.jsonl \
    --other results/test_layer_full_2wiki.jsonl \
            results/test_layer_full_hotpotqa.jsonl \
            results/test_layer_full_musique.jsonl \
    --label-mine 描述式 --label-other 实例化

输出：两组 EM/F1、Δ、分歧题分布（净胜），以及 McNemar 精确检验的双尾 p 值。
p<0.05 即可写"显著"，并在论文里标注净胜题数。
"""

import argparse
import json
import math
from pathlib import Path


def load(paths: list[str]) -> dict:
    rows = {}
    for p in paths:
        path = Path(p)
        if not path.exists():
            raise SystemExit(f"结果文件不存在：{p}")
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                rows[str(d["id"])] = d
    return rows


def mcnemar_exact(b: int, c: int) -> float:
    """McNemar 精确检验（双尾）：分歧题里"偏向一边"的概率是否极端。

    b = 只 mine 对的题数，c = 只 other 对的题数；n=b+c 个分歧题，零假设 p=0.5。
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def _stats(rows: dict, ids: list[str]):
    n = len(ids)
    em = sum(float(rows[i].get("em", 0.0)) for i in ids)
    f1 = sum(float(rows[i].get("f1", 0.0)) for i in ids)
    return (em / n if n else 0.0, f1 / n if n else 0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mine", nargs="+", required=True, help="实验组结果 jsonl（可多份）")
    ap.add_argument("--other", nargs="+", required=True, help="对照组结果 jsonl（可多份）")
    ap.add_argument("--label-mine", default="mine")
    ap.add_argument("--label-other", default="other")
    ap.add_argument("--show", type=int, default=5,
                    help="打印前 N 道分歧题（0=不打印）")
    args = ap.parse_args()

    M, O = load(args.mine), load(args.other)
    ids = sorted(set(M) & set(O), key=lambda x: (len(x), x))
    only_m = len(set(M) - set(O))
    only_o = len(set(O) - set(M))
    if not ids:
        raise SystemExit("两份结果没有共同的 id，无法配对")

    em_m, f1_m = _stats(M, ids)
    em_o, f1_o = _stats(O, ids)

    both = only_mine = only_other = neither = 0
    mine_only_ids, other_only_ids = [], []
    for i in ids:
        a = bool(M[i].get("em"))
        b = bool(O[i].get("em"))
        if a and b:
            both += 1
        elif a:
            only_mine += 1
            mine_only_ids.append(i)
        elif b:
            only_other += 1
            other_only_ids.append(i)
        else:
            neither += 1

    p = mcnemar_exact(only_mine, only_other)
    net = only_mine - only_other

    print(f"配对题数 {len(ids)}"
          + (f"（仅 {args.label_mine} 有 {only_m} 题，仅 {args.label_other} 有 {only_o} 题）"
             if only_m or only_o else ""))
    print(f"{'':<12}{'EM':>8}{'F1':>8}")
    print(f"{args.label_mine:<12}{em_m:>8.4f}{f1_m:>8.4f}")
    print(f"{args.label_other:<12}{em_o:>8.4f}{f1_o:>8.4f}")
    print(f"{'Δ':<12}{em_m - em_o:>+8.4f}{f1_m - f1_o:>+8.4f}")
    print()
    print(f"配对 2x2：都对 {both} | 只 {args.label_mine} 对 {only_mine} | "
          f"只 {args.label_other} 对 {only_other} | 都错 {neither}")
    print(f"净胜 {net:+d} 题；McNemar 精确检验 双尾 p = {p:.4g}"
          + ("  → 显著(p<0.05)" if p < 0.05 else "  → 不显著"))
    if args.show and (mine_only_ids or other_only_ids):
        print(f"\n只 {args.label_mine} 对的题（前 {args.show}）：")
        for i in mine_only_ids[:args.show]:
            print(f"  [{i}] gold={M[i]['gold']!r} other={O[i]['pred']!r} "
                  f"mine={M[i]['pred']!r}")
        print(f"只 {args.label_other} 对的题（前 {args.show}）：")
        for i in other_only_ids[:args.show]:
            print(f"  [{i}] gold={O[i]['gold']!r} mine={M[i]['pred']!r} "
                  f"other={O[i]['pred']!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
