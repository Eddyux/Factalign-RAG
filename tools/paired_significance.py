#!/usr/bin/env python
"""配对显著性 / 效应量诊断：比较两个 tag 的 test_layer 结果。

为什么需要这个脚本
------------------
300 题上 1 EM = 3 题。配对 EM 差值的标准误 ≈ sqrt(不一致对数)/n，
在不一致对数 ~45 时约 2.2 个百分点 —— 也就是说 **"+1.x EM" 常常连 1 个标准误都不到**。
"重跑稳定" 与 "跨题显著" 是两回事：前者只排除采样噪声，后者才回答
"这个差异能不能推广到别的题"。本脚本算的是后者。

用法
----
  python tools/paired_significance.py --a ablate_desc --b ablate_stmt
  python tools/paired_significance.py --a ablate_desc --b stmt --datasets 2wiki hotpotqa musique
  python tools/paired_significance.py --a ablate_desc --b stmt --flip-cases 20

读：results/test_layer_{tag}_{dataset}.jsonl
每行：{"id", "question", "gold", "pred", "em", "f1"}
比较方向：b - a（正数 = b 更好）
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

DEFAULT_DATASETS = ["2wiki", "hotpotqa", "musique"]


# ─────────────────────────── 基础统计 ───────────────────────────

def binom_test_two_sided(k: int, n: int, p: float = 0.5) -> float:
    """精确二项检验（small-p-values 法，等价 scipy.stats.binomtest 双侧）。"""
    if n == 0:
        return 1.0
    pk = math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
    total = 0.0
    for i in range(n + 1):
        pi = math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
        if pi <= pk + 1e-12:
            total += pi
    return min(1.0, total)


def holm(pvals: list[float]) -> list[float]:
    """Holm 逐步校正，返回与输入同序的校正后 p 值。"""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * pvals[i])
        running = max(running, val)
        adj[i] = running
    return adj


def paired_bootstrap(em_a, em_b, f1_a, f1_b, n_boot: int = 10000, seed: int = 0):
    """按题重采样，给 EM / F1 差值做百分位 CI。"""
    n = len(em_a)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    d_em = em_b[idx].mean(axis=1) - em_a[idx].mean(axis=1)
    d_f1 = f1_b[idx].mean(axis=1) - f1_a[idx].mean(axis=1)
    return (
        float(np.percentile(d_em, 2.5)), float(np.percentile(d_em, 97.5)),
        float(np.percentile(d_f1, 2.5)), float(np.percentile(d_f1, 97.5)),
    )


def required_n(delta: float, disc_rate: float, power_z: float = 2.80) -> float:
    """80% 功效、α=0.05 下检出 |delta| 所需的题目数（配对二项近似）。

    n ≈ (z_{α/2}+z_β)^2 * p_disc / delta^2 ；power_z=2.80 对应 80% 功效。
    """
    if abs(delta) < 1e-9 or disc_rate <= 0:
        return float("inf")
    return (power_z ** 2) * disc_rate / (delta ** 2)


# ─────────────────────────── 数据读取 ───────────────────────────

def load(tag: str, dataset: str, results_dir: Path = RESULTS_DIR) -> dict[str, dict] | None:
    path = results_dir / f"test_layer_{tag}_{dataset}.jsonl"
    if not path.exists():
        return None
    rows: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = str(r.get("id", ""))
            if rid:
                rows[rid] = r  # 同 id 后者覆盖（重跑 = 替换语义）
    return rows or None


def align(rows_a: dict[str, dict], rows_b: dict[str, dict]):
    ids = sorted(set(rows_a) & set(rows_b))
    em_a = np.array([float(rows_a[i].get("em", 0)) for i in ids])
    em_b = np.array([float(rows_b[i].get("em", 0)) for i in ids])
    f1_a = np.array([float(rows_a[i].get("f1", 0.0)) for i in ids])
    f1_b = np.array([float(rows_b[i].get("f1", 0.0)) for i in ids])
    return ids, em_a, em_b, f1_a, f1_b


# ─────────────────────────── 单数据集分析 ───────────────────────────

def _summary(name: str, ids, em_a, em_b, f1_a, f1_b, n_boot: int) -> dict:
    n = len(em_a)
    d_em = float(em_b.mean() - em_a.mean())
    d_f1 = float(f1_b.mean() - f1_a.mean())
    win_b = int(((em_a == 0) & (em_b == 1)).sum())   # a 错 b 对
    win_a = int(((em_a == 1) & (em_b == 0)).sum())   # a 对 b 错
    disc = win_b + win_a
    disc_rate = disc / n if n else 0.0
    ci = paired_bootstrap(em_a, em_b, f1_a, f1_b, n_boot=n_boot)
    return {
        "dataset": name, "n": n,
        # 均值（float）
        "em_a_mean": float(em_a.mean()), "em_b_mean": float(em_b.mean()), "d_em": d_em,
        "f1_a_mean": float(f1_a.mean()), "f1_b_mean": float(f1_b.mean()), "d_f1": d_f1,
        # 检验
        "win_b": win_b, "win_a": win_a, "disc": disc,
        "disc_rate": disc_rate,
        "p": binom_test_two_sided(min(win_b, win_a), disc) if disc else 1.0,
        "ci_em": (ci[0], ci[1]), "ci_f1": (ci[2], ci[3]),
        "n_req": required_n(d_em, disc_rate),
        # 部分分诊断：EM 错但 F1>0（"方向对了、span/格式不对"）
        "partial_a": int(((em_a == 0) & (f1_a > 0)).sum()),
        "partial_b": int(((em_b == 0) & (f1_b > 0)).sum()),
        # 逐题数组（用于 pooled / 明细）
        "ids": ids, "em_a": em_a, "em_b": em_b, "f1_a": f1_a, "f1_b": f1_b,
    }


def analyse(dataset: str, rows_a: dict, rows_b: dict, n_boot: int) -> dict:
    ids, em_a, em_b, f1_a, f1_b = align(rows_a, rows_b)
    if not ids:
        raise SystemExit(f"[{dataset}] 两个文件没有共同 id，无法配对")
    r = _summary(dataset, ids, em_a, em_b, f1_a, f1_b, n_boot)
    r["rows_a"], r["rows_b"] = rows_a, rows_b
    return r


def analyse_pooled(per_ds: list[dict], n_boot: int) -> dict:
    """合并数据集（题目直接拼接，不是 dataset 级平均）。"""
    ids = None
    em_a = np.concatenate([r["em_a"] for r in per_ds])
    em_b = np.concatenate([r["em_b"] for r in per_ds])
    f1_a = np.concatenate([r["f1_a"] for r in per_ds])
    f1_b = np.concatenate([r["f1_b"] for r in per_ds])
    r = _summary("POOLED", ids, em_a, em_b, f1_a, f1_b, n_boot)
    r["rows_a"], r["rows_b"] = None, None
    return r


# ─────────────────────────── 输出 ───────────────────────────

def pct(x: float) -> str:
    return f"{100 * x:+.2f}"


def print_table(tag_a: str, tag_b: str, rows: list[dict]) -> None:
    print(f"\n比较方向：{tag_b} - {tag_a}（正数 = {tag_b} 更好）\n")
    hdr = (f"{'数据集':<10}{'n':>5}{'EM(a)':>9}{'EM(b)':>9}{'ΔEM':>9}"
           f"{'ΔF1':>9}{'b胜':>6}{'a胜':>6}{'p(McNemar)':>12}{'p(Holm)':>10}"
           f"{'95%CI(ΔEM)':>20}")
    print(hdr)
    print("-" * len(hdr))
    padj = holm([r["p"] for r in rows])
    for r, pa in zip(rows, padj):
        star = " *" if pa < 0.05 else ""
        ci = f"[{pct(r['ci_em'][0])},{pct(r['ci_em'][1])}]"
        print(f"{r['dataset']:<10}{r['n']:>5}"
              f"{100 * r['em_a_mean']:>9.2f}{100 * r['em_b_mean']:>9.2f}"
              f"{pct(r['d_em']):>9}{pct(r['d_f1']):>9}"
              f"{r['win_b']:>6}{r['win_a']:>6}"
              f"{r['p']:>12.4f}{pa:>10.4f}{ci:>20}{star}")
    print("\n列说明：b胜 = a 错 b 对的题数（净增益的主要来源）；a胜 = a 对 b 错的题数。")
    print("        ΔEM/ΔF1/CI 单位都是百分点；Holm 校正后 p<0.05 标 *。")


def print_detail(rows: list[dict]) -> None:
    print("\n" + "=" * 72)
    print("效应量与可行性")
    print("=" * 72)
    for r in rows:
        print(f"\n[{r['dataset']}] n={r['n']}  不一致对数={r['disc']} "
              f"({100 * r['disc_rate']:.1f}%)")
        if r["disc"]:
            se = math.sqrt(r["disc"]) / r["n"]
            print(f"  配对 EM 差值 SE ≈ sqrt({r['disc']})/{r['n']} = "
                  f"{100 * se:.2f} 个百分点")
            print(f"  ΔEM = {pct(r['d_em'])} 点 → {abs(r['d_em']) / se:.2f} 个标准误")
        print(f"  95% CI：ΔEM [{pct(r['ci_em'][0])}, {pct(r['ci_em'][1])}]  "
              f"ΔF1 [{pct(r['ci_f1'][0])}, {pct(r['ci_f1'][1])}]")
        if r["n_req"] != float("inf"):
            print(f"  80% 功效下检出 ΔEM={pct(r['d_em'])} 点约需 "
                  f"n≈{int(math.ceil(r['n_req']))} 题/数据集（当前 {r['n']}）")
        print(f"  部分分诊断：EM=0 但 F1>0 的题数  a={r['partial_a']}  "
              f"b={r['partial_b']}")


def print_flips(res: list[dict], limit: int) -> None:
    """把 EM 翻转的题打出来，人工看是「检索没到」还是「抽取/格式错」。"""
    print("\n" + "=" * 72)
    print(f"EM 翻转案例（最多 {limit} 条/方向）")
    print("=" * 72)
    shown_b = shown_a = 0
    for r in res:
        if r["rows_a"] is None:
            continue
        for i, rid in enumerate(r["ids"]):
            a_ok, b_ok = r["em_a"][i], r["em_b"][i]
            if a_ok == b_ok:
                continue
            if b_ok == 1 and shown_b >= limit:
                continue
            if a_ok == 1 and shown_a >= limit:
                continue
            if shown_b >= limit and shown_a >= limit:
                return
            ra, rb = r["rows_a"][rid], r["rows_b"][rid]
            who = "b 胜" if b_ok == 1 else "a 胜"
            shown_b += int(b_ok == 1)
            shown_a += int(a_ok == 1)
            print(f"\n[{r['dataset']}] {who} | id={rid}")
            print(f"  Q   : {str(ra.get('question', ''))[:110]}")
            print(f"  gold: {ra.get('gold', '')!r}")
            print(f"  a   : {ra.get('pred', '')!r}  (em={a_ok:.0f} f1={ra.get('f1', 0):.2f})")
            print(f"  b   : {rb.get('pred', '')!r}  (em={b_ok:.0f} f1={rb.get('f1', 0):.2f})")
    if shown_a + shown_b == 0:
        print("（无翻转案例）")


# ─────────────────────────── 入口 ───────────────────────────

def main() -> None:
    try:  # Windows GBK 控制台：宁可替换字符也不要崩
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="基线 tag（results/test_layer_{a}_{ds}.jsonl）")
    ap.add_argument("--b", required=True, help="对照 tag（正数 = b 更好）")
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    ap.add_argument("--results-dir", default=str(RESULTS_DIR),
                    help="结果目录（默认 <repo>/results）")
    ap.add_argument("--n-boot", type=int, default=10000, help="bootstrap 次数")
    ap.add_argument("--flip-cases", type=int, default=0,
                    help=">0：打印每个方向最多 N 条 EM 翻转案例（人工分类用）")
    args = ap.parse_args()

    rdir = Path(args.results_dir)
    res = []
    for ds in args.datasets:
        ra, rb = load(args.a, ds, rdir), load(args.b, ds, rdir)
        if ra is None or rb is None:
            missing = args.a if ra is None else args.b
            print(f"[跳过] {ds}: 缺 results/test_layer_{missing}_{ds}.jsonl", file=sys.stderr)
            continue
        res.append(analyse(ds, ra, rb, args.n_boot))

    if not res:
        raise SystemExit("没有任何数据集可比 —— 检查 --a/--b 的 tag 是否与文件名一致")

    rows = list(res)
    if len(res) > 1:
        rows.append(analyse_pooled(res, args.n_boot))

    print_table(args.a, args.b, rows)
    print_detail(rows)
    if args.flip_cases > 0:
        print_flips(rows, args.flip_cases)

    print("\n提醒：Holm 校正后的 p 值才是逐数据集该报的；POOLED 行只作补充。")
    print("      '重跑稳定' ≠ '跨题显著'：本表衡量的是后者。")


if __name__ == "__main__":
    main()
