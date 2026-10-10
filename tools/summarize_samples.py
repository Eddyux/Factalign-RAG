"""把 data/samples/*.json 的规模信息汇总成一张表（论文 Dataset 表直接用）。

每个样本文件在构建时已经写好了 stats 头（题数/段数/去重段数/≈token/≈块数），
本脚本把它们汇总成 data/samples/SUMMARY.md（人读）+ 打印到终端。

用法:
  python tools/summarize_samples.py
  python tools/summarize_samples.py --markdown data/samples/SUMMARY.md
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import SAMPLES_DIR


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markdown", default=str(SAMPLES_DIR / "SUMMARY.md"),
                    help="Markdown 输出路径（默认 data/samples/SUMMARY.md）")
    a = ap.parse_args()

    files = sorted(SAMPLES_DIR.glob("*.json"))
    if not files:
        print(f"没有样本文件（{SAMPLES_DIR}/*.json）")
        return 1

    rows = []
    for f in files:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  ! 读不了 {f.name}: {type(e).__name__}")
            continue
        it = d.get("items") or []
        s = d.get("stats") or {}
        # stats 缺失（旧文件）就现算最关键的几项
        if not s:
            ctx = [len(x.get("contexts") or []) for x in it]
            s = {"n": len(it),
                 "avg_ctx_per_q": (sum(ctx) / len(ctx)) if ctx else 0.0,
                 "paras_uniq": None, "tok_est": None, "blocks_est": None}
        rows.append({
            "file": f.name,
            "dataset": d.get("dataset", f.stem),
            "n": s.get("n", len(it)),
            "seed": d.get("random_seed"),
            "source": (d.get("source") or {}).get("path", ""),
            "split": (d.get("source") or {}).get("split", ""),
            "avg_ctx": s.get("avg_ctx_per_q"),
            "paras": s.get("paras_uniq"),
            "tok": s.get("tok_est"),
            "blocks": s.get("blocks_est"),
            "built_at": d.get("built_at", ""),
            "builder": d.get("builder", ""),
        })

    def fmt(v, n=1):
        return "—" if v is None else (f"{v:,.{n}f}" if isinstance(v, float) else f"{v:,}")

    lines = ["# 评测集汇总（自动生成，勿手改）", "",
             f"来源目录：`{SAMPLES_DIR}`　生成时间：{__import__('time').strftime('%Y-%m-%d %H:%M:%S')}", "",
             "| 数据集 | 文件 | 题数 | seed | 段/题 | 段落(去重) | ≈token | ≈块数 | 来源 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        lines.append(
            f"| {r['dataset']} | `{r['file']}` | {r['n']:,} | {r['seed']} | "
            f"{fmt(r['avg_ctx'])} | {fmt(r['paras'])} | {fmt(r['tok'])} | {fmt(r['blocks'])} | "
            f"`{r['source']}`/{r['split']} |")
    lines += ["",
              "说明：",
              "- **段/题** = 每题平均上下文段落数；**段落(去重)** = 全样本按 (title,text) 去重后的段数",
              "- **≈token** = 按 4 字符/token 粗估；**≈块数** = 按项目语料单元切块粗估（默认不切块 → 一块/段）",
              "- 全部来自公开数据集的原生 split，只做固定 seed 随机抽样，不做题目/上下文筛选",
              ""]
    out = Path(a.markdown)
    body = "\n".join(lines)
    out.write_text(body, encoding="utf-8")

    print(f"{'数据集':<11}{'文件':<22}{'题数':>6}{'段/题':>7}{'段落':>9}{'≈token':>11}{'≈块数':>9}")
    for r in rows:
        print(f"{r['dataset']:<11}{r['file']:<22}{r['n']:>6}{fmt(r['avg_ctx']):>7}"
              f"{fmt(r['paras']):>9}{fmt(r['tok']):>11}{fmt(r['blocks']):>9}")
    print(f"\n已写: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
