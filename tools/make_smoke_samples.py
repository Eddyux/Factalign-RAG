"""从 data/samples/{ds}_500.json 各取前 10 题，做 smoke 测试集。

用途：一条命令快速验证「管线通不通」（建图能跑、检索能出数），不追求指标。
取的是 **seed=42 那次抽样的前 10 题**（原始抽样的前缀，不是另外挑的）——
所以口径与正式样本一致，能复现、也不是"专挑好题"。

产物**平铺在 data/samples/**：`{ds}_10.json`（`common/data.py` 只认这个位置/命名，
放子目录它会找不到）。原 500 题文件不动。

跑法（run.py 的 --sample 10 会自动找到这些文件）：
    python run.py --method <m> --dataset <ds> --sample 10 --workers 4

用法：
    python tools/make_smoke_samples.py                 # 每个数据集取前 10 题
    python tools/make_smoke_samples.py --n 5           # 改题数
    python tools/make_smoke_samples.py --datasets hotpotqa,2wiki
    python tools/make_smoke_samples.py --force         # 覆盖已有
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import SAMPLES_DIR

SMOKE_DIR = SAMPLES_DIR          # 平铺（load_samples 只认这里）


def describe_subset(items: list[dict]) -> dict:
    """只算"规模"，与 build_eval_samples 的 stats 口径一致（不含答案覆盖那类判断）。"""
    n = len(items)
    if not n:
        return {"n": 0}
    uniq, chars = set(), 0
    for it in items:
        for c in it.get("contexts", []):
            key = (c.get("title", ""), c.get("text", ""))
            if key in uniq:
                continue
            uniq.add(key)
            chars += len(c.get("text", ""))
    ctx = [len(it.get("contexts", [])) for it in items]
    return {"n": n, "avg_ctx_per_q": sum(ctx) / n, "paras_uniq": len(uniq),
            "chars": chars, "tok_est": int(chars / 4),
            "blocks_est": int(chars / 4 / 360) + 1}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="每个数据集取多少题（默认 10）")
    ap.add_argument("--datasets", default="", help="逗号分隔；默认扫 data/samples 下所有 {ds}_*.json")
    ap.add_argument("--force", action="store_true", help="覆盖已有 smoke 文件")
    a = ap.parse_args()

    # 源文件：优先 {ds}_500.json，否则取该数据集**题数最多**的那个（排除已生成的 smoke 小文件）
    wanted = [x.strip() for x in a.datasets.split(",") if x.strip()]
    sources: dict[str, Path] = {}
    for p in sorted(SAMPLES_DIR.glob("*.json")):
        ds = p.stem.rsplit("_", 1)[0]
        if wanted and ds not in wanted:
            continue
        try:
            num = int(p.stem.rsplit("_", 1)[1])
        except ValueError:
            continue                       # 文件名不是 {ds}_{N}.json → 跳过
        if num <= a.n:
            continue                       # 别拿 smoke 自己当源
        if ds not in sources or num > int(sources[ds].stem.rsplit("_", 1)[1]):
            sources[ds] = p

    if not sources:
        print(f"没找到可用的样本文件（{SAMPLES_DIR}/*.json）")
        return 1

    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    made = []
    for ds, src in sorted(sources.items()):
        out = SMOKE_DIR / f"{ds}_{a.n}.json"
        if out.exists() and not a.force:
            print(f"[SKIP] {ds}：{out.name} 已存在（要覆盖加 --force）")
            continue
        d = json.loads(src.read_text(encoding="utf-8"))
        items = (d.get("items") or [])[: a.n]
        if not items:
            print(f"[SKIP] {ds}：源文件没有 items")
            continue
        info = describe_subset(items)
        payload = {"dataset": ds, "sample_size": len(items),
                   "random_seed": d.get("random_seed"),
                   "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "builder": f"make_smoke_samples (prefix of {src.name})",
                   "subset_of": src.name, "subset_note": f"seed 不变的前 {a.n} 题（原抽样前缀）",
                   "source": d.get("source"), "stats": info, "items": items}
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        made.append((ds, out.name, info))
        print(f"[OK] {ds}: {len(items)} 题 -> {out.name}"
              f"（源 {src.name}，去重后 {info['paras_uniq']} 段 ≈{info['tok_est']:,} token "
              f"≈{info['blocks_est']:,} 块）")

    if made:
        lines = ["# smoke 测试集（自动生成）", "",
                 f"来自 `data/samples/{{ds}}_500.json` 的**前 N 题**（seed 不变 → 原抽样前缀）。",
                 f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}", "",
                 "| 数据集 | 文件 | 题数 | 段/题 | 段落(去重) | ≈token | ≈块数 |",
                 "|---|---|---:|---:|---:|---:|---:|"]
        for ds, name, i in made:
            lines.append(f"| {ds} | `{name}` | {i['n']} | {i['avg_ctx_per_q']:.1f} | "
                         f"{i['paras_uniq']:,} | {i['tok_est']:,} | {i['blocks_est']:,} |")
        lines += ["",
                  "## 跑法", "",
                  "```bash",
                  "# 建图 + 检索（run.py 的 --sample 10 会自动找到 {ds}_10.json）",
                  "python run.py --method factalign --dataset hotpotqa --sample 10 --workers 4",
                  "# 换方法就换 --method；外部 baseline 同理",
                  "python run.py --method hipporag --dataset hotpotqa --sample 10 --workers 4",
                  "```", "",
                  "## 注意", "",
                  "- 只用于**验证管线通不通**（建图能跑、检索能出数），**指标无意义**（10 题）。",
                  "- 语料很小：检索天花板极低（答案常不在语料里），全错是正常的。",
                  "- 与正式实验**别共用 v 号**：建议 `--graph-version 99`，免得 smoke 的小图",
                  "  和 500 题的大图互相覆盖（`data/corpus/{ds}/v99` vs `v1`）。",
                  "- 想看真实指标：`--sample 500`（或 300）+ 正式 v 号。",
                  ""]
        (SMOKE_DIR / "SMOKE_README.md").write_text("\n".join(lines), encoding="utf-8")
        print(f"\n已写: {SMOKE_DIR / 'SMOKE_README.md'}（含跑法与注意事项）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
