r"""从 results/ 生成论文用的图与表（SIGMOD ARI 要求「生成论文每张图的脚本」）。

输入（全部来自 `results/`，不需要重跑实验）:
  results/summary.jsonl                        每次 run 一行（含 cost 块：prefill/decode/embed
                                               token、client 挂钟、gpu_busy_seconds、峰值显存）
  results/{method}/{dataset}/v{n}r{m}.jsonl    逐题结果（em/f1 + 每题成本）
  results/summary.jsonl 里的 build_* / query_* 桶                      ← 建图 vs 检索拆分

输出（默认 results/figures/）:
  fig1_quality.{png,pdf}        EM/F1 分组柱状图（x=方法，按数据集分组）+ 95% CI 误差棒
  fig2_cost_quality.{png,pdf}   成本-效果帕累托图（x=每题 GPU busy 秒[对数]，y=EM，点=方法）
  fig3_token_split.{png,pdf}    堆叠柱状：prefill / decode / embed 三类 token 分开
  fig4_time_breakdown.{png,pdf} 时间构成：客户端挂钟 vs GPU busy，建图/检索分段
  table1_main.tex               主结果表（EM/F1/成本，可直接 \input 进论文）

用法:
  python tools/make_figures.py                                  # 全部
  python tools/make_figures.py --only fig1,table1
  python tools/make_figures.py --methods factalign,standardrag,ircot --datasets hotpotqa
  python tools/make_figures.py --metric em --format pdf
  python tools/make_figures.py --summary results/summary.fixture.jsonl --out results/figures_demo
  python tools/make_figures.py --make-fixture                    # 造假数据（先看效果/自检）

约定：
  · 同一个 (method, dataset, v, r) 有多次 run → 取最后一次；`complete=false` 的默认丢弃。
  · 每个数据集的柱子都要带误差棒：用的是**逐题 EM/F1 的 95% 置信区间**
    （正态近似，n 大时够用；n<30 时改用 bootstrap，见 --ci boot）。
  · 论文里报的"时间"默认 = **gpu_busy_seconds**（真算力占用），客户端挂钟另画一张
    （它含 vLLM 排队，随并发变化，不能当 GPU 时间用 —— 见 summary.cost.measurement）。
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 方法显示名（论文里的写法）；不在表里的方法用原名
DISPLAY = {
    "standardrag": "Standard RAG",
    "zeroshot": "No Retrieval",
    "factalign": "FactAlign-RAG",
    "ircot": "IRCoT",
    "selfask": "Self-Ask",
    "proprag": "PropRAG",
    "s2grag": "S2G-RAG",
    "hipporag": "HippoRAG 2",
    "hypergraphrag": "HyperGraphRAG",
    "lightrag": "LightRAG",
}
# 画图时的固定顺序（论文表格顺序）：自带地基 → 本方法 → 图谱类 → 迭代类
ORDER = ["zeroshot", "standardrag", "factalign", "hypergraphrag",
         "lightrag", "hipporag", "proprag", "ircot", "selfask", "s2grag"]
# 每数据集一套配色（色盲友好）
DATASET_COLORS = {"hotpotqa": "#4C72B0", "2wiki": "#DD8452", "musique": "#55A868",
                  "popqa": "#C44E52", "triviaqa": "#8172B3"}


# 图里的文字语言：论文用 en（默认），本地看用 zh
LABELS = {
    "en": {"tok_title": "{ds} · token breakdown (log scale)",
           "tok_ylabel": "tokens / question",
           "time_title": "{ds} · GPU-busy vs client wall-clock (latter includes queueing)",
           "pareto": "Pareto front", "tbl_caption":
           r"Main results. EM/F1 over \texttt{%s}; cost is per question "
           r"(prefill/decode/embed tokens)."},
    "zh": {"tok_title": "{ds} · token 构成（对数轴）",
           "tok_ylabel": "tokens / 题",
           "time_title": "{ds} · GPU busy vs 客户端挂钟（后者含排队）",
           "pareto": "帕累托前沿", "tbl_caption":
           r"主结果。EM/F1（\texttt{%s}）；成本按每题统计（prefill/decode/embed token）。"},
}
LANG = "en"


def L(key: str) -> str:
    return LABELS.get(LANG, LABELS["en"]).get(key, key)


def dname(m: str) -> str:
    return DISPLAY.get(m, m)


# ══════════════════════════════════════════════════════════════════════════════
# 读数据
# ══════════════════════════════════════════════════════════════════════════════
def load_summary(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"没有 {path}：先至少跑完一次 run.py（会写这一行汇总）")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except Exception:                       # noqa: BLE001
                pass
    return rows


def latest_runs(rows: list[dict], methods=None, datasets=None, only_complete=True) -> dict:
    """(method, dataset) → 最后一次完整 run 的汇总行。"""
    best = {}
    for r in rows:
        if only_complete and not r.get("complete", True):
            continue
        if methods and r.get("method") not in methods:
            continue
        if datasets and r.get("dataset") not in datasets:
            continue
        key = (r.get("method"), r.get("dataset"))
        # ts 字符串可直接比较（YYYY-MM-DD HH:MM:SS）
        if key not in best or str(r.get("ts", "")) >= str(best[key].get("ts", "")):
            best[key] = r
    return best


def load_per_question(method: str, dataset: str, v: int, r: int) -> list[dict]:
    from common.result import result_path
    p = result_path(method, dataset, v, r)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except Exception:                       # noqa: BLE001
                pass
    return out


def ci95(vals: list[float], mode: str = "normal", n_boot: int = 2000, seed: int = 0):
    """逐题指标 → (均值, 半宽)。normal = 正态近似；boot = bootstrap 百分位。"""
    a = np.asarray([v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))],
                   dtype=float)
    if a.size == 0:
        return float("nan"), float("nan")
    mean = float(a.mean())
    if a.size == 1:
        return mean, 0.0
    if mode == "boot":
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, a.size, size=(n_boot, a.size))
        means = a[idx].mean(axis=1)
        lo, hi = np.percentile(means, [2.5, 97.5])
        return mean, float((hi - lo) / 2)
    se = float(a.std(ddof=1) / math.sqrt(a.size))
    return mean, 1.96 * se


# ══════════════════════════════════════════════════════════════════════════════
# 量纲/取值辅助
# ══════════════════════════════════════════════════════════════════════════════
def cost_of(row: dict) -> dict:
    """从汇总行里取成本（优先新的 cost 块；老的只有扁平键时也能画）。"""
    c = dict(row.get("cost") or {})
    s = row
    get = lambda k, alt=None: (c.get(k) if c.get(k) is not None
                               else (s.get(alt or k) if s.get(alt or k) is not None else 0))
    n = max(1, int(s.get("n") or 1))
    out = {
        "prefill": get("prefill_tokens"),
        "decode": get("decode_tokens"),
        "embed": get("embed_tokens"),
        "prefill_build": get("prefill_tokens_build", "build_prompt_tokens"),
        "decode_build": get("decode_tokens_build", "build_completion_tokens"),
        "embed_build": get("embed_tokens_build", "build_embed_tokens"),
        "prefill_query": get("prefill_tokens_query"),
        "decode_query": get("decode_tokens_query"),
        "embed_query": get("embed_tokens_query", "query_embed_tokens"),
        "wall_s": get("wall_seconds_total", "wall_seconds"),
        "client_chat_s": get("client_chat_seconds"),
        "client_embed_s": get("client_embed_seconds"),
        "gpu_s": (c.get("gpu_busy_seconds") if c.get("gpu_busy_seconds") is not None
                  else row.get("gpu_busy_seconds")),
        "gpu_s_build": (c.get("gpu_busy_seconds_build") if c.get("gpu_busy_seconds_build") is not None
                        else row.get("gpu_busy_seconds_build")),
        "gpu_s_query": (c.get("gpu_busy_seconds_query") if c.get("gpu_busy_seconds_query") is not None
                        else row.get("gpu_busy_seconds_query")),
        "peak_mem_mb": (c.get("gpu_peak_mem_mb") if c.get("gpu_peak_mem_mb") is not None
                        else row.get("gpu_peak_mem_mb")),
        "n": n,
    }
    # 每题平均
    for k in ("prefill", "decode", "embed", "wall_s", "client_chat_s", "client_embed_s", "gpu_s"):
        out[k + "_pq"] = (out[k] / n) if out[k] else 0.0
    return out


def has_gpu(runs) -> bool:
    """runs 是 {(method, dataset): 汇总行} 的 dict → 取 values() 判断。"""
    return any(cost_of(r)["gpu_s"] for r in runs.values())


# ══════════════════════════════════════════════════════════════════════════════
# 图 1：质量（EM/F1）分组柱状 + 误差棒
# ══════════════════════════════════════════════════════════════════════════════
def fig_quality(runs: dict, per_q: dict, out_dir: Path, metric="em", ci="normal",
                fmt="png", dpi=200, datasets=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = datasets or sorted({d for _, d in runs})
    methods = [m for m in ORDER if any(k[0] == m for k in runs)]
    methods += sorted({m for m, _ in runs if m not in methods})          # 表外方法附在后面

    fig, ax = plt.subplots(figsize=(0.78 * len(methods) + 2.0, 2.9))
    w = 0.8 / max(1, len(datasets))
    for di, ds in enumerate(datasets):
        xs, ys, es = [], [], []
        for mi, m in enumerate(methods):
            r = runs.get((m, ds))
            if not r:
                continue
            vals = [q.get(metric) for q in per_q.get((m, ds), [])]
            if vals and any(v is not None for v in vals):
                mu, half = ci95(vals, mode=ci)
            else:                                   # 逐题文件不在 → 退回汇总均值（无误差棒）
                mu, half = float(r.get(metric, 0.0)) * (100 if r.get(metric, 0) <= 1 else 1), 0.0
            xs.append(mi + (di - (len(datasets) - 1) / 2) * w)
            ys.append(mu if mu <= 1 else mu / 100)
            es.append(half)
        ax.bar(xs, ys, width=w, yerr=es, capsize=2.0, label=ds,
               color=DATASET_COLORS.get(ds, None), edgecolor="black", linewidth=0.4,
               error_kw={"elinewidth": 0.8, "ecolor": "#333333"})
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([dname(m) for m in methods], rotation=32, ha="right", fontsize=8)
    ax.set_ylabel(f"{metric.upper()} (95% CI)")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", linestyle=":", alpha=0.45, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(ncol=min(3, len(datasets)), fontsize=8, frameon=False)
    fig.tight_layout()
    paths = save(fig, out_dir, f"fig1_quality_{metric}", fmt, dpi)
    plt.close(fig)
    return paths


# ══════════════════════════════════════════════════════════════════════════════
# 图 2：成本-效果帕累托（x=每题 GPU 秒 对数轴，y=EM）
# ══════════════════════════════════════════════════════════════════════════════
def fig_cost_quality(runs: dict, out_dir: Path, metric="em", fmt="png", dpi=200,
                     datasets=None, xkey="gpu_s_pq"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = datasets or sorted({d for _, d in runs})
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.2 * len(datasets), 3.2),
                             squeeze=False, sharey=True)
    for ax, ds in zip(axes[0], datasets):
        pts = []
        for (m, d), r in runs.items():
            if d != ds:
                continue
            c = cost_of(r)
            x = c.get(xkey) or 0.0
            if x <= 0:
                x = c.get("prefill_pq", 0) + c.get("decode_pq", 0) + c.get("embed_pq", 0)
                xlab = "tokens / question"
            else:
                xlab = "GPU-busy seconds / question"
            y = float(r.get(metric, 0.0))
            y = y if y <= 1 else y / 100
            pts.append((x, y, m, r.get("em")))
        for x, y, m, _em in pts:
            if x <= 0:
                continue
            ax.scatter(x, y, s=46, zorder=3, edgecolor="black", linewidth=0.5,
                       color=DATASET_COLORS.get(ds, "#4C72B0"))
        # 标签防重叠：在**数据坐标**里贪心下移并夹到 [0.03, 0.97]，越界的用引线连回点。
        # （早先用像素 offset 排，遇到密集簇会甩到坐标轴外面、盖住 x 轴标签。）
        span, ymin, ymax = 0.048, 0.03, 0.97
        placed = []                       # (label_y, x) 已放置的标签位置
        for x, y, m, _em in sorted([p for p in pts if p[0] > 0], key=lambda p: -p[1]):
            ly = y
            while any(abs(ly - py) < span and abs(math.log10(x) - math.log10(px)) < 0.20
                      for py, px in placed):
                ly -= span
            ly = max(ymin, min(ymax, ly))
            placed.append((ly, x))
            dy = ly - y
            # 下半部分往左标（右侧往往已堆满），上半往右标
            ha = "right" if y < 0.55 else "left"
            ax.annotate(dname(m), (x, y), xytext=(x, ly), textcoords="data",
                        fontsize=6.5, va="center", ha=ha, annotation_clip=False,
                        xycoords="data",
                        arrowprops=({"arrowstyle": "-", "color": "#bbbbbb", "lw": 0.5,
                                     "shrinkA": 1, "shrinkB": 3}
                                    if abs(dy) > 0.02 else None))
        # 帕累托前沿（越左越上越好）
        good = sorted([p for p in pts if p[0] > 0], key=lambda p: (p[0], -p[1]))
        front, best = [], -1.0
        for p in good:
            if p[1] > best:
                front.append(p)
                best = p[1]
        if len(front) > 1:
            ax.plot([p[0] for p in front], [p[1] for p in front], "--",
                    color="#888888", linewidth=1.0, zorder=2, label=L("pareto"))
        ax.set_xscale("log")
        ax.set_xlabel(xlab, fontsize=8)
        ax.set_title(ds, fontsize=9)
        ax.grid(linestyle=":", alpha=0.45, linewidth=0.6)
    axes[0][0].set_ylabel(f"{metric.upper()}")
    axes[0][0].set_ylim(0, 1.0)
    fig.tight_layout()
    paths = save(fig, out_dir, f"fig2_cost_quality_{xkey}", fmt, dpi)
    plt.close(fig)
    return paths


# ══════════════════════════════════════════════════════════════════════════════
# 图 3：三类 token 堆叠（prefill / decode / embed），建图与检索分段
# ══════════════════════════════════════════════════════════════════════════════
def fig_token_split(runs: dict, out_dir: Path, fmt="png", dpi=200, dataset=None,
                    per_question=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dataset = dataset or sorted({d for _, d in runs})[0]
    items = [(m, cost_of(r)) for (m, d), r in runs.items() if d == dataset]
    items.sort(key=lambda t: t[0] in ORDER and ORDER.index(t[0]) or 99)
    methods = [m for m, _ in items]
    if not methods:
        return []
    div = 1 if not per_question else max(1, items[0][1]["n"])

    fig, ax = plt.subplots(figsize=(0.78 * len(methods) + 2.0, 2.9))
    x = np.arange(len(methods))
    prefill = np.array([c["prefill"] / div for _, c in items])
    decode = np.array([c["decode"] / div for _, c in items])
    embed = np.array([c["embed"] / div for _, c in items])
    ax.bar(x, prefill, label="prefill (prompt)", color="#4C72B0", edgecolor="black", linewidth=0.4)
    ax.bar(x, decode, bottom=prefill, label="decode (generated)", color="#DD8452",
           edgecolor="black", linewidth=0.4)
    ax.bar(x, embed, bottom=prefill + decode, label="embedding", color="#55A868",
           edgecolor="black", linewidth=0.4)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([dname(m) for m in methods], rotation=32, ha="right", fontsize=8)
    ax.set_ylabel(L("tok_ylabel") if per_question else "tokens / run")
    ax.set_title(L("tok_title").format(ds=dataset), fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.45, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    paths = save(fig, out_dir, f"fig3_token_split_{dataset}", fmt, dpi)
    plt.close(fig)
    return paths


# ══════════════════════════════════════════════════════════════════════════════
# 图 4：时间构成（客户端挂钟 vs GPU busy；建图/检索分段）
# ══════════════════════════════════════════════════════════════════════════════
def fig_time_breakdown(runs: dict, out_dir: Path, fmt="png", dpi=200, dataset=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dataset = dataset or sorted({d for _, d in runs})[0]
    items = [(m, cost_of(r), r) for (m, d), r in runs.items() if d == dataset]
    items.sort(key=lambda t: t[0] in ORDER and ORDER.index(t[0]) or 99)
    if not items:
        return []
    methods = [m for m, _, _ in items]
    x = np.arange(len(methods))
    w = 0.38
    gpu_b = np.array([c["gpu_s_build"] or 0.0 for _, c, _ in items])
    gpu_q = np.array([c["gpu_s_query"] or 0.0 for _, c, _ in items])
    cli_b = np.array([(r.get("build_seconds") or 0.0) for _, _, r in items])
    cli_q = np.array([(r.get("query_seconds") or 0.0) for _, _, r in items])

    fig, ax = plt.subplots(figsize=(0.78 * len(methods) + 2.0, 2.9))
    ax.bar(x - w / 2, gpu_b, width=w, label="GPU busy · build", color="#4C72B0",
           edgecolor="black", linewidth=0.4)
    ax.bar(x - w / 2, gpu_q, width=w, bottom=gpu_b, label="GPU busy · query",
           color="#9ecae1", edgecolor="black", linewidth=0.4)
    ax.bar(x + w / 2, cli_b, width=w, label="client wall · build", color="#DD8452",
           edgecolor="black", linewidth=0.4)
    ax.bar(x + w / 2, cli_q, width=w, bottom=cli_b, label="client wall · query",
           color="#f5c396", edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([dname(m) for m in methods], rotation=32, ha="right", fontsize=8)
    ax.set_ylabel("seconds (whole run)")
    ax.set_title(L("time_title").format(ds=dataset), fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.45, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7.5, frameon=False, ncol=2, loc="upper left")
    fig.tight_layout()
    paths = save(fig, out_dir, f"fig4_time_breakdown_{dataset}", fmt, dpi)
    plt.close(fig)
    return paths


# ══════════════════════════════════════════════════════════════════════════════
# 表 1：主结果表（LaTeX）
# ══════════════════════════════════════════════════════════════════════════════
def table_main(runs: dict, out_dir: Path, datasets=None):
    datasets = datasets or sorted({d for _, d in runs})
    methods = [m for m in ORDER if any(k[0] == m for k in runs)]
    methods += sorted({m for m, _ in runs if m not in methods})
    # 成本列跟着数据集走（每个数据集一份），**不是跨数据集平均** —— 平均会把
    # musique 的高成本摊到 hotpotqa 上，审稿人一眼就能看出问题。
    cost_head = ["prefill", "decode", "embed", "GPU s"]
    ncost = len(cost_head)
    any_gpu = has_gpu(runs)
    if not any_gpu:
        cost_head = cost_head[:3]
        ncost = 3
    lines = [
        r"% Auto-generated by tools/make_figures.py -- do not edit by hand",
        r"\begin{table*}[t]\centering\small",
        r"\caption{" + L("tbl_caption") % ", ".join(datasets) + "}",
        r"\label{tab:main}",
        r"\begin{tabular}{l" + "rr" * len(datasets) + "r" * (ncost * len(datasets)) + r"}",
        r"\toprule",
        r" & " + " & ".join(
            rf"\multicolumn{{2}}{{c}}{{{d}}}" for d in datasets)
        + " & " + " & ".join(
            rf"\multicolumn{{{ncost}}}{{c}}{{{d} · cost/q}}" for d in datasets) + r" \\",
        r"\cmidrule(lr){2-%d}" % (1 + 2 * len(datasets))
        + "".join(
            r" \cmidrule(lr){%d-%d}" % (2 + 2 * len(datasets) + (ncost * i),
                                        1 + 2 * len(datasets) + ncost * (i + 1))
            for i in range(len(datasets))) + r" \\",
        r"Method & " + " & ".join(["EM & F1"] * len(datasets)) + " & "
        + " & ".join(" & ".join(cost_head) for _ in datasets) + r" \\",
        r"\midrule",
    ]
    for m in methods:
        cells = []
        # 先质量列（每个数据集 EM/F1）
        for ds in datasets:
            r = runs.get((m, ds))
            if not r:
                cells += ["--", "--"]
                continue
            em, f1 = r.get("em"), r.get("f1")
            cells += [f"{em * 100:.1f}" if em is not None else "--",
                      f"{f1 * 100:.1f}" if f1 is not None else "--"]
        # 再成本列（与上面**同一数据集顺序**，不是一个方法一个均值）
        for ds in datasets:
            r = runs.get((m, ds))
            if not r:
                cells += ["--"] * ncost
                continue
            c = cost_of(r)
            cells += [f"{c['prefill_pq']:,.0f}", f"{c['decode_pq']:,.0f}"]
            if any_gpu:
                cells += [f"{c['embed_pq']:,.0f}",
                          f"{c['gpu_s_pq']:,.2f}" if c["gpu_s_pq"] else "--"]
            else:
                cells += [f"{c['embed_pq']:,.0f}"]
        name = dname(m).replace("_", r"\_") + (r" \textbf{(ours)}" if m == "factalign" else "")
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "table1_main.tex"
    p.write_text("\n".join(lines), encoding="utf-8")
    return [p]


# ══════════════════════════════════════════════════════════════════════════════
def save(fig, out_dir: Path, stem: str, fmt: str, dpi: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    fmts = [f.strip() for f in fmt.split(",") if f.strip()]
    paths = []
    for f in fmts:
        p = out_dir / f"{stem}.{f}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        paths.append(p)
    return paths


# ══════════════════════════════════════════════════════════════════════════════
# 假数据（先看效果 / 自检脚本本身）
# ══════════════════════════════════════════════════════════════════════════════
def make_fixture(path: Path, seed: int = 42, n: int = 300):
    """造一份结构正确的 summary.jsonl + 逐题 jsonl，用来验证画图脚本（不是真结果！）。"""
    rng = np.random.default_rng(seed)
    from common.result import result_path
    methods = {"zeroshot": 0.28, "standardrag": 0.50, "factalign": 0.62,
               "hypergraphrag": 0.53, "lightrag": 0.51, "hipporag": 0.58,
               "proprag": 0.55, "ircot": 0.57, "selfask": 0.49, "s2grag": 0.60}
    # (prefill tok/题, decode tok/题, embed tok/题)；数量级按实测口径：
    # standardrag ≈ 8.5k 字符 prompt ≈ 1.4k token；一块正文 ≈ 2.8k 字符（**默认不切块**：
    # 一个 passage 一块，旧 400/40 口径下才约 400 token）→ 按 char/token≈6.1 折算 ≈ 460 token/块
    per_q_cost = {"zeroshot": (25, 8, 0),
                  "standardrag": (520, 8, 45), "factalign": (760, 55, 180),
                  "hypergraphrag": (640, 60, 260), "lightrag": (700, 60, 300),
                  "hipporag": (720, 65, 320),
                  "proprag": (1250, 180, 380),
                  "ircot": (4200, 22, 4), "selfask": (900, 65, 6),
                  "s2grag": (5900, 60, 45)}
    rows = []
    for ds, scale in (("hotpotqa", 1.0), ("2wiki", 0.95), ("musique", 0.72)):
        for m, base in methods.items():
            acc = base * scale
            pf, dc, eb = per_q_cost[m]
            tok = {"prefill_tokens": pf * n, "decode_tokens": dc * n, "embed_tokens": eb * n,
                   "prefill_tokens_build": pf * n * (0.35 if eb else 0.2),
                   "decode_tokens_build": dc * n * 0.3, "embed_tokens_build": eb * n * 0.8,
                   "prefill_tokens_query": pf * n * (0.65 if eb else 0.8),
                   "decode_tokens_query": dc * n * 0.7, "embed_tokens_query": eb * n * 0.2}
            gpu_pq = (pf / 9500 + dc / 260 + eb / 6000)     # 粗算：prefill/decode/embed 吞吐不同
            gpu_s = gpu_pq * n
            row = {"ts": "2026-09-22 12:00:00", "method": m, "dataset": ds, "sample": n,
                   "graph_version": 1, "retrieval": 1, "n": n, "complete": True,
                   "em": round(acc + rng.normal(0, 0.01), 4),
                   "f1": round(acc + 0.08 + rng.normal(0, 0.01), 4),
                   "prompt_tokens": pf * n, "completion_tokens": dc * n,
                   "embed_tokens": eb * n, "llm_seconds": round(gpu_s * 0.6, 1),
                   "build_seconds": round(gpu_s * 0.55, 1), "build_prompt_tokens": pf * n * 0.35,
                   "build_completion_tokens": dc * n * 0.3, "build_embed_tokens": eb * n * 0.8,
                   "query_seconds": round(gpu_s * 0.45, 1), "query_embed_tokens": eb * n * 0.2,
                   "chat_calls": int(n * (2 if m in ("proprag", "s2grag") else 1)),
                   "wall_seconds": round(gpu_s * 1.8, 1), "gpu_busy_seconds": round(gpu_s, 1),
                   "gpu_busy_seconds_build": round(gpu_s * 0.55, 1),
                   "gpu_busy_seconds_query": round(gpu_s * 0.45, 1),
                   "gpu_peak_mem_mb": 18000 + (6000 if eb > 1000 else 0)}
            row["cost"] = {**tok, "wall_seconds_total": row["wall_seconds"],
                           "client_chat_seconds": round(gpu_s * 1.1, 1),
                           "client_embed_seconds": round(gpu_s * 0.5, 1),
                           "gpu_busy_seconds": row["gpu_busy_seconds"],
                           "gpu_busy_seconds_build": row["gpu_busy_seconds_build"],
                           "gpu_busy_seconds_query": row["gpu_busy_seconds_query"],
                           "gpu_peak_mem_mb": row["gpu_peak_mem_mb"],
                           "chat_calls": row["chat_calls"], "embed_calls": int(n * 1.2),
                           "per_question": {k: round(tok["prefill_tokens"] / n, 1)
                                            for k in ()} or {},
                           "measurement": {"gpu_time_method": "fixture"}}
            rows.append(row)
            # 逐题 jsonl（画误差棒用）
            p = result_path(m, ds, 1, 1)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("w", encoding="utf-8") as f:
                for i in range(n):
                    em = bool(rng.random() < acc)
                    f.write(json.dumps({"id": f"{ds}_{i:04d}", "question": "q", "gold": "a",
                                        "pred": "x", "em": em,
                                        "f1": round(1.0 if em else rng.random() * 0.5, 3),
                                        "prompt_tokens": pf, "completion_tokens": dc,
                                        "embed_tokens": eb}, ensure_ascii=False) + "\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                    encoding="utf-8")
    return path


# ══════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="从 results/ 生成论文图与表")
    ap.add_argument("--summary", default=str(ROOT / "results" / "summary.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "results" / "figures"))
    ap.add_argument("--methods", default=None, help="逗号分隔；默认全部")
    ap.add_argument("--datasets", default=None, help="逗号分隔；默认全部")
    ap.add_argument("--metric", default="em", choices=["em", "f1"])
    ap.add_argument("--only", default="fig1,fig2,fig3,fig4,table1",
                    help="逗号分隔子集")
    ap.add_argument("--format", default="png", help="png,pdf,svg…（ARI 建议给矢量图）")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--ci", default="normal", choices=["normal", "boot"])
    ap.add_argument("--lang", default="en", choices=["en", "zh"],
                    help="图内文字语言：论文用 en（默认）；zh 需要系统有中文字体")
    ap.add_argument("--include-incomplete", action="store_true",
                    help="把 complete=false 的 run 也算进来（默认只画跑完的）")
    ap.add_argument("--make-fixture", action="store_true",
                    help="生成假数据（results/summary.fixture.jsonl + 逐题 jsonl）然后退出")
    ap.add_argument("--fixture-n", type=int, default=300)
    args = ap.parse_args()

    if args.make_fixture:
        p = make_fixture(ROOT / "results" / "summary.fixture.jsonl", n=args.fixture_n)
        print(f"[fixture] 写好了（**假数据，只能用来验证脚本**）：{p}")
        print(f"          画图：python tools/make_figures.py --summary {p.relative_to(ROOT)} "
              f"--out results/figures_demo")
        return

    global LANG
    LANG = args.lang
    if LANG == "zh":
        import matplotlib
        matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                                                  "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False
    methods = [m.strip() for m in args.methods.split(",")] if args.methods else None
    datasets = [d.strip() for d in args.datasets.split(",")] if args.datasets else None
    only = {s.strip() for s in args.only.split(",")}
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir

    rows = load_summary(Path(args.summary))
    runs = latest_runs(rows, methods=methods, datasets=datasets,
                       only_complete=not args.include_incomplete)
    if not runs:
        raise SystemExit(f"{args.summary} 里没有可画的 run（method/dataset 过滤太严？"
                         f"或用 --include-incomplete）")
    print(f"[figures] 汇总 {len(rows)} 次 run → 取到 {len(runs)} 个 (方法, 数据集) 组合："
          f"{sorted({m for m, _ in runs})}")
    if not has_gpu(runs):
        print("[figures] ⚠ 这些 run 没有 GPU 采样数据（gpu_busy_seconds 为空）→ "
              "fig2/fig4 会用 token 或客户端时间兜底；论文里若要报 GPU 时间请在有 "
              "nvidia-smi 的机器上重跑")

    per_q = {}
    for (m, ds), r in runs.items():
        per_q[(m, ds)] = load_per_question(m, ds, int(r.get("graph_version", 1)),
                                           int(r.get("retrieval", 1)))
    made = []
    if "fig1" in only:
        made += fig_quality(runs, per_q, out_dir, metric=args.metric, ci=args.ci,
                            fmt=args.format, dpi=args.dpi, datasets=datasets)
    if "fig2" in only:
        made += fig_cost_quality(runs, out_dir, metric=args.metric, fmt=args.format,
                                 dpi=args.dpi, datasets=datasets)
    if "fig3" in only:
        made += fig_token_split(runs, out_dir, fmt=args.format, dpi=args.dpi,
                                dataset=(datasets[0] if datasets else None))
    if "fig4" in only:
        made += fig_time_breakdown(runs, out_dir, fmt=args.format, dpi=args.dpi,
                                   dataset=(datasets[0] if datasets else None))
    if "table1" in only:
        made += table_main(runs, out_dir, datasets=datasets)
    for p in made:
        print("  →", p.relative_to(ROOT) if str(p).startswith(str(ROOT)) else p)
    print(f"[figures] 共 {len(made)} 个文件 → {out_dir}")


if __name__ == "__main__":
    main()
