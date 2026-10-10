"""一键构建：原始语料 → 样本 → 合并语料/切块 → 建图/建索引 → 评测（+ 复现凭证）。

用户要求「必须有一键构建脚本，参数通过配置文件管理」。本脚本就是唯一入口：

    python tools/build_all.py --config configs/baselines.json \
        --datasets hotpotqa,2wiki,musique --sample 500 --methods standardrag,proprag

链路（每步可单独跳过；已存在且校验通过就跳过，不重做）：
  0. 前置检查        tools/check_env.py（vLLM/tokenizer/依赖；失败只警告，可 --skip-check）
  1. 原始语料        HF 下载 → data/raw/{ds}/ + provenance.json（sha256/行数/revision）
  2. 评测样本        tools/build_eval_samples.py → data/samples/{ds}_{n}.json + .meta.json
  3. 合并语料 + 切块  data/merged/{ds}_{n}/{corpus,chunks}.jsonl + checksums.json + build.json
  4. 建图/建索引      python run.py --method M …（不跑题；--methods 里非检索型的会自己跳过）
  5. 复现凭证        tools/write_build_meta.py（run.py 里也会自动写）
  6. 评测            python run.py --method M …（逐题；不是每次都要，用 --no-eval 关）

产物与凭证落点（详见 common/paths.py）：
  data/raw/{ds}/provenance.json          输入版本（来源/链接/revision/哈希）
  data/samples/{ds}_{n}.meta.json        抽样参数 + 来源 + 内容哈希
  data/merged/{ds}_{n}/checksums.json    语料指纹
  data/{method}/{ds}/v{n}/build/         config/env/stats/checksums 四件套
  results/{method}/{ds}/v{n}r{m}.build.json  结果侧登记
  traces/{method}/{ds}/v{n}r{m}/         逐题 prompt/原始输出/每轮检索

用法示例:
  # 只做语料与切块（不建图、不评测）
  python tools/build_all.py --config configs/default.json --datasets hotpotqa --sample 500 \
      --steps corpus
  # 全链路
  python tools/build_all.py --config configs/baselines.json --methods proprag,ircot \
      --datasets hotpotqa --sample 500
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# --config 必须在 import 项目模块之前生效（config.py / adapter 顶层读 env）
def _prescan_config(argv=None):
    a = list(sys.argv[1:] if argv is None else argv)
    for i, tok in enumerate(a):
        if tok == "--config" and i + 1 < len(a):
            return a[i + 1]
        if tok.startswith("--config="):
            return tok.split("=", 1)[1]
    return None


CFG_PATH = _prescan_config()
CFG_APPLIED = None
if CFG_PATH:
    from common.config_file import apply_config_file
    CFG_APPLIED = apply_config_file(CFG_PATH, verbose=False)

from common.paths import (merged_paths, provenance_stamp, raw_dir, read_json,
                          rel_to_root, sha256_file, source_path, write_json)
from config import SAMPLES_DIR, DATA_DIR

STEPS = ("check", "raw", "samples", "corpus", "graph", "meta", "eval")
DEFAULT_DATASETS = ("hotpotqa", "2wiki", "musique")
# 语料级建图/索引的方法（其余是 per-question，无需建图）
GRAPH_METHODS = {"standardrag", "factalign", "lightrag", "hipporag",
                 "hypergraphrag", "ircot", "s2grag", "selfask",
                 "proprag"}


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd: list[str], allow_fail: bool = False, env_extra: dict | None = None) -> int:
    """跑一个子命令（实时透传到终端，不吞输出）。返回 returncode。"""
    env = dict(os.environ)
    if CFG_PATH:
        env["BUILD_ALL_CONFIG"] = str(CFG_PATH)
    env.update(env_extra or {})
    log("$ " + " ".join(str(c) for c in cmd))
    rc = subprocess.call([str(c) for c in cmd], cwd=str(ROOT), env=env)
    if rc != 0 and not allow_fail:
        raise SystemExit(f"命令失败（rc={rc}）：{' '.join(str(c) for c in cmd)}")
    return rc


def py(*args) -> list[str]:
    return [sys.executable, *args]


# ══════════════════════════════════════════════════════════════════════════════
# 步骤实现
# ══════════════════════════════════════════════════════════════════════════════
def step_check(args):
    if args.skip_check:
        log("跳过环境自检（--skip-check）")
        return
    p = ROOT / "tools" / "check_env.py"
    if p.exists():
        run(py(str(p)), allow_fail=True)
    else:
        log("没找到 tools/check_env.py，跳过环境自检")


def step_raw(args, dataset: str):
    """下载原始语料 → data/raw/{ds}/ + provenance.json（requirement 2）。"""
    from tools.record_source import record_dataset_source, fetch_raw
    d = raw_dir(dataset)
    if args.force or not (d / "provenance.json").exists():
        info = fetch_raw(dataset, d, sample=None, force=args.force)
        record_dataset_source(dataset, d, info)
        log(f"原始语料 {dataset} → {rel_to_root(d)}（{info.get('n_rows')} 行）")
    else:
        log(f"原始语料已登记：{rel_to_root(d / 'provenance.json')}（--force 可重下）")


def step_samples(args, dataset: str):
    out = SAMPLES_DIR / f"{dataset}_{args.sample}.json"
    if out.exists() and not args.force:
        log(f"样本已存在：{rel_to_root(out)}（--force 可重建）")
    else:
        run(py(str(ROOT / "tools" / "build_eval_samples.py"),
               "--datasets", dataset, "--sample", str(args.sample),
               "--seed", str(args.seed))
            + (["--force"] if args.force else []))
    # 抽样元信息（来源 revision + 内容哈希）：requirement 2/3
    # 无论样本是新建还是复用都要写（复用旧样本时元信息可能还没生成过）
    write_sample_meta(dataset, args.sample)


def write_sample_meta(dataset: str, sample: int):
    p = SAMPLES_DIR / f"{dataset}_{sample}.json"
    if not p.exists():
        return
    src = read_json(source_path(dataset), default={}) or {}
    raw = read_json(raw_dir(dataset) / "provenance.json", default={}) or {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        items = data.get("items", data if isinstance(data, list) else [])
        n_items = len(items)
        meta_head = {k: v for k, v in data.items() if k != "items"} \
            if isinstance(data, dict) else {}
    except Exception as e:                                   # noqa: BLE001
        n_items, meta_head = None, {"error": f"{type(e).__name__}: {e}"}
    write_json(SAMPLES_DIR / f"{dataset}_{sample}.meta.json", {
        "dataset": dataset, "sample": sample, "n_items": n_items,
        "sample_file": rel_to_root(p), "sha256": sha256_file(p),
        "bytes": p.stat().st_size,
        "dataset_source": src or None,            # sources/{ds}.json 原文
        "raw_provenance": raw or None,            # data/raw/{ds}/provenance.json
        "sample_header": meta_head,               # seed / built_at / builder
        **provenance_stamp(),
    })
    log(f"抽样元信息 → {rel_to_root(SAMPLES_DIR / f'{dataset}_{sample}.meta.json')}")


def step_corpus(args, dataset: str):
    """合并语料 + 不切块（整段）+ 校验（本地、无 LLM；建图前就能查出语料问题）。"""
    paths = merged_paths(dataset, args.sample)
    if paths["checksums"].exists() and not args.force:
        log(f"合并语料已存在：{rel_to_root(paths['corpus'])}（--force 可重建）")
        return
    run(py(str(ROOT / "tools" / "build_merged_corpus.py"),
           "--dataset", dataset, "--sample", str(args.sample))
        + (["--force"] if args.force else []))


def step_graph(args, dataset: str, method: str):
    if method not in GRAPH_METHODS:
        log(f"{method} 不需要建图/索引（per-question 方法），跳过")
        return
    # run.py 自己会判断"图已存在 → 加载不重建"，这里不重复判断
    run(py(str(ROOT / "run.py"), "--method", method, "--dataset", dataset,
           "--sample", str(args.sample), "--workers", str(args.build_workers),
           "--graph-version", str(args.graph_version), "--retrieval", str(args.retrieval),
           "--no-trace")
        + (["--config", str(CFG_PATH)] if CFG_PATH else []))


def step_eval(args, dataset: str, method: str):
    run(py(str(ROOT / "run.py"), "--method", method, "--dataset", dataset,
           "--sample", str(args.sample), "--workers", str(args.workers),
           "--graph-version", str(args.graph_version), "--retrieval", str(args.retrieval),
           "--trace")
        + (["--config", str(CFG_PATH)] if CFG_PATH else []))


def main():
    ap = argparse.ArgumentParser(description="一键构建：原始语料 → 图谱/索引 → 评测")
    ap.add_argument("--config", default=CFG_PATH, help="configs/*.json（参数管理入口）")
    ap.add_argument("--datasets", default=",".join(DEFAULT_DATASETS),
                    help=f"逗号分隔；默认 {','.join(DEFAULT_DATASETS)}")
    ap.add_argument("--methods", default="standardrag",
                    help="要建图/评测的方法（逗号分隔，见 run.py --method）")
    ap.add_argument("--sample", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8, help="评测并发")
    ap.add_argument("--build-workers", type=int, default=8, help="建图并发")
    ap.add_argument("--graph-version", type=int, default=1)
    ap.add_argument("--retrieval", type=int, default=1)
    ap.add_argument("--steps", default="check,raw,samples,corpus,graph,meta",
                    help=f"逗号分隔的子集（{','.join(STEPS)}）；默认不含 eval")
    ap.add_argument("--force", action="store_true", help="已存在的产物也重建")
    ap.add_argument("--skip-check", action="store_true")
    args = ap.parse_args()

    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    bad = [s for s in steps if s not in STEPS]
    if bad:
        raise SystemExit(f"未知步骤 {bad}；可选 {list(STEPS)}")
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    log(f"config={CFG_PATH or '(未给，全用代码默认值/环境变量)'} "
        f"datasets={datasets} methods={methods} sample={args.sample} steps={steps}")
    t0 = time.time()
    if "check" in steps:
        step_check(args)
    for ds in datasets:
        if "raw" in steps:
            step_raw(args, ds)
        if "samples" in steps:
            step_samples(args, ds)
        if "corpus" in steps:
            step_corpus(args, ds)
        for m in methods:
            if "graph" in steps:
                step_graph(args, ds, m)
            if "meta" in steps:
                for d in (DATA_DIR / m / ds / f"v{args.graph_version}",) :
                    if d.exists():
                        run(py(str(ROOT / "tools" / "write_build_meta.py"),
                               "--dir", str(d), "--method", m, "--dataset", ds,
                               "--sample", str(args.sample),
                               "--graph-version", str(args.graph_version),
                               "--retrieval", str(args.retrieval))
                            + (["--config", str(CFG_PATH)] if CFG_PATH else []),
                            allow_fail=True)
            if "eval" in steps:
                step_eval(args, ds, m)
    log(f"全部完成，用时 {time.time() - t0:.1f}s")
    log("复现凭证：data/{method}/{dataset}/v{n}/build/{config,env,stats,checksums}.json；"
        "results/…/{v}r{m}.build.json 有登记")


if __name__ == "__main__":
    main()
