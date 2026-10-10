"""登记「原始语料是从哪来的、哪一版」——requirement 2（标注输入版本）。

产出 `data/raw/{dataset}/provenance.json`：

```jsonc
{
  "dataset": "hotpotqa",
  "source": {                         // sources/{dataset}.json 的原文（人工维护）
     "hf_id": "hotpotqa/hotpot_qa", "hf_config": "distractor", "split": "validation",
     "download_url": "https://huggingface.co/datasets/hotpotqa/hotpot_qa",
     "license": "CC BY-SA 4.0", "paper": "arXiv:1809.09600", "version": "…"
  },
  "downloaded": {                     // 本次实际拉到的版本（自动采集）
     "at": "…", "hf_revision": "…", "n_rows": 7405,
     "file": "data/raw/hotpotqa/raw.jsonl", "sha256": "…", "bytes": 123
  },
  "env": { "HF_ENDPOINT": "https://hf-mirror.com", "datasets": "3.2.0", ... },
  "code": { "git_commit": "…", "dirty": false }
}
```

用法：
  python tools/record_source.py --dataset hotpotqa              # 只登记（不下载）
  python tools/record_source.py --dataset hotpotqa --fetch      # 下载 + 登记
  python tools/record_source.py --all --fetch
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from common.paths import (provenance_stamp, raw_dir, read_json, rel_to_root,
                          sha256_file, source_path, write_json)


def load_source(dataset: str) -> dict:
    """读人工维护的来源声明 sources/{dataset}.json。"""
    src = read_json(source_path(dataset))
    if not src:
        print(f"[source] ⚠ 缺 {rel_to_root(source_path(dataset))}："
              f"下载链接/许可/版本无法自动得知，请补该文件（模板见 sources/hotpotqa.json）")
        return {}
    return src


def fetch_raw(dataset: str, out_dir: Path, sample=None, force: bool = False) -> dict:
    """把原始 HF 数据集落到 data/raw/{dataset}/raw.jsonl。

    ⚠️ 这一步只为「可复现的输入版本」留档；评测样本仍由
    `tools/build_eval_samples.py` 从 HF 现抽（两者用同一个 HF revision 时应一致——
    不一致会在 provenance 里看出来，这正是记录它的意义）。
    """
    src = load_source(dataset)
    hf_id = (src.get("hf_id") or "").strip()
    if not hf_id:
        return {"error": "sources 里没有 hf_id，无法下载", "source": src}
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "raw.jsonl"
    if out.exists() and not force:
        return {"file": rel_to_root(out), "bytes": out.stat().st_size,
                "sha256": sha256_file(out), "reused": True, "source": src}
    try:
        from datasets import load_dataset
        import datasets as _ds
        config = src.get("hf_config") or None
        split = src.get("split") or "validation"
        kwargs = {"split": split}
        if config:
            kwargs["name"] = config
        if src.get("hf_revision"):
            kwargs["revision"] = src["hf_revision"]
        ds = (load_dataset(hf_id, **kwargs) if config
              else load_dataset(hf_id, split=split))
        n = 0
        with out.open("w", encoding="utf-8") as f:
            for ex in ds:
                f.write(json.dumps(ex, ensure_ascii=False, default=str) + "\n")
                n += 1
        return {"file": rel_to_root(out), "bytes": out.stat().st_size,
                "sha256": sha256_file(out), "n_rows": n,
                "hf_revision": getattr(getattr(ds, "info", None), "version", None)
                or src.get("hf_revision"),
                "datasets_version": getattr(_ds, "__version__", "?"),
                "source": src}
    except Exception as e:                                   # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}", "source": src}


def record_dataset_source(dataset: str, out_dir: Path, fetch_info: dict | None = None) -> Path:
    """把来源 + 下载信息 + 代码版本写进 data/raw/{dataset}/provenance.json。"""
    src = (fetch_info or {}).get("source") or load_source(dataset)
    rec = {
        "dataset": dataset,
        "source": src or None,
        "downloaded": {k: v for k, v in (fetch_info or {}).items() if k != "source"},
        "env": {"HF_ENDPOINT": os.environ.get("HF_ENDPOINT"),
                "datasets": _pkg_version("datasets")},
        "code": provenance_stamp().get("git"),
        "registered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "本文件是「输入版本」的凭证：来源/revision/下载链接/内容哈希。",
    }
    return write_json(Path(out_dir) / "provenance.json", rec)


def _pkg_version(name: str) -> str:
    try:
        import importlib.metadata as md
        return md.version(name)
    except Exception:                                        # noqa: BLE001
        return "?"


def main():
    ap = argparse.ArgumentParser(description="登记/下载原始语料并写 provenance.json")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--all", action="store_true", help="处理 sources/ 下所有数据集")
    ap.add_argument("--fetch", action="store_true", help="联网下载原始数据到 data/raw/")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    datasets = []
    if args.all:
        from common.paths import SOURCES_DIR
        datasets = sorted(p.stem for p in SOURCES_DIR.glob("*.json") if p.stem != "models")
    if args.dataset:
        datasets.append(args.dataset)
    if not datasets:
        raise SystemExit("给 --dataset <名> 或 --all")

    for ds in datasets:
        d = raw_dir(ds)
        info = fetch_raw(ds, d, force=args.force) if args.fetch else {"source": load_source(ds)}
        p = record_dataset_source(ds, d, info)
        src = info.get("source") or {}
        print(f"[source] {ds}: hf_id={src.get('hf_id')} split={src.get('split')} "
              f"revision={info.get('hf_revision') or src.get('hf_revision') or '(未记录)'} "
              f"rows={info.get('n_rows', '?')} → {rel_to_root(p)}")
        if info.get("error"):
            print(f"[source] ⚠ 下载未成功（只登记了声明）：{info['error']}")


if __name__ == "__main__":
    main()
