"""把「复现凭证」写进图/索引目录：config.json + env.json + stats.json + checksums.json。

用户的 requirement 2/3：
  2. 标注输入版本 —— 语料来源/版本/链接（sources/*.json + data/raw/*/provenance.json）、
     构建工具与模型精确版本（本模块的 env.json）、切块/参数（config.json）
  3. 留存校验信息 —— 实体/关系数量统计（stats.json）、产物 sha256（checksums.json）

**不含**逐行抽取中间结果（requirement 4）。

两种用法：
  ① 跑完一次 run（library）：`write_build_meta_for_run(method, dataset, sample, v, r, method_obj=…)`
     图目录取 `common.result.graph_dir()`；本方法语料图取 `data/corpus/{ds}/v{n}/`。
  ② 事后补写（CLI）：`python tools/write_build_meta.py --dir data/proprag/hotpotqa/v1`

写出来的东西（都在 `{图目录}/build/` 下）：
  config.json    本次建图的完整参数（含 --config 原文、env 覆盖、切块口径）
  env.json       python/torch/vllm/transformers/igraph/numpy + GPU + git commit
  stats.json     ★ 实体数 / 关系数（三元组）/ 命题数 / 超边数 / chunk 数 / 图顶点边数
  checksums.json ★ 图目录下所有"小"产物的 sha256 + 字节数（.npy 等大文件只记字节数）
另在 results/{method}/{dataset}/v{n}r{m}.build.json 留一份登记（指向上面这堆）。
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.paths import (BUILD_META_FILES, build_dir, file_info, provenance_stamp,
                          rel_to_root, run_build_record, sha256_file, write_json)
from common.result import graph_dir
from config import CORPUS_DIR

# checksums 里只对小于这个体积的文件算 sha256（大 .npy 仍记 bytes，但跳过哈希）
HASH_MAX_MB = 64
# 这些后缀不参与 checksums（临时/缓存/大二进制）
_SKIP_SUFFIX = {".log", ".tmp", ".lock", ".pid"}


# ══════════════════════════════════════════════════════════════════════════════
# 1) 图目录定位
# ══════════════════════════════════════════════════════════════════════════════
def resolve_graph_dirs(method: str, dataset: str, v: int) -> list[Path]:
    """该方法这次用的**所有**图/索引目录（存在才返回）。

    - 外部 baseline / 新注册基线：data/{method}/{dataset}/v{n}/
    - 本方法（factalign）：data/corpus/{dataset}/v{n}/
    """
    cands = [graph_dir(method, dataset, v)]
    if method == "factalign":
        cands.append(CORPUS_DIR / dataset / f"v{v}")
    out, seen = [], set()
    for c in cands:
        if c.exists() and c.resolve() not in seen:
            out.append(c)
            seen.add(c.resolve())
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 2) stats：实体 / 关系 / 命题 / 超边 / chunk 计数
# ══════════════════════════════════════════════════════════════════════════════
def _count_json_list(p: Path):
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                   # noqa: BLE001
        return None
    if isinstance(obj, list):
        return len(obj)
    if isinstance(obj, dict):
        return len(obj)
    return None


def collect_stats(method_obj=None, dirs=None, work_dir=None) -> dict:
    """尽量把「图规模」统计出来（拿不到的键就不写，不编数字）。"""
    st: dict = {"counted_at": time.strftime("%Y-%m-%d %H:%M:%S"), "method": None,
                "sources": []}

    # ① 从内存里的 adapter 对象上读（最准：就是本次真正建出来的）
    if method_obj is not None:
        st["method"] = getattr(method_obj, "name", None)
        probe = {
            "entities": ("entities", "entity_ids", "ent_node_to_num_chunk", "graph"),
            "hyperedges": ("hyperedges",),
            "chunks": ("chunks", "passages", "_chunks", "docs"),
            "propositions": ("prop_ids", "proposition_texts", "proposition_to_entities_map",
                            "propositions"),      # ★ factalign 的属性已改名 proposition(s)
            "relations": ("triples", "relations", "edges"),
        }
        for key, attrs in probe.items():
            for a in attrs:
                if not hasattr(method_obj, a):
                    continue
                v = getattr(method_obj, a)
                try:
                    n = v.vcount() if a == "graph" else len(v)
                except Exception:                       # noqa: BLE001
                    n = None
                if isinstance(n, int) and n > 0:
                    st.setdefault(key, {})[f"from.{a}"] = n
                    break
        # ★ 语料**包在检索底座对象里**的方法：s2grag 只把语料单元存在
        #   `self._index`（`methods/corpus_index.py` 的 `CorpusIndex.units`），**不落盘**
        #   （`run.py:455-460` 已注明它是"纯内存、从不落盘"）。上面那批探针只认
        #   顶层 list 属性（chunks/passages/_chunks/docs）→ 一个都命不中 →
        #   `summary` 为空 → 验收里同时挂两条：
        #     「图规模统计非空」FAIL +「chunk 数与 merged 一致」WARN（缺一方 图=None）
        #   明明 Search 用的就是这些单元，**数得出来**，所以这里补一条探针。
        #   取 `units` 而不是 `len(_index)`：后者还会把 RAPTOR 那种派生节点（level≥1）
        #   算进"chunk 数"，与 merged 的 `n_chunks`（叶子块）口径不同。
        _idx = getattr(method_obj, "_index", None)
        if _idx is not None and "chunks" not in st:
            _units = getattr(_idx, "units", None)
            if isinstance(_units, (list, tuple)) and len(_units) > 0:
                st["chunks"] = {"from._index.units": len(_units)}
        g = getattr(method_obj, "graph", None)
        if g is not None:
            try:
                st["graph"] = {"vertices": g.vcount(), "edges": g.ecount()}
            except Exception:                           # noqa: BLE001
                pass
        if getattr(method_obj, "entity_vecs", None) is not None:
            try:
                st.setdefault("entities", {})["vectors"] = int(method_obj.entity_vecs.shape[0])
            except Exception:                           # noqa: BLE001
                pass

    # ② 从盘上的产物文件数（adapter 各自落盘的 json/npz）
    for d in (dirs or []):
        info = {"dir": rel_to_root(d), "files": {}}
        for name in ("entities.json", "hyperedges.json", "propositions.json", "chunks.json"):
            p = Path(d) / name
            if p.exists():
                n = _count_json_list(p)
                if n is not None:
                    info["files"][name] = n
        # npz 产物（如 proprag 的 artifacts.npz）：按数组名给计数
        for name in ("artifacts.npz",):
            p = Path(d) / name
            if not p.exists():
                continue
            try:
                with np.load(p, allow_pickle=True) as z:
                    keys = list(z.files)
                    counts = {k: int(z[k].shape[0]) for k in keys if z[k].ndim >= 1}
                info["files"][name] = {"bytes": p.stat().st_size, "arrays": counts}
                # 映射到统一键名
                for arr, key in (("chunks", "chunks"), ("passages", "chunks"),
                                 ("entity_ids", "entities"), ("prop_ids", "propositions"),
                                 ("graph_edges", "graph_edges")):
                    if arr in counts:
                        st.setdefault(key if key != "graph_edges" else "graph_meta",
                                      {})[f"npz.{arr}"] = counts[arr]
            except Exception as e:                          # noqa: BLE001
                info["files"][name] = {"bytes": p.stat().st_size,
                                       "error": f"{type(e).__name__}: {e}"}
        for name in ("hvecs.npy", "vdb_entity.parquet"):
            p = Path(d) / name
            if p.exists():
                info["files"][name] = {"bytes": p.stat().st_size}
        if (Path(d) / "meta.json").exists():
            info["meta"] = json.loads((Path(d) / "meta.json").read_text(encoding="utf-8"))
        st["sources"].append(info)

    # ③ 汇总一个「人看的一行」（各来源取最大值，避免同一量被重复计两次）
    flat = {}
    for key in ("entities", "relations", "propositions", "hyperedges", "chunks"):
        node = st.get(key)
        if isinstance(node, dict):
            vals = [v for v in node.values() if isinstance(v, int)]
            if vals:
                flat[key] = max(vals)
    if flat:
        st["summary"] = flat
    return st


# ══════════════════════════════════════════════════════════════════════════════
# 3) checksums
# ══════════════════════════════════════════════════════════════════════════════
def collect_checksums(dirs, max_mb: int = HASH_MAX_MB) -> dict:
    out = {"hashed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
           "policy": f"sha256 for files <= {max_mb}MB; larger files: bytes only",
           "roots": []}
    for d in (dirs or []):
        d = Path(d)
        files = []
        for p in sorted(d.rglob("*")):
            if not p.is_file() or p.suffix in _SKIP_SUFFIX:
                continue
            # 不重复登记 build/ 自己的产物
            if p.parent.name == "build" and p.name in BUILD_META_FILES:
                continue
            try:
                size = p.stat().st_size
            except Exception:                           # noqa: BLE001
                continue
            rec = {"path": rel_to_root(p), "bytes": size}
            if size <= max_mb * 1024 * 1024:
                rec["sha256"] = sha256_file(p)
            files.append(rec)
        out["roots"].append({"dir": rel_to_root(d), "n_files": len(files), "files": files})
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 4) 写四件套（+ 结果侧登记）
# ══════════════════════════════════════════════════════════════════════════════
def write_build_meta(artifact_dir, method: str = "", dataset: str = "", sample: int = 0,
                     v: int = 1, r: int = 1, method_obj=None, config_file=None,
                     built: bool = True, extra: dict | None = None) -> dict:
    """给一个图/索引目录写 build/{config,env,stats,checksums}.json，返回 {名: 路径}。"""
    d = Path(artifact_dir)
    bd = build_dir(d)
    bd.mkdir(parents=True, exist_ok=True)

    cfg = {
        **provenance_stamp(),
        "method": method or getattr(method_obj, "name", ""),
        "dataset": dataset, "sample": sample, "graph_version": v, "retrieval": r,
        "artifact_dir": rel_to_root(d),
        "built_this_run": bool(built),
        "config_file": rel_to_root(config_file) if config_file else None,
        "config_values": None,
        "env_overrides": {k: val for k, val in sorted(os.environ.items())
                          if k.startswith(("HIPPORAG_", "HGR_", "LIGHTRAG_",
                                           "IRCOT_", "SELFASK_", "PROPRAG_",
                                           "S2G", "LLM_", "EMBED_", "EXTRACT_", "CHAT_",
                                           "VLLM_", "GATE_", "OMP_"))},
    }
    if config_file:
        try:
            from common.config_file import load_config_file
            cfg["config_values"] = load_config_file(config_file)
        except Exception as e:                          # noqa: BLE001
            cfg["config_values_error"] = f"{type(e).__name__}: {e}"
    if extra:
        cfg.update(extra)

    env = {}
    try:
        from common.run_record import env_snapshot
        env = env_snapshot() or {}
    except Exception as e:                              # noqa: BLE001
        env = {"error": f"{type(e).__name__}: {e}"}
    env = {"captured_at": time.strftime("%Y-%m-%d %H:%M:%S"), **env}
    try:
        import igraph                                     # type: ignore
        env.setdefault("packages", {})["python-igraph"] = getattr(igraph, "__version__", "?")
    except Exception:                                     # noqa: BLE001
        pass

    stats = collect_stats(method_obj=method_obj, dirs=[d])

    paths = {}
    paths["config"] = write_json(bd / "config.json", cfg)
    paths["env"] = write_json(bd / "env.json", env)
    paths["stats"] = write_json(bd / "stats.json", stats)
    paths["checksums"] = write_json(bd / "checksums.json", collect_checksums([d]))
    return paths


def write_build_meta_for_run(method: str, dataset: str, sample: int, v: int, r: int,
                             method_obj=None, config_file=None, built: bool = True):
    """run.py 的入口：对本次用到的所有图目录写凭证，并在结果侧登记。"""
    dirs = resolve_graph_dirs(method, dataset, v)
    if not dirs:
        return None
    paths = None
    for d in dirs:
        paths = write_build_meta(d, method=method, dataset=dataset, sample=sample, v=v, r=r,
                                 method_obj=method_obj, config_file=config_file, built=built)
    # 结果侧登记：指向那堆凭证（复现者从 results/ 就能找到）
    rec = {
        "method": method, "dataset": dataset, "sample": sample,
        "graph_version": v, "retrieval": r,
        "built_this_run": bool(built),
        "artifact_dirs": [rel_to_root(d) for d in dirs],
        "build_meta": {k: rel_to_root(p) for k, p in (paths or {}).items()},
        "summary": (json.loads((build_dir(dirs[0]) / "stats.json").read_text(encoding="utf-8"))
                    .get("summary") if (build_dir(dirs[0]) / "stats.json").exists() else None),
        "at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(run_build_record(method, dataset, v, r), rec)
    return build_dir(dirs[0])


# ══════════════════════════════════════════════════════════════════════════════
# CLI（事后补写）
# ══════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="写/补写某个图目录的复现凭证")
    ap.add_argument("--dir", action="append", required=True,
                    help="图/索引目录（可多次），如 data/proprag/hotpotqa/v1")
    ap.add_argument("--method", default="")
    ap.add_argument("--dataset", default="")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--graph-version", type=int, default=1)
    ap.add_argument("--retrieval", type=int, default=1)
    ap.add_argument("--config", default=None, help="本次用的 configs/*.json（写进 config.json）")
    args = ap.parse_args()

    for d in args.dir:
        p = Path(d)
        if not p.exists():
            print(f"跳过（不存在）：{p}")
            continue
        paths = write_build_meta(p, method=args.method, dataset=args.dataset,
                                 sample=args.sample, v=args.graph_version,
                                 r=args.retrieval, config_file=args.config)
        print(f"[build-meta] {p}")
        for k, v in paths.items():
            print(f"    {k:10s} → {rel_to_root(v)}")
        st = json.loads(paths["stats"].read_text(encoding="utf-8"))
        print(f"    计数：{st.get('summary')}")


if __name__ == "__main__":
    main()
