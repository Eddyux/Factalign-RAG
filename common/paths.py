"""统一路径与版本出处（原地不改动既有目录，只把新东西收进固定位置）。

设计原则：**不迁移既有产物**。`data/{method}/{dataset}/v{n}/`（各 adapter 自管）、
`results/{method}/{dataset}/v{n}r{m}.jsonl`、`logs/{method}/*.log` 全部保持原样，
本模块只负责新增的三类目录：

  configs/                     参数文件（run.py --config → 注入 env → config.py）
  sources/{dataset}.json       入参出处声明（HF id / revision / 下载链接 / license）
  sources/models.json          chat / embedding 模型的精确版本与下载源
  data/DATASET_CARD.md         人读版汇总
  data/raw/{dataset}/          原始下载落点 + provenance.json（sha256）
  data/merged/{dataset}_{n}/   合并语料 + 切块 + 校验（≈14MB/数据集，可进 git）
  data/corpus/{ds}/v{n}/build/ 本方法语料图的复现凭证（config/env/stats/checksums）
  {graph_dir}/build/           外部 baseline 索引目录下的同一套凭证
  results/{method}/{ds}/v{n}r{m}.build.json   -> 上面那份的登记（软链/副本）
  traces/{method}/{ds}/v{n}r{m}/{qid}.json    逐题：prompt + 原始输出 + 每轮检索
  traces/{method}/{ds}/v{n}r{m}/_run.json     本次 run 的 trace 头
  traces/{method}/{ds}/v{n}r{m}/_build.json   本次 run 的**建图口径**（新建/复用 + 产物哈希）
  traces/{method}/{ds}/v{n}r{m}.jsonl         逐题结果的索引行

版本口径（写进 build.json / trace.code，回答"这是哪一版"）：
  dataset_git / 代码 commit / prompt 哈希 / 模型名 / 切块参数 / extract_version
"""

import hashlib
import json
import time
from pathlib import Path

# ⚠️ 这里**故意不 `from config import ...`**：`config.py` 的常量是「import 时读一次
# os.environ」得到的，谁先 import 谁就把参数冻住。而 `--config` 的注入路径是
#   run.py/tools 预扫描 --config → common.config_file → common.paths
# 一旦 paths.py 顶层 import config，config 就会在**注入之前**被加载，
# 配置文件里的 CHAT_MODEL/EMBED_DIM/… 全部静默失效（真实存在过的 bug）。
# 因此这几个目录在这里独立推导（与 config.py 的取值等价：都以仓库根为基准），
# 需要 config 常量时在**函数内部**延迟 import。
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

CONFIGS_DIR = ROOT / "configs"
SOURCES_DIR = ROOT / "sources"
TRACES_DIR = ROOT / "traces"
RAW_DIR = DATA_DIR / "raw"
MERGED_DIR = DATA_DIR / "merged"
DATASET_CARD = DATA_DIR / "DATASET_CARD.md"

# ── 建图类方法的"复现凭证"文件名（tools/write_build_meta.py 写，两者同名同义）──
BUILD_META_FILES = ("config.json", "env.json", "stats.json", "checksums.json")


# ══════════════════════════════════════════════════════════════════════════════
# 语料侧
# ══════════════════════════════════════════════════════════════════════════════
def raw_dir(dataset: str) -> Path:
    """原始下载落点：data/raw/{dataset}/（provenance.json + 原始 jsonl）。"""
    return RAW_DIR / dataset


def merged_dir(dataset: str, sample: int) -> Path:
    """合并语料 + 切块：data/merged/{dataset}_{n}/。"""
    return MERGED_DIR / f"{dataset}_{sample}"


def merged_paths(dataset: str, sample: int) -> dict:
    d = merged_dir(dataset, sample)
    return {"corpus": d / "corpus.jsonl", "chunks": d / "chunks.jsonl",
            "checksums": d / "checksums.json", "build": d / "build.json"}


def source_path(dataset: str) -> Path:
    """入参出处声明：sources/{dataset}.json。"""
    return SOURCES_DIR / f"{dataset}.json"


def sources_models_path() -> Path:
    return SOURCES_DIR / "models.json"


def config_path(name: str) -> Path:
    """参数文件：configs/{name}.json（name 可带 .json）。"""
    return CONFIGS_DIR / (name if name.endswith(".json") else f"{name}.json")


# ══════════════════════════════════════════════════════════════════════════════
# 建图凭证
# ══════════════════════════════════════════════════════════════════════════════
def build_dir(artifact_dir) -> Path:
    """某个图/索引目录下的凭证子目录：{artifact_dir}/build/。

    artifact_dir 由 common.result.graph_dir()（外部 baseline）或
    corpus._corpus_dir()（本方法语料图）给出 —— 两者都适用。
    """
    return Path(artifact_dir) / "build"


def run_build_record(method: str, dataset: str, v: int, r: int) -> Path:
    """建图凭证在结果侧的登记：results/{method}/{dataset}/v{n}r{m}.build.json。"""
    return RESULTS_DIR / method / dataset / f"v{v}r{r}.build.json"


# ══════════════════════════════════════════════════════════════════════════════
# 逐题 trace
# ══════════════════════════════════════════════════════════════════════════════
def trace_run_dir(method: str, dataset: str, v: int, r: int, sample: int = 0) -> Path:
    """逐题 trace 目录：traces/{method}/{dataset}/v{n}r{m}[_n{sample}]/。"""
    tail = f"v{v}r{r}" + (f"_n{sample}" if sample else "")
    return TRACES_DIR / method / dataset / tail


def trace_index_path(method: str, dataset: str, v: int, r: int, sample: int = 0) -> Path:
    """逐题索引：traces/{method}/{dataset}/v{n}r{m}[_n{sample}].jsonl。"""
    d = trace_run_dir(method, dataset, v, r, sample)
    return d.parent / (d.name + ".jsonl")


def question_trace_path(method: str, dataset: str, v: int, r: int, qid: str,
                        sample: int = 0) -> Path:
    """单题 trace：.../v{n}r{m}[_n{sample}]/{qid}.json（qid 做文件系统安全化）。"""
    safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(qid))
    return trace_run_dir(method, dataset, v, r, sample) / f"{safe}.json"


def trace_build_path(method: str, dataset: str, v: int, r: int, sample: int = 0) -> Path:
    """本次 run 的**建图口径**凭证：trace 目录下 `_build.json`。

    回答"这篇论文的建图成本/图内容是哪次跑出来的"：图目录在哪、这次是**新建还是复用**、
    产物字节数与 sha256、建图段 token 消耗。逐题 trace 只覆盖 query 期，建图期不在这里，
    所以单独留一份。
    """
    return trace_run_dir(method, dataset, v, r, sample) / "_build.json"


def rel_to_root(p) -> str:
    """相对仓库根的路径（写进 json 用，换机器也能读）。"""
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except Exception:                       # noqa: BLE001
        return str(p)


# ══════════════════════════════════════════════════════════════════════════════
# 校验/出处小工具
# ══════════════════════════════════════════════════════════════════════════════
def sha256_file(path, chunk: int = 1 << 20) -> str:
    """文件 sha256（分块读，大文件也不吃内存）。不存在返回 "missing"。"""
    p = Path(path)
    if not p.exists() or not p.is_file():
        return "missing"
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def file_info(path) -> dict:
    """文件 → {path, bytes, sha256}（不存在时 bytes=0、sha256="missing"）。"""
    p = Path(path)
    try:
        n = p.stat().st_size if p.exists() else 0
    except Exception:                       # noqa: BLE001
        n = 0
    return {"path": rel_to_root(p), "bytes": n, "sha256": sha256_file(p)}


def provenance_stamp() -> dict:
    """这份产物是"谁/什么时候/哪版代码"造的。"""
    out = {"built_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        import config as C
        out["extract_version"] = C.EXTRACT_VERSION
        out["chat_model"] = C.CHAT_MODEL
        out["embed_model"] = C.EMBED_MODEL_SERVED
        out["chunking"] = {"tokens": C.EXTRACT_CHUNK_TOKENS,
                           "overlap": C.EXTRACT_CHUNK_OVERLAP,
                           "tokenizer": getattr(C, "CHUNK_TOKENIZER", "bge")}
        out["embedding"] = {"served": C.EMBED_MODEL_SERVED, "dim": C.EMBED_DIM,
                            "max_tokens": C.EMBED_MAX_TOKENS,
                            "tokenizer": C.EMBED_TOKENIZER,
                            "query_format": C.EMBED_QUERY_FORMAT}
    except Exception:                       # noqa: BLE001
        pass
    try:
        from common.run_record import env_snapshot
        env = env_snapshot() or {}
        g = env.get("git") or {}
        out["git"] = {"commit": g.get("commit"), "dirty": g.get("dirty"),
                      "dirty_files": g.get("dirty_files")}
        out["packages"] = env.get("packages")
    except Exception:                       # noqa: BLE001
        pass
    return out


def write_json(path, obj) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:                       # noqa: BLE001
        return default
