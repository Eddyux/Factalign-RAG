"""每次 run 的「可追溯身份」记录：manifest / failures / env。

为什么需要（论文/审稿视角）：指标（EM/F1）和成本已经在 summary.jsonl 里了，但
「这次跑的是哪版代码、哪版 prompt、什么环境、哪次采样」没有任何地方落盘 →
审稿人问"你怎么证明两个方法用的是同一套 prompt/环境"时无法回答。

落盘（都在 results/ 下）：
  run_manifest.jsonl  每次 run 一行：代码版本 + 模型 + prompt 哈希 + 解码参数 +
                      切块口径 + 环境指纹（GPU/torch/vllm/CUDA/驱动）
  failures.jsonl      每次 run 一行：失败/重试统计（`<ERROR:>` 的题、重试次数分布、
                      建图侧 adapter 留下的 failed_docs.json 汇总）
  env_<host>.txt      环境快照（人读版，一次/机器，随第一次 run 写）

设计：任何一步失败都不能影响实验本身 → 全部 best-effort，异常只打印不抛。
"""

import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

from config import ROOT, RESULTS_DIR, PROMPTS_DIR, EXTRACT_VERSION, \
    CHAT_MODEL, EMBED_MODEL_SERVED, EMBED_MODEL, \
    EMBED_DIM, EMBED_MAX_TOKENS, EMBED_TOKENIZER, CHUNK_TOKENIZER, \
    EMBED_QUERY_FORMAT, EMBED_REQUEST_TOKENS, \
    VLLM_BASE_URL, VLLM_EMBED_BASE_URL, dataset_chunking
from common.llm import tokenizer_failure
from common.paths import file_info, rel_to_root, provenance_stamp

# embedding 侧的「口径指纹」：换模型/换维度/换尺子/换前缀都会出现在 manifest 里，
# 这样两份结果是不是同一个向量空间，事后能直接读出来（而不是靠翻聊天记录）。
EMBED_SETTINGS = {
    "served": None,            # 在 _embed_settings() 里填（见下）
    "dim": EMBED_DIM,
    "max_tokens": EMBED_MAX_TOKENS,
    "tokenizer": EMBED_TOKENIZER,
    "request_tokens": EMBED_REQUEST_TOKENS,
    "query_format": EMBED_QUERY_FORMAT,
    "chunk_tokenizer": CHUNK_TOKENIZER,
}


def _embed_settings() -> dict:
    d = dict(EMBED_SETTINGS)
    d["served"] = EMBED_MODEL_SERVED
    d["path"] = str(EMBED_MODEL)
    return d


def embed_fingerprint() -> str:
    """向量口径指纹（前 16 位 sha256）：连**切块尺子**一起算。

    用途：把磁盘上缓存的向量（语料 pvecs、实体/窗口向量、图）和"造它时用的 embedder"
    绑在一起。改 `EMBED_DIM` 是硬错（形状就对不上），但改 `EMBED_TOKENIZER`
    （bge→Qwen 的 tokenizer）、`EMBED_MAX_TOKENS`、`CHUNK_TOKENIZER` 之后**形状完全一致**，
    旧向量会被静默复用 → 新旧向量空间混在一起，检索结果不可信且没有任何报错。
    所以缓存判定必须带这个指纹，而不是只看维度和文件是否存在。
    """
    import hashlib
    import json as _json
    d = _embed_settings()
    s = _json.dumps({k: d[k] for k in sorted(d)}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

# 各方法实际用到的 prompt 文件（哈希进 manifest，用于证明"两次跑同一套 prompt"）
PROMPT_FILES = {
    "extract_graph": PROMPTS_DIR / "extract_graph.txt",   # 建图（新方法）
    "rag_prompt": PROMPTS_DIR / "rag_prompt.txt",         # 统一作答（先 Thought 后 Answer）
    # triviaqa 专属作答（S2G-RAG 单跳口径：最短精确 span + canonical alias）
    "rag_prompt_triviaqa": PROMPTS_DIR / "rag_prompt_triviaqa.txt",
    # 无证据作答（zeroshot）：**不按数据集分派**，见 methods/base.rag_prompt_for
    "rag_prompt_nocontext": PROMPTS_DIR / "rag_prompt_nocontext.txt",
}
# 每个方法真正的"身份 prompt"（写进 manifest 的 prompt_used，免得看一堆无关哈希）
# ★ zeroshot 登记的是 `rag_prompt_nocontext`：它出的是最终答案但不经过检索侧路由，
#   以前这里写 `rag_prompt` → manifest 里记的是一个**它从没用过**的 prompt。
METHOD_PROMPTS = {
    "factalign": ["extract_graph", "rag_prompt"],
    "standardrag": ["rag_prompt"],
    "ircot": ["rag_prompt"],
    "hipporag": ["rag_prompt"],
    "lightrag": ["rag_prompt"],
    "hypergraphrag": ["rag_prompt"],
    "s2grag": ["rag_prompt"],
    "selfask": ["rag_prompt"],
    "proprag": ["rag_prompt"],
    "zeroshot": ["rag_prompt_nocontext"],
}
# 数据集专属 prompt：跑到该数据集时，**改为**登记它实际用的那份
# （triviaqa 上所有方法的作答走 `rag_prompt_triviaqa`，见 methods/base.rag_prompt_for）
DATASET_PROMPTS = {"triviaqa": ["rag_prompt_triviaqa"]}
# ★ 无证据路径（zeroshot）**不参与数据集分派**：它的 prompt 只有一个维度
#   （有证据/无证据），triviaqa 那份单跳口径讲的是"从证据里抄哪个 span"，对它不适用。
NO_CONTEXT_METHODS = ("zeroshot",)
# 作答类 prompt 的键（建图类如 extract_graph 不参与数据集替换）
_ANSWER_PROMPT_KEYS = ("rag_prompt", "rag_prompt_triviaqa", "rag_prompt_nocontext")


def method_prompt_keys(method: str, dataset: str = "") -> list[str]:
    """该方法在**该数据集**上实际用到的 prompt 键。

    manifest 是给审稿人看的复现凭证，**只登记真正被读过的文件**：
      · 无证据方法（zeroshot）→ 恒为 `rag_prompt_nocontext`，不随数据集变（见下）；
      · 某数据集有专属 prompt（triviaqa）→ **替换**掉默认那份，不是两份都登记。
        以前这里是"并集"，结果 triviaqa 上每个方法都同时登记 `rag_prompt` 与
        `rag_prompt_triviaqa` —— 实际只读了后者，凭证多出一份从没打开的文件。
    """
    keys = list(METHOD_PROMPTS.get(method, ["rag_prompt"]))
    if (method or "").strip().lower() in NO_CONTEXT_METHODS:
        return keys
    extra = DATASET_PROMPTS.get((dataset or "").strip().lower())
    if not extra:
        return keys
    # 有数据集专属 prompt：把它**替换**进作答键的位置，建图类键（extract_graph）原样保留
    out = [k for k in keys if k not in _ANSWER_PROMPT_KEYS]
    for k in extra:
        if k not in out:
            out.append(k)
    if not out or all(k not in _ANSWER_PROMPT_KEYS for k in out):
        # 方法本来只有作答 prompt → 直接用专属那份
        return list(extra)
    return out

_ENV_CACHE = None


def sha256_of(path: Path, n: int = 16) -> str:
    """文件内容 sha256 前 n 位（不存在返回 "missing"）。"""
    try:
        h = hashlib.sha256(path.read_bytes()).hexdigest()
        return h[:n]
    except Exception:
        return "missing"


def prompt_hashes() -> dict:
    return {k: sha256_of(p) for k, p in PROMPT_FILES.items()}


def _short(s, n: int = 200) -> str:
    s = str(s)
    return s if len(s) <= n else s[:n] + "…"


def _cmd(args: list[str], timeout: int = 15) -> str:
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return (r.stdout or "").strip()
    except Exception:
        return ""


def _git_info() -> dict:
    """代码版本：commit + 是否有未提交改动。非 git 仓库则标 unknown。"""
    if not (ROOT / ".git").exists():
        return {"commit": "no-git", "dirty": None}
    commit = _cmd(["git", "-C", str(ROOT), "rev-parse", "HEAD"]) or "unknown"
    status = _cmd(["git", "-C", str(ROOT), "status", "--porcelain"])
    return {"commit": commit[:12], "dirty": bool(status),
            "dirty_files": len(status.splitlines()) if status else 0}


def _gpu_info() -> list:
    if not shutil.which("nvidia-smi"):
        return []
    out = _cmd(["nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,compute_cap",
                "--format=csv,noheader"])
    return [l.strip() for l in out.splitlines() if l.strip()]


def _cpu_info() -> dict:
    """CPU 型号/核数。ARI 要求写清「有哪些硬件依赖」（如 25GB RAM、多少核）。"""
    d: dict = {}
    try:
        d["machine"] = platform.machine()
        d["cores_logical"] = os.cpu_count()
    except Exception:                               # noqa: BLE001
        pass
    try:                                            # Linux
        for line in Path("/proc/cpuinfo").read_text(errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                d["model"] = line.split(":", 1)[1].strip()
                break
            if "model name" not in d and line.lower().startswith("hardware"):
                d["model"] = line.split(":", 1)[1].strip()
    except Exception:                               # noqa: BLE001
        pass
    if "model" not in d:                            # Windows / macOS
        try:
            if sys.platform.startswith("win"):
                d["model"] = os.environ.get("PROCESSOR_IDENTIFIER", "").strip() or None
            elif sys.platform == "darwin":
                d["model"] = _cmd(["sysctl", "-n", "machdep.cpu.brand_string"]) or None
        except Exception:                           # noqa: BLE001
            pass
    try:                                            # 物理核数（Linux 优先）
        if Path("/proc/cpuinfo").exists():
            n = sum(1 for l in Path("/proc/cpuinfo").read_text(errors="ignore").splitlines()
                    if l.lower().startswith("processor"))
            if n:
                d["cores_logical_procfs"] = n
    except Exception:                               # noqa: BLE001
        pass
    return d


def _mem_info() -> dict:
    """总内存（GB）。"""
    try:
        if Path("/proc/meminfo").exists():
            for line in Path("/proc/meminfo").read_text(errors="ignore").splitlines():
                if line.startswith("MemTotal:"):
                    kb = float(line.split()[1])
                    return {"total_gb": round(kb / 1024 / 1024, 1), "source": "/proc/meminfo"}
    except Exception:                               # noqa: BLE001
        pass
    try:
        if sys.platform.startswith("win"):
            import ctypes

            class _MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            st = _MS()
            st.dwLength = ctypes.sizeof(_MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            return {"total_gb": round(st.ullTotalPhys / 1024 ** 3, 1), "source": "win32"}
        if sys.platform == "darwin":
            out = _cmd(["sysctl", "-n", "hw.memsize"])
            if out:
                return {"total_gb": round(int(out) / 1024 ** 3, 1), "source": "sysctl"}
    except Exception:                               # noqa: BLE001
        pass
    return {}


def _server_versions() -> dict:
    """从 vLLM 服务端读版本（比 pip 包版本可靠：可能是容器里的 nightly）。"""
    out: dict = {}
    try:
        import requests
        for name, base in (("chat", VLLM_BASE_URL), ("embed", VLLM_EMBED_BASE_URL)):
            root = base.rsplit("/v1", 1)[0] if "/v1" in base else base
            try:
                r = requests.get(f"{root}/version", timeout=5)
                if r.status_code == 200:
                    out[name] = (r.json() or {}).get("version")
            except Exception:                       # noqa: BLE001
                pass
    except Exception:                               # noqa: BLE001
        pass
    return out


def _command_line() -> dict:
    """本次执行的真实命令行（ARI：评审员要能照抄执行）。"""
    try:
        argv = list(sys.argv)
    except Exception:                               # noqa: BLE001
        argv = []
    return {
        "argv": argv,
        "command": " ".join(str(a) for a in argv),
        "cwd": os.getcwd(),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
    }


def _pkg_versions() -> dict:
    out = {}
    for mod in ("torch", "vllm", "transformers", "tiktoken", "datasets", "numpy"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "ok")
        except Exception:
            out[mod] = None
    try:
        import torch
        out["torch_cuda"] = torch.version.cuda
        out["cuda_available"] = torch.cuda.is_available()
        out["arch_list"] = torch.cuda.get_arch_list()
    except Exception:
        pass
    return out


def env_snapshot(refresh: bool = False) -> dict:
    """环境指纹（进程内缓存一份，避免每次 run 都跑 nvidia-smi）。"""
    global _ENV_CACHE
    if _ENV_CACHE is not None and not refresh:
        return _ENV_CACHE
    _ENV_CACHE = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "cpu": _cpu_info(),
        "memory": _mem_info(),
        "gpu": _gpu_info(),
        "vllm_server_versions": _server_versions(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "packages": _pkg_versions(),
        "git": _git_info(),
        "command_line": _command_line(),
    }
    return _ENV_CACHE


def write_env_file() -> Path | None:
    """人读版环境快照：results/env_<hostname>.txt（已存在且内容相同则跳过）。"""
    try:
        env = env_snapshot()
        p = RESULTS_DIR / f"env_{env['hostname']}.txt"
        txt = json.dumps(env, ensure_ascii=False, indent=2)
        if p.exists() and p.read_text(encoding="utf-8") == txt:
            return p
        p.write_text(txt, encoding="utf-8")
        return p
    except Exception as e:
        print(f"[record] ⚠ 环境快照写入失败（不影响实验）：{e}", flush=True)
        return None


def _decode_params() -> dict:
    """采样/解码参数（散在代码里的那些，统一记一处）。"""
    d = {"temperature": 0.0, "thinking_default": False}
    try:
        import inspect
        from common.llm import chat
        sig = inspect.signature(chat)
        for k in ("json_mode", "timeout", "thinking", "max_tokens", "flatten"):
            if k in sig.parameters:
                d[f"chat_default_{k}"] = sig.parameters[k].default
    except Exception:
        pass
    return d


def run_settings(extra: dict | None = None, server_snapshot: dict | None = None) -> dict:
    """这次 run 的完整身份：命令行 + 环境指纹（含 CPU/内存/驱动/vLLM 版本）+ 服务端配置。

    给日志头（`common/log_setup.py`）与 manifest 共用一份，避免"日志里看到的"
    和"落盘的"是两套信息。**顶层只依赖 stdlib**：log_setup 会在很早的时候调它，
    此时不能 import 项目内其它模块（会绕圈）。传 server_snapshot 就把服务端配置带上。
    """
    out = {"command_line": _command_line(), "env": env_snapshot()}
    if extra:
        out.update(extra)
    if server_snapshot:
        out["server_config"] = server_snapshot
    return out


def sample_provenance(dataset: str, sample: int) -> dict:
    """**这次跑的是哪份样本文件**：路径 + sha256 + 题数 + 抽样时的 seed。

    为什么必须记：`--sample 300` 只是一个请求值，真正决定"这 300 题是哪 300 题"的是
    `data/samples/{dataset}_{n}.json` 的**抽样 seed**，而 seed 是在上游
    `tools/build_eval_samples.py --seed` 用的，run.py 里根本没有 seed 可记。
    以前 manifest 里只有 `sample=300` 这个数字 → 换台机器/重新抽样后用同名文件覆盖，
    两次 run 会**看起来一模一样**却跑的不是同一批题。这里把文件哈希与 seed 一起钉住。
    """
    out: dict = {"requested": sample, "dataset": dataset}
    try:
        from common.data import sample_path
        p = sample_path(dataset, sample)
    except Exception as e:                          # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
        return out
    out["path"] = rel_to_root(p)
    out["file"] = file_info(p)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        items = data.get("items") or []
        out["n_items"] = len(items)
        # 抽样元信息：不同生成脚本的键名不完全一样，全收进来（没有就不写）
        for k in ("random_seed", "seed", "sample_size", "built_at", "builder",
                  "source", "dataset", "n_hops", "note"):
            if k in data:
                out[k] = data[k]
    except Exception as e:                          # noqa: BLE001
        out["error_items"] = f"{type(e).__name__}: {e}"
    # 同名旁挂元信息文件（tools/build_all.py 会写 {dataset}_{n}.meta.json）
    try:
        mp = p.with_suffix(".meta.json")
        if mp.exists():
            out["meta_file"] = file_info(mp)
            out["meta"] = json.loads(mp.read_text(encoding="utf-8"))
    except Exception:                               # noqa: BLE001
        pass
    return out


def graph_record(method: str, dataset: str, graph_dir, reused: bool,
                 built_now: bool, build_dir=None, tokens: dict | None = None) -> dict:
    """本次 run 用的那张图：在哪、**新建还是复用**、产物指纹、建图段 token。

    为什么必须记：`built=False`（复用旧图）时，这次 run 的 build 成本是 0，图内容取决于
    **更早某次**跑的参数。摘要里只有一行"本次未构建，直接复用已有索引"，没有留档 →
    论文里那张成本表审稿人问"build token 是哪次跑出来的"就答不上来。
    """
    rec: dict = {"reused": bool(reused), "built_this_run": bool(built_now),
                 "method": method, "dataset": dataset,
                 "tokens": dict(tokens or {})}
    if graph_dir:
        rec["graph_dir"] = rel_to_root(graph_dir)
    d = Path(build_dir) if build_dir else (Path(graph_dir) / "build" if graph_dir else None)
    if d and Path(d).exists():
        rec["build_dir"] = rel_to_root(d)
        try:
            files = {}
            for pat in ("config.json", "env.json", "stats.json", "checksums.json"):
                f = Path(d) / pat
                if f.exists():
                    files[pat] = file_info(f)
            rec["build_files"] = files
            st = Path(d) / "stats.json"
            if st.exists():
                rec["stats"] = json.loads(st.read_text(encoding="utf-8"))
        except Exception as e:                      # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"
    return rec


def write_build_record(path, rec: dict) -> Path | None:
    """把建图口径写进 trace 目录（`_build.json`）。失败只警告，不影响实验。"""
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({**provenance_stamp(), **rec}, ensure_ascii=False, indent=2),
                     encoding="utf-8")
        return p
    except Exception as e:                          # noqa: BLE001
        print(f"[record] ⚠ 建图口径写入失败（不影响实验）：{e}", flush=True)
        return None


def method_manifest(method: str, dataset: str, sample: int, v: int, r: int,
                    n_done: int, complete: bool, extra: dict | None = None) -> dict:
    """组装这次 run 的一行 manifest。"""
    ph = prompt_hashes()
    used = method_prompt_keys(method, dataset)
    # ★ 切块口径要按**数据集解析后**的值记录（triviaqa=1200/100，其余=不切块），
    #   不是全局默认值 —— 凭证写错等于没有凭证。
    _ck = dataset_chunking(dataset)
    row = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "method": method, "dataset": dataset, "sample": sample,
        "graph_version": v, "retrieval": r,
        "n_done": n_done, "complete": complete,
        "models": {"chat": CHAT_MODEL, "embed_served": EMBED_MODEL_SERVED,
                   "embed_path": str(EMBED_MODEL)},
        "embed_settings": _embed_settings(),
        "prompt_used": {k: ph.get(k, "missing") for k in used},
        "prompt_all": ph,
        "chunking": {"tokens": _ck[0], "overlap": _ck[1],
                     "tokenizer": CHUNK_TOKENIZER,
                     # ★ 尺子不可用 → 切块退化字符级（`1200 token` 实为 4800 字符）。
                     #   这条必须写进凭证：否则图是按字符切的、账上写着 token，事后对不上。
                     "tokenizer_failure": tokenizer_failure(CHUNK_TOKENIZER)},
        "extract_version": EXTRACT_VERSION,
        "decode": _decode_params(),
        "env": env_snapshot(),
        # ★ 样本溯源（哪个文件 / 哪个 seed / 多少题）+ 建图口径（新建还是复用）
        "sample_file": sample_provenance(dataset, sample),
    }
    if extra:
        row.update(extra)
    return row


def append_jsonl(path: Path, row: dict) -> None:
    """append 一行 jsonl（best-effort：失败只警告，不抛）。"""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[record] ⚠ 写 {path.name} 失败（不影响实验）：{e}", flush=True)


def write_manifest(method: str, dataset: str, sample: int, v: int, r: int,
                   n_done: int, complete: bool, extra: dict | None = None) -> Path:
    write_env_file()
    p = RESULTS_DIR / "run_manifest.jsonl"
    append_jsonl(p, method_manifest(method, dataset, sample, v, r, n_done, complete, extra))
    return p


def build_failure_stats(method: str, dataset: str, v: int) -> dict:
    """建图侧失败汇总：各 adapter 目录下的 failed_docs.json（OpenIE/HGR 重试耗尽留的）。"""
    stats = {}
    try:
        from common.result import graph_dir
        d = graph_dir(method, dataset, v)
        p = Path(d) / "failed_docs.json"
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            stats["failed_docs"] = len(data) if isinstance(data, list) else "?"
            stats["failed_docs_path"] = str(p)
    except Exception:
        pass
    return stats


def write_failures(method: str, dataset: str, sample: int, v: int, r: int,
                   total: int, done: int, retry_hist: dict, errors: list,
                   error_counts: dict | None = None) -> Path:
    """写「这次 run 失败了什么」：`<ERROR:>` 的题、重试次数分布、建图失败块。"""
    p = RESULTS_DIR / "failures.jsonl"
    row = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "method": method, "dataset": dataset, "sample": sample,
        "graph_version": v, "retrieval": r,
        "n_total": total, "n_done": done,
        "n_failed": len(errors),
        "fail_rate": round(len(errors) / done, 4) if done else None,
        "retry_hist": {str(k): int(vv) for k, vv in sorted(retry_hist.items())},
        "error_kinds": error_counts or {},
        "errors": errors[:50],            # 最多留 50 条明细，避免文件爆
    }
    row.update(build_failure_stats(method, dataset, v))
    append_jsonl(p, row)
    return p


def error_kind(msg: str) -> str:
    """把异常信息归成粗粒度类别（便于统计"失败原因分布"）。"""
    s = (msg or "").lower()
    for key, tag in (("does not exist", "model_name_mismatch"),
                     ("404", "http_404"),
                     ("400", "http_400"),
                     ("timeout", "timeout"),
                     ("timed out", "timeout"),
                     ("connection", "connection"),
                     ("max_model_len", "context_overflow"),
                     ("json", "parse_json"),
                     ("no module named", "missing_dependency"),
                     ("unexpected keyword", "version_mismatch"),
                     ("cuda", "cuda")):
        if key in s:
            return tag
    return "other"
