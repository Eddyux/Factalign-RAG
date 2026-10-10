"""终端 + 分方法日志文件双写（支持按阶段分段：建图 / 检索）。

目录约定：logs/{method}/{dataset}_{sample}[_{tag}][_{阶段}]_{时间戳}.log

用途：所有实验脚本（run.py / test_layer.py 等）跑起来就自动留档，
不用每次手打 `nohup ... > logs/x.log`；带时间戳不覆盖历史，便于事后追溯
（EM/F1、token、报错、vLLM 超时都在同一份文件里）。

分段模式（run.py 用）：同一次 run 会把日志切成互相不重叠的两段——
建图段（到建图结束、build token 落盘为止）与检索段（从检索开始到最终汇总），
两段拼起来就是完整的一次运行输出：
    logs/{method}/{ds}_{n}_v1r1_build_0713_101530.log
    logs/{method}/{ds}_{n}_v1r1_query_0713_101530.log

用法：
    单份日志（老行为，test_layer.py 等）：
        from common.log_setup import setup_log
        setup_log(args.method, args.dataset, args.sample, extra=args.tag)

    分段日志（run.py）：
        from common.log_setup import start_log, switch_log
        tee = start_log(method, dataset, sample, extra="v1r1", segments=True)
        ...
        tee = switch_log("query")     # 建图段封盘，检索段另开一个文件
"""

import os
import sys
import threading
import time
from pathlib import Path

from config import ROOT

# 终端过滤器：fn(写入片段) -> bool（True=允许进终端）。只影响终端那一路，
# 日志文件永远收全量。外部库用裸 print() 刷屏时（如 PRoH 的 DAG 明细）用它挡掉。
_terminal_filter = None

# 当前活动的分段日志（switch_log 用它找到「本进程那一个」tee 切段）。
_active_tee = None

# 阶段名 → 文件名后缀（extra 为空时的兜底，也是分段名）
_STAGE_SUFFIX = {"build": "build", "query": "query"}


def set_terminal_filter(fn):
    """安装/清除终端过滤器（None 清除）。见 `_Tee.write`。"""
    global _terminal_filter
    _terminal_filter = fn


class _Tee:
    """把 stdout/stderr 同时写到终端和日志文件（线程安全：并发 print 不会交错）。

    老接口 `_Tee(term, logf)` 保持兼容；分段日志请用 `_SplitTee`（一个流，
    内部按阶段换文件），这样 sys.stdout / sys.stderr 只被替换一次。
    """

    def __init__(self, *streams):
        self._streams = streams
        self._lock = threading.Lock()

    def write(self, data):
        with self._lock:
            for i, s in enumerate(self._streams):
                # 第 0 路是终端：可被「终端过滤器」挡掉；日志文件始终收全量
                if i == 0 and _terminal_filter is not None:
                    try:
                        if not _terminal_filter(data):
                            continue
                    except Exception:
                        pass
                try:
                    s.write(data)
                    # ★ 每遇到换行就落盘：不然日志文件是 8KB 块缓冲，跑的过程中
                    #   `tail -f` 看到的是空的，要等 close()/进程退出（实测踩过）。
                    if "\n" in data:
                        s.flush()
                except Exception:
                    pass

    def flush(self):
        with self._lock:
            for s in self._streams:
                try:
                    s.flush()
                except Exception:
                    pass

    def isatty(self):
        return False

    def fileno(self):
        """透传底层 fd。

        没有这个方法时，任何直接要真实 fd 的东西都会炸在 sys.stderr 上，例如
        `faulthandler.enable()/dump_traceback_later()`（C 层直接 `file.fileno()`）、
        tqdm、rich、部分 Jupyter/IPython 钩子。取第一个真正能给出 fd 的底层流。
        """
        for s in self._streams:
            try:
                return s.fileno()
            except Exception:
                continue
        raise OSError("_Tee 底层流都没有 fileno()")


class _SplitTee(_Tee):
    """单个 stdout/stderr 替身，内部把内容写进「当前阶段」的日志文件。

    - 终端永远拿全量（受 `set_terminal_filter` 约束）
    - 日志按阶段切文件：open_stage("build") → open_stage("query")
    - `self.logs`：阶段名 → 文件路径；`self.active`：当前阶段名
    """

    def __init__(self, terminal, meta=None):
        # 只铺终端那一路；文件那一路由 open_stage() 动态挂上（不走 _streams）
        super().__init__(terminal)
        self._terminal = terminal
        self._file = None
        self.logs = {}
        self.active = None
        self.meta = dict(meta or {})   # method/dataset/sample/extra/stamp，切段时复用

    def open_stage(self, stage, path, start_ts=None, announce=True):
        """封盘当前段，切到 stage 的新日志文件（追加打开）。

        announce=False：不写那行 `[log] … 段日志` 元信息（stdout/stderr 两个 tee
        指向**同一个文件**，各自 announce 会把每行写两遍）。
        """
        with self._lock:
            if self._file is not None:
                try:
                    self._file.flush()
                    self._file.close()
                except Exception:
                    pass
                self._file = None
            f = open(path, "a", encoding="utf-8")
            self._file = f
            self.active = stage
            self.logs[stage] = Path(path)
        # 元信息只写日志文件本身，不进终端（否则每段会多刷两行）
        if announce:
            self.note(f"[log] {stage} 段日志: {path}"
                      + (f"（起始 {start_ts}）" if start_ts else ""))

    def note(self, text):
        """只写当前段日志文件（不进终端、不走 stdout）：用于分段边界等元信息。"""
        with self._lock:
            if self._file is None:
                return
            try:
                self._file.write(text + "\n")
                self._file.flush()
            except Exception:
                pass

    def switch_file(self, path):
        """换掉当前段的文件句柄（不打印元信息）。配合 `skip_build_log()`。"""
        with self._lock:
            if self._file is not None:
                try:
                    self._file.flush()
                    self._file.close()
                except Exception:
                    pass
            try:
                self._file = open(path, "a", encoding="utf-8")
            except Exception:
                self._file = None
        return self

    def write(self, data):
        with self._lock:
            if _terminal_filter is not None:
                try:
                    allow = _terminal_filter(data)
                except Exception:
                    allow = True
                if not allow:
                    data = ""
            if data:
                try:
                    self._terminal.write(data)
                except Exception:
                    pass
                try:
                    if self._file is not None:
                        self._file.write(data)
                        # ★ 逐行落盘（同 _Tee.write）：否则建图/检索期间日志文件是空的，
                        #   要等切段 close() 或进程退出才出现——"建图完 log 还没写"就是这个。
                        if "\n" in data:
                            self._file.flush()
                except Exception:
                    pass

    def flush(self):
        with self._lock:
            for s in (self._terminal, self._file):
                if s is None:
                    continue
                try:
                    s.flush()
                except Exception:
                    pass

    def fileno(self):
        try:
            return self._terminal.fileno()
        except Exception as e:
            raise OSError(f"_SplitTee 终端流没有 fileno(): {e}")

    def isatty(self):
        try:
            return bool(self._terminal.isatty())
        except Exception:
            return False



# ══════════════════════════════════════════════════════════════════════════════
# 日志头：**版本 + 环境 + 口径**。写进日志文件本身（不进终端，避免刷屏）。
# 为什么必须有：事后看 log 的人要能回答"这是哪版代码/prompt、什么模型、什么切块口径、
# 哪些环境变量被改过"。这些以前只在 results/run_manifest.jsonl 里，log 里看不到。
# ══════════════════════════════════════════════════════════════════════════════

# 影响结果的环境变量前缀（只在被显式设置时才记，避免把无关变量全铺上）
_ENV_PREFIXES = ("HIPPORAG_", "HGR_", "LIGHTRAG_", "PROH_", "IRCOT_",
                 "LLM_", "EMBED_", "EXTRACT_", "CHAT_", "VLLM_", "S2G", "GATE_", "OMP_")


def _env_overrides() -> dict:
    import os
    return {k: v for k, v in sorted(os.environ.items())
            if k.startswith(_ENV_PREFIXES)}


def _server_lines(server_snapshot: dict) -> list[str]:
    """把服务端探测结果渲染成日志头几行（log_setup 顶层只依赖 stdlib，所以这里自己拼）。"""
    chat = server_snapshot.get("chat") or {}
    emb = server_snapshot.get("embedding") or {}
    tk = server_snapshot.get("tokenizer") or {}
    L = [
        f"[cfg] vLLM chat = {chat.get('served_model') or '?'}"
        f" v{chat.get('vllm_version') or '?'}"
        f" max_model_len={chat.get('max_model_len')}",
        f"[cfg] vLLM embed = {emb.get('served_model') or '?'}"
        f" max_model_len={emb.get('max_model_len')}",
        # 这一行是给审稿人看的结论：成本口径取决于 prefix caching
        f"[cfg] prefix caching = {server_snapshot.get('prefix_caching')}"
        f"（{server_snapshot.get('prefix_caching_evidence')}）",
        f"[cfg] 切块 tokenizer = {'可用' if tk.get('available') else '不可用⚠'} {tk.get('model', '')}",
    ]
    if not tk.get("available"):
        # ★ 把**失败原因**打出来：只写"不可用"等于没说 —— 现场只知道"切块退化"，不知道
        #   是缺 `tokenizer_config.json`、还是 offline 闸开着缓存里没有、还是 transformers 没装。
        if tk.get("error"):
            L.append(f"[cfg] ⚠️ tokenizer 加载失败原因：{tk['error']}")
        L.append(f"[cfg] ⚠️ → 切块**退化为字符级**（1200 token 会按 4800 字符切），"
                 f"chunk 数与其他 run 不可比；manifest 的 chunking.tokenizer_failure 也会记下。"
                 f"修法：确认 EMBED_MODEL 指向的目录里真有 tokenizer_config.json + tokenizer.json")
    return L


def log_header_lines(extra: dict | None = None, server_snapshot: dict | None = None) -> list[str]:
    """返回日志头若干行（best-effort：任何一处失败都只是少一行）。

    server_snapshot：`common/server_probe.server_snapshot()` 的结果。带上它以后
    日志头会写明**哪次跑的 vLLM 服务端配置**（prefix caching 开没开、真实 max_model_len、
    vLLM 版本），而不是一行 `max_model_len = None`。
    """
    import os
    import platform
    import socket
    import time

    L: list[str] = []
    add = L.append
    add("=" * 78)
    add(f"[run] 时间 = {time.strftime('%Y-%m-%d %H:%M:%S')}  主机 = {socket.gethostname()}"
        f"  pid = {os.getpid()}")
    try:
        add(f"[run] 工作目录 = {os.getcwd()}")
    except Exception:
        pass
    for k, v in (extra or {}).items():
        add(f"[run] {k} = {v}")
    # ★ P0-3：真实命令行（评审员照抄就能跑）
    try:
        import sys
        add(f"[run] 命令行 = {' '.join(str(a) for a in sys.argv)}")
        add(f"[run] PYTHONHASHSEED = {os.environ.get('PYTHONHASHSEED')}")
    except Exception:
        pass

    # ── 代码/prompt 版本 ──
    try:
        import config as C
        add(f"[ver] python = {platform.python_version()}  platform = {platform.platform()}")
        add(f"[ver] EXTRACT_VERSION = {C.EXTRACT_VERSION}")
        try:
            from common.run_record import env_snapshot, prompt_hashes
            env = env_snapshot()
            g = env.get("git") or {}
            add(f"[ver] git = {g.get('commit', 'unknown')}"
                + (f"（未提交改动 {g.get('dirty_files')} 个文件）" if g.get("dirty") else "（干净）"))
            pk = env.get("packages") or {}
            add("[ver] 包版本 = " + "、".join(
                f"{k}={pk[k]}" for k in ("torch", "vllm", "transformers", "numpy")
                if pk.get(k)))
            if pk.get("arch_list"):
                add(f"[ver] torch arch = {pk['arch_list']}  cuda={pk.get('torch_cuda')}"
                    f"  cuda_available={pk.get('cuda_available')}")
            if env.get("gpu"):
                add("[ver] GPU = " + "；".join(env["gpu"]))
            # ★ P0-2：CPU / 内存 /（服务端）vLLM 版本 —— ARI 要求写清硬件依赖
            cpu = env.get("cpu") or {}
            if cpu:
                add(f"[ver] CPU = {cpu.get('model') or '?'}"
                    f"（{cpu.get('cores_logical') or '?'} 逻辑核）"
                    f"  内存 = {(env.get('memory') or {}).get('total_gb', '?')} GB")
            sv = env.get("vllm_server_versions") or {}
            if sv:
                add("[ver] vLLM 服务端版本 = "
                    + "、".join(f"{k}={v or '?'}" for k, v in sv.items()))
            add("[ver] prompt 哈希 = " + "、".join(f"{k}:{v}" for k, v in prompt_hashes().items()))
        except Exception as e:                                   # noqa: BLE001
            add(f"[ver] 环境快照拿不到（{type(e).__name__}: {e}）")
    except Exception as e:                                       # noqa: BLE001
        add(f"[ver] 版本信息拿不到（{type(e).__name__}: {e}）")

    # ── 模型 / 口径 ──
    try:
        import config as C
        add(f"[cfg] chat = {C.CHAT_MODEL} @ {C.VLLM_BASE_URL}"
            f"   embed = {C.EMBED_MODEL_SERVED} @ {C.VLLM_EMBED_BASE_URL}")
        add(f"[cfg] embed 口径 = dim {C.EMBED_DIM} / 单条截断 {C.EMBED_MAX_TOKENS} token"
            f" / 尺子 {C.EMBED_TOKENIZER} / 查询前缀 {C.EMBED_QUERY_FORMAT}"
            f" / 请求预算 {C.EMBED_REQUEST_TOKENS}")
        # ★ 切块口径行：**默认值 + 实际值**都打。只看默认的 0/0 会让人以为"这个 run 没切块"，
        #   而 triviaqa 实际被切成 1200/100（`config.dataset_chunking`）—— 那是会被写进
        #   manifest / corpus meta 的口径，日志里必须是真的那一份，否则事后对不上账。
        _ds_now = (extra or {}).get("dataset", "")
        _ct_now, _co_now = C.dataset_chunking(_ds_now) if _ds_now else (
            C.EXTRACT_CHUNK_TOKENS, C.EXTRACT_CHUNK_OVERLAP)
        _chunk_line = (f"[cfg] 切块 = {_ct_now}/{_co_now} token（{C.CHUNK_TOKENIZER} 尺子）")
        if (_ct_now, _co_now) != (C.EXTRACT_CHUNK_TOKENS, C.EXTRACT_CHUNK_OVERLAP):
            _chunk_line += (f" ← **{_ds_now} 专属**（全局 {C.EXTRACT_CHUNK_TOKENS}/"
                            f"{C.EXTRACT_CHUNK_OVERLAP}）")
        add(_chunk_line + f"  EXTRACT_WORKERS = {C.EXTRACT_WORKERS}")
        # ★ 生成证据预算：**这才是"喂给模型几条"**。原这里打的是 `SEARCH_TOP_K` /
        #   `SEARCH_TOP_K_CHUNKS`（两个旋钮已删除：生产路径无人消费，见 config.py 注释），
        #   留着会让人把日志里的 `topk=20` 误读成本次 run 的发文数量（真踩过这个坑）。
        #   注意：这两个常量定义在 `methods/base.py`（不在 config.py），别用 C.* 取。
        try:
            from methods import base as _B
            _ew = _B.EVIDENCE_WINDOWS
        except Exception:                                        # noqa: BLE001
            _ew = "—"
        try:
            from methods.factalign import (FACTALIGN_EVIDENCE_PROPS as _fp,
                                           FACTALIGN_EVIDENCE_WINDOWS as _fw)
            _fa = f"；factalign 自己那组 = 命题 {_fp} / 窗口 {_fw}"
        except Exception:                                        # noqa: BLE001
            _fa = ""
        add(f"[cfg] 生成证据预算 = 窗口 {_ew}（命题全发，条数由调用方的 top-N 决定）{_fa}")
        # ★ HF 镜像 / 离线状态要打出来：`HF_ENDPOINT` 若没在 import 前生效，下载会走
        #   huggingface.co 并超时（报错里看不出原因）；离线闸开着而本地没缓存时则必然失败。
        _hf_off = [k for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
                   if os.environ.get(k, "").strip() in ("1", "true", "True", "yes")]
        _em = str(C.EMBED_MODEL)
        _em_ispath = os.path.isdir(_em) or _em.startswith(("/", "~", ".")) or (
            len(_em) > 1 and _em[1] == ":")            # 绝对/相对路径形态
        if os.path.isdir(_em):
            _em_kind = "  ← 本地目录（local_files_only 离线直读）"
        elif _em_ispath:
            _em_kind = "  ← 本地目录路径（本机不存在该目录；服务器需存在）"
        else:
            _em_kind = "  ← HF 名（需能访问镜像/缓存）"
        add(f"[cfg] HF = {os.environ.get('HF_ENDPOINT', '（未设 → 直连 huggingface.co）')}"
            f"{'  离线闸: ' + ','.join(_hf_off) if _hf_off else ''}"
            f"  EMBED_MODEL = {_em}{_em_kind}")
        try:
            from common.llm import tokenizer_failure
            _tf = tokenizer_failure(C.CHUNK_TOKENIZER)
            if _tf:
                add(f"[cfg] ⚠️ 切块尺子 '{C.CHUNK_TOKENIZER}' 不可用：{_tf}"
                    f"\n[cfg] ⚠️ → 切块已**退化为字符级**，语料单元口径与配置不符（见 manifest 的 chunking.tokenizer_failure）")
        except Exception:                                        # noqa: BLE001
            pass
        try:
            from common.llm import (LLM_RETRY_ATTEMPTS, LLM_RETRY_BUDGET_SEC,
                                    GATE_KV_HIGH, GATE_METRICS_SEC)
            add(f"[cfg] 重试 = {LLM_RETRY_ATTEMPTS} 次/预算 {LLM_RETRY_BUDGET_SEC:.0f}s"
                f"  并发闸 KV 阈值 = {GATE_KV_HIGH}（采样 {GATE_METRICS_SEC:.0f}s）")
        except Exception:                                        # noqa: BLE001
            pass
        # ★ P0-1：服务端配置（prefix caching / 真实 max_model_len / tokenizer）
        if server_snapshot:
            try:
                for line in _server_lines(server_snapshot):
                    add(line)
            except Exception:                                    # noqa: BLE001
                pass
        else:
            try:
                from common.llm import _max_model_len
                add(f"[cfg] vLLM max_model_len = {_max_model_len()}"
                    f"（未做服务端探测 → prefix caching 未知，成本结论需注明）")
            except Exception:                                    # noqa: BLE001
                pass
    except Exception as e:                                       # noqa: BLE001
        add(f"[cfg] 口径信息拿不到（{type(e).__name__}: {e}）")

    ov = _env_overrides()
    add("[cfg] 环境变量覆盖 = " + ("、".join(f"{k}={v}" for k, v in ov.items()) if ov else "（无）"))
    add("=" * 78)
    return L

def skip_build_log() -> Path | None:
    """图已存在（本次只是加载）时调用：把建图段文件**改名成检索段**，本 run 只留一份 log。

    "建图就是建图，检索就是检索"——没建图就不该留 `_build_` 文件。做法是把当前那份
    （只有"图已存在，直接加载"几行）改名为 `..._query_{时间戳}.log` 并继续追加，
    随后 `switch_log("query")` 看到 stage 已在 `tee.logs` 里就不会再另开文件。
    """
    tee = _active_tee
    if tee is None:
        return None
    prev = tee.active
    p = tee.logs.get(prev)
    if p is None or not Path(p).exists():
        return None
    meta = tee.meta
    newp = (meta.get("logdir") or ROOT / "logs") / _stamp_filename(
        str(meta.get("dataset", "run")), int(meta.get("sample", 0) or 0),
        str(meta.get("extra", "")), "query",
        str(meta.get("stamp", time.strftime("%m%d_%H%M%S"))))
    try:
        # Windows 下句柄没关就不能 rename（WinError 32）：先全关 → 改名 → 重开
        for s in (sys.stdout, sys.stderr):
            if isinstance(s, _SplitTee) and s._file is not None:
                try:
                    s._file.flush()
                    s._file.close()
                except Exception:
                    pass
                s._file = None
        Path(p).replace(newp)
    except Exception as e:
        print(f"[log] 省略索引构建段日志失败（保持分段）：{e}", flush=True)
        for s in (sys.stdout, sys.stderr):        # 改名失败 → 把旧文件重新接上，别丢日志
            if isinstance(s, _SplitTee):
                s.switch_file(p)
        return None
    for s in (sys.stdout, sys.stderr):
        if isinstance(s, _SplitTee):
            s.switch_file(newp)                  # 句柄指向改名后的文件
            s.logs.pop(prev, None)
            s.logs["query"] = newp               # switch_log 看到已存在 → 不另开
            s.active = "query"
    print(f"[log] 索引已存在（本次未构建）→ 不生成 build log，只留检索段: {newp.name}", flush=True)
    return newp


def _note_once(text: str) -> None:
    """只写**一次**到当前段日志文件。

    stdout / stderr 是**两个 `_SplitTee`，但指向同一个日志文件**——逐个 note 会把每行
    写两遍（实测：日志头重复 2 次）。这里统一走「当前活动的那个 tee」。
    """
    tee = _active_tee
    if tee is not None:
        tee.note(text)
        return
    for s in (sys.stdout, sys.stderr):
        if isinstance(s, _SplitTee):
            s.note(text)
            return


def note_to_log(text: str) -> None:
    """把一行写进**当前段**日志文件（不进终端）。用于"本段用了哪张图"这类段职责说明。"""
    _note_once(text)


def setup_log(method: str, dataset: str, sample: int = 0,
              extra: str = "", server_snapshot: dict | None = None) -> Path:
    """打开单份日志并把 stdout/stderr 双写过去（老行为，不分段）。返回日志路径。

    method → 一级目录（方法名，如 hypergraphrag / lightrag / ours）
    extra  → 可选后缀（如消融 tag：ablate_desc）
    """
    logdir = ROOT / "logs" / method
    logdir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%m%d_%H%M%S")
    name = f"{dataset}_{sample}" + (f"_{extra}" if extra else "") + f"_{stamp}.log"
    path = logdir / name
    # ★ 头先算好（此时 stdout 还是真终端）：header 里会 import 别的模块，
    #   它们顶层若打印警告，不该插进日志文件最上面。
    hdr = log_header_lines({"method": method, "dataset": dataset,
                            "sample": sample, "tag/extra": extra or "(无)"},
                           server_snapshot=server_snapshot)
    f = open(path, "a", encoding="utf-8")
    for line in hdr:                      # 版本 + 环境 + 口径（文件开头）
        try:
            f.write(line + "\n")
        except Exception:
            break
    try:
        f.flush()
    except Exception:
        pass
    sys.stdout = _Tee(sys.__stdout__, f)
    sys.stderr = _Tee(sys.__stderr__, f)
    print(f"[log] 终端输出同时写入: {path}", flush=True)
    return path


def _stamp_filename(dataset: str, sample: int, extra: str, stage: str | None,
                    stamp: str) -> str:
    """{dataset}_{sample}[_{extra}][_{stage}]_{时间戳}.log（stage=None = 单份）。"""
    parts = [f"{dataset}_{sample}"]
    if extra:
        parts.append(extra)
    if stage:
        parts.append(_STAGE_SUFFIX.get(stage, stage))
    return "_".join(parts) + f"_{stamp}.log"


def start_log(method: str, dataset: str, sample: int = 0, extra: str = "",
              segments: bool = False, server_snapshot: dict | None = None):
    """开始记录日志。返回 tee 对象（`tee.logs` / `tee.active` / `switch_log`）。

    segments=False → 单份日志（等价 setup_log，返回 _Tee）。
    segments=True  → 分段日志：先开建图段（stage="build"），
                     建图结束时调用 `switch_log("query")` 封盘并另开检索段。
                     两段时间戳一致，便于把同一次 run 的两份日志对上。
    server_snapshot → `common/server_probe.server_snapshot()`；带上它日志头会写明
                      vLLM 服务端配置（prefix caching / max_model_len / 版本）。
    """
    if not segments:
        return setup_log(method, dataset, sample, extra=extra,
                         server_snapshot=server_snapshot)

    logdir = ROOT / "logs" / method
    logdir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%m%d_%H%M%S")
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    # ★ 头**先算好**：log_header_lines 会 import common.llm 等，那些模块顶层可能打印
    #   警告；此时 sys.stdout 还是真终端，不会把警告插进日志文件开头。
    hdr = log_header_lines({"method": method, "dataset": dataset, "sample": sample,
                            "graph_version/retrieval(tag)": extra},
                           server_snapshot=server_snapshot)
    meta = {"method": method, "dataset": dataset, "sample": sample,
            "extra": extra, "stamp": stamp, "logdir": logdir, "header": hdr}
    query_name = _stamp_filename(dataset, sample, extra, "query", stamp)

    term_out = getattr(sys, "stdout", None) or sys.__stdout__
    term_err = getattr(sys, "stderr", None) or sys.__stderr__
    tee_out = _SplitTee(term_out, meta)
    tee_err = _SplitTee(term_err, meta)
    sys.stdout = tee_out
    sys.stderr = tee_err

    build_path = logdir / _stamp_filename(dataset, sample, extra, "build", stamp)
    for tee in (tee_out, tee_err):      # stderr 那份不 announce（同一文件，写两遍会重复）
        tee.open_stage("build", build_path,
                       start_ts=(started if tee is tee_out else None),
                       announce=(tee is tee_out))
    # ★ 版本 + 环境 + 口径 头（只写文件一次，紧跟 [log] 那行）
    for line in hdr:
        _note_once(line)

    global _active_tee
    _active_tee = tee_out
    print(f"[log] 分段日志（同一次 run 两段，拼起来即完整输出）:"
          f" 建图={build_path.name} → 检索={query_name}", flush=True)
    return tee_out


def switch_log(stage: str, notes: list[str] | None = None):
    """封盘当前日志段，另开 stage 段的新文件（run.py: 建图完 → switch_log("query")）。

    stage 同时是文件名后缀（"query" → ..._query_{时间戳}.log；已知名见 _STAGE_SUFFIX）。
    notes：写进**新段文件开头**的行（不进终端），例如检索段标注"用的哪张图"。
    返回原 tee 对象（active 已切到 stage）；新路径在 `tee.logs[stage]`。
    没调用过 start_log(segments=True) 时返回 None（静默跳过，保证日志问题不影响实验）。
    """
    tee = _active_tee
    if tee is None:
        return None
    prev = tee.active
    if prev == stage:
        return tee
    meta = tee.meta
    path = (meta.get("logdir") or ROOT / "logs") / _stamp_filename(
        str(meta.get("dataset", "run")), int(meta.get("sample", 0) or 0),
        str(meta.get("extra", "")), stage,
        str(meta.get("stamp", time.strftime("%m%d_%H%M%S"))))
    streams = [s for s in (sys.stdout, sys.stderr) if isinstance(s, _SplitTee)]
    # 封盘标记写进「旧」文件，开始标记写进「新」文件（各段自带边界，便于单独看）
    if prev in tee.logs:
        for s in streams:
            s.note(f"=== [{prev}] 段日志封盘: {tee.logs[prev]} ===")
    print(f"\n=== 切换到 [{stage}] 段日志: {path.name} ===", flush=True)
    for s in streams:                   # 两个 tee 都要切文件句柄，但 note 只写一次
        s.open_stage(stage, path, start_ts=None, announce=False)
    _note_once(f"=== [{stage}] 段日志开始"
               + (f"（上一段 [{prev}] 已封盘）" if prev else "") + " ===")
    for line in (notes or []):          # 段职责说明（如：本段检索用的哪张图）
        _note_once(line)
    # ★ 检索段也自带一份「版本 + 环境 + 口径」头：单看 query log 也能自证身份
    for line in (meta.get("header") or []):
        _note_once(line)
    return tee


def flush_logs():
    """把所有日志流 flush 到盘（异常/中断路径上调用，保证已打印内容不丢）。"""
    for s in (sys.stdout, sys.stderr):
        try:
            s.flush()
        except Exception:
            pass
