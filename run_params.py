#!/usr/bin/env python
"""「这次 run 真正生效的参数」—— 供 `run.py` 打印日志头 + 写进 manifest。

## 为什么需要（踩过的坑）

`apply_config_file()` 原来直接把**注入成功的全部键**逐行打出来。问题是：

1. **顶层键对所有方法注入**，而 `configs/*.json` 的顶层混了几十个方法私有键
   （`IRCOT_BM25_K1` / `FACTALIGN_EVIDENCE_PROPS` / `HGR_*` …）。于是跑 `zeroshot` 时
   日志里照样出现 `topk=20`、`IRCOT_MAX_SENTENCES=10` —— 被误读成"这次 run 用了它"。
2. **只打配置文件给的键**：只存在于**代码默认值**的旋钮（如 `EVIDENCE_WINDOWS`）
   完全不打；被**真实环境变量覆盖**的键只列个名字不打值。
   → 日志里看到的不是"这次 run 的参数"，而是"这份配置文件的键"。

所以：**只打印当前方法真正读的那些键**，并标注每个值的来源。

## 判定口径（与 `tools/audit_params.py` 一致）

- 键名从**源码扫描**得到：某个文件里出现 `os.environ.get("<KEY>"` / `os.environ["<KEY>"]`，
  以及 `DATASET_CHUNK_ENV` 那种"键名写在数据结构里"的间接读法。
- 共用文件（`SHARED_FILES`）里的键对所有方法生效；方法私有键由该方法的适配器文件决定。
- 来源优先级（高→低）：CLI > 真实环境变量 > `configs/*.json` > 代码默认值。
"""

from __future__ import annotations

import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# 与 tools/audit_params.py 保持一致；两边都改（这一份给运行时用，不依赖 tools/）
_NOT_A_KNOB = {
    "YES", "NO", "TRUE", "FALSE", "ON", "OFF", "NONE", "AUTO", "JSON", "UTF", "GET", "POST",
    "HTTP", "HTTPS", "VLLM", "OLLAMA", "OPENAI", "ANSWER", "THOUGHT", "REASON", "QUERY",
    "PASSAGE", "DOC", "SYSTEM", "USER", "ASSISTANT", "ROLE", "CONTENT", "INPUT", "OUTPUT",
}
_KNOWN_NOT_READ: set[str] = set()

# 系统/解释器环境变量：它们被 `run_record.env_snapshot()` 等"读来记录"（用于复现凭证），
# **不是**本项目的旋钮。不排除的话会混进"本次生效参数"里（实测踩过：PROCESSOR_IDENTIFIER
# 被当成旋钮打印出来）。排除它们不动凭证——env_snapshot 照旧记录。
_SYSTEM_ENV = {
    "PROCESSOR_IDENTIFIER", "PROCESSOR_ARCHITECTURE", "OS", "NUMBER_OF_PROCESSORS",
    "CUDA_VISIBLE_DEVICES", "PYTHONHASHSEED", "PYTHONPATH", "PATH", "HOME", "USER",
    "HOSTNAME", "SHELL", "TERM", "LANG", "TMPDIR", "TEMP", "TMP",
    "VLLM_USE_V1", "VLLM_ATTENTION_BACKEND", "VLLM_LOGGING_LEVEL",
}

# 共用文件：这些旋钮对每个方法都生效
SHARED_FILES = ("config.py", "extract.py", "corpus.py", "search.py", "methods/base.py",
                "common/llm.py", "common/paths.py", "common/run_record.py",
                "common/log_setup.py")

# 方法名 → 该方法的适配器文件（相对项目根）
METHOD_FILES = {
    "standardrag": ("methods/standard_rag.py", "methods/corpus_index.py"),
    "factalign": ("methods/factalign.py",),
    "lightrag": ("methods/lightrag_adapter.py",),
    "hypergraphrag": ("methods/hypergraphrag_adapter.py",),
    "hipporag": ("methods/hipporag_adapter.py",),
    "ircot": ("methods/ircot_adapter.py",),
    "zeroshot": ("methods/zeroshot.py",),
    "s2grag": ("methods/s2grag_adapter.py",),
    "selfask": ("methods/selfask_adapter.py",),
    "proprag": ("methods/proprag_adapter.py",),
}

_READ_RE = re.compile(
    r'(?:os\.environ(?:\.get)?\s*[\(\[]\s*|_e(?:i|f|b)?\s*\(\s*)["\']([A-Z][A-Z0-9_]{2,})["\']')
# 间接读：键名写在元组数据结构里再喂给 os.environ.get(k)
# （如 config.DATASET_CHUNK_ENV = (("triviaqa", "TRIVIAQA_CHUNK_TOKENS", "TRIVIAQA_CHUNK_OVERLAP"),)）
_REGISTRY_RE = re.compile(r'\(\s*["\'][a-z0-9_]+["\']\s*,\s*["\']([A-Z][A-Z0-9_]{2,})["\']'
                          r'\s*,\s*["\']([A-Z][A-Z0-9_]{2,})["\']\s*\)')


def _read_call_value(txt: str, key: str) -> bool:
    if re.search(r'os\.environ(?:\.get)?\s*[\(\[]\s*["\']' + key + r'["\']', txt):
        return True
    return bool(re.search(r'_e(?:i|f|b)?\s*\(\s*["\']' + key + r'["\']', txt))


def _scan(txt: str) -> set[str]:
    keys: set[str] = set()
    for k in set(_READ_RE.findall(txt)):
        if k in _NOT_A_KNOB or k in _KNOWN_NOT_READ or k in _SYSTEM_ENV:
            continue
        if _read_call_value(txt, k):
            keys.add(k)
    for m in _REGISTRY_RE.finditer(txt):
        keys.update(k for k in m.groups()
                    if k not in _NOT_A_KNOB and k not in _KNOWN_NOT_READ and k not in _SYSTEM_ENV)
    return keys


def scan_keys_of(files) -> set[str]:
    """这些文件里读到的全部环境变量键（已剔除系统环境变量与非旋钮字面量）。"""
    keys: set[str] = set()
    for rel in files:
        p = ROOT / rel
        if p.exists():
            keys |= _scan(p.read_text(encoding="utf-8", errors="ignore"))
    return keys


def method_param_keys(method: str, dataset: str = "") -> tuple[set[str], set[str]]:
    """返回 (共用键, 方法私有键)。数据集相关键归入"共用"（它对任何方法都生效）。"""
    shared = scan_keys_of(SHARED_FILES)
    own = scan_keys_of(METHOD_FILES.get((method or "").strip().lower(), ()))
    return shared, own


def dataset_scoped_keys(dataset: str = "") -> set[str]:
    """返回**当前数据集不适用**的数据集专属键（应从"本次生效"里剔除）。

    做法：解析 `config.DATASET_CHUNK_ENV`（形如 `(("triviaqa", "TRIVIAQA_CHUNK_TOKENS",
    "TRIVIAQA_CHUNK_OVERLAP"),)`）拿到"数据集 → 键"的对应关系，然后把**不属于当前 dataset**
    的那些键排除。不这么做的话，跑 `ircot --dataset hotpotqa` 也会打印
    `TRIVIAQA_CHUNK_TOKENS=1200` —— 那是 triviaqa 专属口径，对本次 run 毫无作用，
    正是"日志里的无关参数"这一类。
    """
    try:
        import config as _C
        pairs = getattr(_C, "DATASET_CHUNK_ENV", ())
    except Exception:                                            # noqa: BLE001
        return set()
    cur = (dataset or "").strip().lower()
    out: set[str] = set()
    for row in pairs:
        try:
            name, kt, ko = row
        except Exception:                                        # noqa: BLE001
            continue
        if str(name).strip().lower() != cur:
            out.add(str(kt))
            out.add(str(ko))
    return out


def effective_params(method: str, cfg_raw: dict | None = None,
                     kept_env: set[str] | None = None,
                     dataset: str = "") -> dict:
    """当前方法真正生效的参数。

    返回 ``{"shared": {K: (值, 来源)}, "own": {K: (值, 来源)}, "n_other": int}``：

    · 值 = `os.environ[K]`（即最终生效值；CLI 在 argparse 之后另说，这里反映 env 层）。
    · 来源：`env`（真实环境变量覆盖了配置文件）/ `config`（配置文件注入）/
      `code`（只在代码里有默认值，`os.environ` 里不存在）。
    · `n_other` = 配置文件注入了、但**与当前方法无关**的键数（顶层键混进来的，
      就是原来把日志刷成 34 行的那批）。

    `cfg_raw` / `kept_env`：从 `apply_config_file()` 的返回值里取
    （`raw` = 配置原文；`kept_env` = 已被真实环境变量占用、配置文件不覆盖的键）。
    """
    shared, own = method_param_keys(method, dataset)
    # 数据集专属键里**不属于本次 dataset** 的那些：不算"本次生效"
    _ds_off = dataset_scoped_keys(dataset)
    shared = shared - _ds_off
    own = own - _ds_off
    cfg_keys = set()
    if cfg_raw:
        try:
            from common.config_file import config_to_env
            cfg_keys = {k for k in config_to_env(cfg_raw, method=method) if not k.startswith("_")}
        except Exception:                                        # noqa: BLE001
            cfg_keys = set()
    kept_env = set(kept_env or ())

    def _src(k: str) -> str:
        if k in kept_env and k in os.environ:
            return "env"            # 真实环境变量优先（配置文件没覆盖它）
        if k in cfg_keys:
            return "config"         # 配置文件注入的值
        if k in os.environ:
            return "env"            # 配置文件里没有，但环境里有（比如 shell 里 export 的）
        return "code"               # 只有代码默认值

    def _one(k: str) -> tuple[str, str]:
        return (os.environ.get(k, "（代码默认）"), _src(k))

    out: dict = {"shared": {}, "own": {}, "n_other": 0}
    for k in sorted(shared | own):
        if k not in os.environ and k not in cfg_keys:
            continue                # 纯代码默认值：manifest 里已有记录渠道，日志里不必逐条刷
        (out["own"] if k in own else out["shared"])[k] = _one(k)
    relevant = shared | own
    out["n_other"] = len(cfg_keys - relevant)
    return out


def format_effective_lines(params: dict, method: str, config_path: str = "",
                           indent: str = "[config]   ",
                           cli_note: str = "") -> list[str]:
    """把 `effective_params()` 的结果渲染成日志行。

    `cli_note`：CLI 层覆盖的说明（CLI 在所有参数里优先级最高，但它不经过环境变量，
    所以无法从 `os.environ` 反推 —— 由调用方在 argparse 之后把关键项传进来）。
    """
    L = []
    n_sh, n_own = len(params["shared"]), len(params["own"])
    L.append(f"[config] {config_path}" + (f"（method={method}）" if method else "")
             + f" → 本次生效 {n_sh + n_own} 个参数"
             + (f"（共用 {n_sh} + {method} 私有 {n_own}）" if method else "")
             + (f"；另有 {params['n_other']} 个配置键与方法无关、已不再打印" if params["n_other"] else ""))
    for title, kv in ((f"{method} 私有", params["own"]), ("共用", params["shared"])):
        if not kv:
            continue
        L.append(f"[config]   ── {title} ──")
        for k, (v, src) in kv.items():
            mark = {"config": "配置文件", "env": "环境变量", "code": "代码默认"}[src]
            L.append(f"{indent}{k}={v}   ← {mark}")
    if cli_note:
        L.append(f"[config]   CLI 覆盖（优先级最高）：{cli_note}")
    return L
