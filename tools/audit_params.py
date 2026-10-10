#!/usr/bin/env python
"""参数审查：`configs/*.json` / `config.py` / `methods/*.py` 三处参数是否对得上。

回答三个具体问题（这三件事过去各踩过一次坑，所以合起来做成工具）：

  ① **`configs/*.json` 里的键真的有人读吗？**（死键 = 配置写了却静默不生效。
     历史上真出现过：`config.py` 写死 `EXTRACT_CHUNK_TOKENS=400`，而配置文件一直在
     提供 `extract_chunk_tokens: 400` —— 改配置文件毫无作用。）
  ② **`methods/*.py` 读的旋钮在配置里存在吗？**（不存在 = 只能靠改代码/export 环境变量，
     配置里看不见、manifest 里也没记，事后无法复现。）
  ③ **跑某个方法时，真正生效的是哪些参数？**（`configs/*.json` 是按方法注入的：
     `methods.<m>.*` 只在跑该方法时注入。但顶层键**对所有方法都注入** —— 所以日志里
     会看到 34 个变量，其中大半跟当前方法无关，容易误判"这个 run 用了 topk=20"。）

用法::

    python tools/audit_params.py                  # 全量审查（三个问题都答）
    python tools/audit_params.py --method ircot   # 只看这个方法真正生效的参数
    python tools/audit_params.py --dead           # 只看死键
    python tools/audit_params.py --unconfigured   # 只看"代码读但配置没写"的旋钮

判定口径（刻意保守，宁少报不误报）：
  · "有人读" = 源码里出现 `os.environ.get("<KEY>"` / `os.environ["<KEY>"]`
    （含 `_e()` / `_ei()` 等封装；`config.py` 的常量写法也算）。
  · "配置提供" = `configs/*.json` 经 `common.config_file.config_to_env()` 展开后的键。
  · 顶层键对所有方法注入；`methods.<m>.*` 只对方法 `m` 注入（前缀由 `method_prefixes` 决定）。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.config_file import config_to_env, load_config_file, method_prefix_for  # noqa: E402

# 这些"看着像环境变量"的字面量其实不是旋钮（值/枚举/协议常量），排除掉免得报告噪音。
_NOT_A_KNOB = {
    "YES", "NO", "TRUE", "FALSE", "ON", "OFF", "NONE", "AUTO", "JSON", "UTF", "GET", "POST",
    "HTTP", "HTTPS", "VLLM", "OLLAMA", "OPENAI", "ANSWER", "THOUGHT", "REASON", "QUERY",
    "PASSAGE", "DOC", "SYSTEM", "USER", "ASSISTANT", "ROLE", "CONTENT", "INPUT", "OUTPUT",
}

# 方法名 → 该方法的适配器文件（用于"这个方法读哪些旋钮"）
METHOD_FILES = {
    "standardrag": ["methods/standard_rag.py", "methods/corpus_index.py"],
    "factalign": ["methods/factalign.py"],
    "lightrag": ["methods/lightrag_adapter.py"],
    "hypergraphrag": ["methods/hypergraphrag_adapter.py"],
    "hipporag": ["methods/hipporag_adapter.py"],
    "ircot": ["methods/ircot_adapter.py"],
    "zeroshot": ["methods/zeroshot.py"],
    "s2grag": ["methods/s2grag_adapter.py"],
    "selfask": ["methods/selfask_adapter.py"],
    "proprag": ["methods/proprag_adapter.py"],
}

# 所有方法共用的文件（基类 / 建图 / LLM / 日志）——这些旋钮对每个方法都生效。
# ★ `extract.py` 必须在内：建图的抽取旋钮（如 `EXTRACT_TRIES`）定义在**根目录**的这个文件里，
#   漏掉它会把 `EXTRACT_TRIES` 误报成死键。
SHARED_FILES = ["config.py", "extract.py", "corpus.py", "search.py", "methods/base.py",
                "common/llm.py", "common/paths.py", "common/run_record.py",
                "common/log_setup.py"]

_READ_RE = re.compile(
    r'(?:os\.environ(?:\.get)?\s*[\(\[]\s*|_e(?:i|f|b)?\s*\(\s*)["\']([A-Z][A-Z0-9_]{2,})["\']')
# 键名写在**数据结构里**、再喂给 `os.environ.get(k)` 的间接读法。必须单独认，否则误报死键。
# 例：`config.DATASET_CHUNK_ENV = (("triviaqa", "TRIVIAQA_CHUNK_TOKENS", "TRIVIAQA_CHUNK_OVERLAP"),)`
# 之后 `os.environ.get(kt)` —— 正则只看得到 `get(kt)`，看不见键名字面量。
_REGISTRY_RE = re.compile(r'\(\s*["\'][a-z0-9_]+["\']\s*,\s*["\']([A-Z][A-Z0-9_]{2,})["\']'
                          r'\s*,\s*["\']([A-Z][A-Z0-9_]{2,})["\']\s*\)')
# 已知"配置里有、但代码里是写死常量、不读环境变量"的键（改它要改代码）。
# 列出来是为了把它们和**真正的死键**分开：真死键是"写了以为能调，其实代码根本没接"。
# ★ 目前为空：`BACKEND` 原来在这里，后来把它改成读环境变量了（config.py），
#   于是它不再是死键，也不该再被豁免 —— 豁免名单必须跟着代码走，否则会掩盖回归。
_KNOWN_NOT_READ: set[str] = set()
_DEF_RE = re.compile(r'^([A-Z][A-Z0-9_]{2,})\s*=\s*(?:int|float|str|bool)?\s*[\(\s]*'
                     r'(?:os\.environ\.get\(\s*["\'][A-Z][A-Z0-9_]{2,}["\']\s*,\s*)?(.*)$')


def _py_files() -> list[Path]:
    # ★ 根目录**全部** .py 都要扫：`extract.py` / `corpus.py` / `search.py` / `run.py`
    #   都在根目录，且 `extract.py` 就定义了 `EXTRACT_TRIES` 这类旋钮 ——
    #   只扫 config.py + methods/ + common/ 会把它误报成死键（真踩过）。
    return (sorted(ROOT.glob("*.py"))
            + sorted((ROOT / "methods").glob("*.py"))
            + sorted((ROOT / "common").glob("*.py")))


def _read_call_value(txt: str, key: str) -> bool:
    """源码里是否真的**读**了这个键（而不是仅出现在注释/文档字符串里）。"""
    if re.search(r'os\.environ(?:\.get)?\s*[\(\[]\s*["\']' + key + r'["\']', txt):
        return True
    return bool(re.search(r'_e(?:i|f|b)?\s*\(\s*["\']' + key + r'["\']', txt))


def scan_read_keys() -> dict[str, list[str]]:
    """每个真正被读的环境变量键 → 读它的文件列表。"""
    out: dict[str, list[str]] = {}
    for f in _py_files():
        txt = f.read_text(encoding="utf-8", errors="ignore")
        rel = str(f.relative_to(ROOT)).replace("\\", "/")
        keys: list[str] = []
        for k in set(_READ_RE.findall(txt)):
            # 正则出来的要二次确认「确实是读」（避免把注释/文档里的字面量算进来）
            if _read_call_value(txt, k) and k not in _NOT_A_KNOB and k not in _KNOWN_NOT_READ:
                keys.append(k)
        for m in _REGISTRY_RE.finditer(txt):      # 间接读：键名写在元组数据结构里
            # 这里已在数据里确证，不必再要求 `get("字面量")`
            keys += [k for k in m.groups()
                     if k not in _NOT_A_KNOB and k not in _KNOWN_NOT_READ]
        for k in keys:
            out.setdefault(k, []).append(rel)
    return {k: sorted(set(v)) for k, v in out.items()}


def scan_code_defaults() -> dict[str, tuple[str, str]]:
    """`config.py` 里的代码默认值：键 → (默认值字面量, 文件:行)。"""
    out: dict[str, tuple[str, str]] = {}
    txt = (ROOT / "config.py").read_text(encoding="utf-8")
    for i, line in enumerate(txt.splitlines(), 1):
        m = re.match(r'^([A-Z][A-Z0-9_]{2,})\s*=\s*(.+)$', line)
        if not m:
            continue
        name, rhs = m.group(1), m.group(2)
        g = re.search(r'os\.environ\.get\(\s*["\']([A-Z][A-Z0-9_]{2,})["\']\s*(?:,\s*(.*?))?\)', rhs)
        if not g:
            continue
        out[g.group(1)] = ((g.group(2) or "").strip().rstrip(")").strip(), f"config.py:{i}")
    return out


def scan_config_keys() -> dict[str, dict[str, list[str]]]:
    """`configs/*.json` 展开后的键：键 → {值(JSON 原文), [来源文件]}。

    顶层键对**所有**方法注入；`methods.<m>.*` 只对 `m` 注入。
    """
    out: dict[str, dict[str, list[str]]] = {}
    for f in sorted((ROOT / "configs").glob("*.json")):
        cfg = load_config_file(f)
        methods = [m for m in (cfg.get("methods") or {}) if not str(m).startswith("_")]
        # 顶层键：method="" 注入
        for k, v in config_to_env(cfg, method="").items():
            if k.startswith("_"):
                continue
            e = out.setdefault(k, {"values": [], "files": []})
            e["values"].append(_fmt(v))
            e["files"].append(f.name)
        for m in methods:
            for k, v in config_to_env(cfg, method=m).items():
                if k.startswith("_"):
                    continue
                e = out.setdefault(k, {"values": [], "files": []})
                e["values"].append(_fmt(v))
                e["files"].append(f"{f.name}(methods.{m})")
    return out


def _fmt(v) -> str:
    if isinstance(v, str):
        return v if len(v) <= 40 else v[:37] + "…"
    return json.dumps(v, ensure_ascii=False)


def method_active_keys(method: str) -> tuple[list[str], list[str]]:
    """跑 `method` 时真正生效的键：返回 (共用键, 该方法私有键)。"""
    shared = [str(ROOT / p) for p in SHARED_FILES if (ROOT / p).exists()]
    own = [str(ROOT / p) for p in METHOD_FILES.get(method, []) if (ROOT / p).exists()]
    all_shared = scan_keys_of(shared)
    all_own = scan_keys_of(own)
    return sorted(all_shared - all_own), sorted(all_own)


def scan_keys_of(files: list[str]) -> set[str]:
    keys: set[str] = set()
    for p in files:
        txt = Path(p).read_text(encoding="utf-8", errors="ignore")
        for k in set(_READ_RE.findall(txt)):
            if k in _NOT_A_KNOB or k in _KNOWN_NOT_READ:
                continue
            if not _read_call_value(txt, k):
                continue
            keys.add(k)
        for m in _REGISTRY_RE.finditer(txt):
            keys.update(k for k in m.groups()
                        if k not in _NOT_A_KNOB and k not in _KNOWN_NOT_READ)
    return keys


def _prefix_of(method: str, cfg_name: str = "") -> str:
    prefix = method_prefix_for({}, method)
    return prefix or method.upper()


def main() -> int:
    ap = argparse.ArgumentParser(description="参数审查（配置/代码/方法三处是否对得上）")
    ap.add_argument("--method", default="", help="只看该方法真正生效的参数")
    ap.add_argument("--dead", action="store_true", help="只看死键（配置提供但无人读）")
    ap.add_argument("--unconfigured", action="store_true",
                    help="只看代码读但配置里没写的旋钮")
    args = ap.parse_args()

    read_keys = scan_read_keys()
    code_def = scan_code_defaults()
    cfg_keys = scan_config_keys()

    only_dead = args.dead
    only_unconf = args.unconfigured

    if args.method:
        shared, own = method_active_keys(args.method)
        cfg_for = {}
        for f in sorted((ROOT / "configs").glob("*.json")):
            cfg = load_config_file(f)
            for k, v in config_to_env(cfg, method=args.method).items():
                if not k.startswith("_"):
                    cfg_for.setdefault(k, []).append(f"{f.name}={_fmt(v)}")
        print(f"══ {args.method} 真正生效的参数 ══")
        print(f"（共用 {len(shared)} + 私有 {len(own)}；`configs/*.json` 的顶层键对所有方法"
              f"注入，`methods.<m>.*` 只对 {args.method} 注入）\n")
        for title, keys in (("共用（基类/LLM/日志）", shared), (f"{args.method} 私有", own)):
            print(f"── {title} " + "─" * max(0, 46 - len(title)))
            for k in keys:
                src = cfg_for.get(k)
                where = ",".join(src) if src else f"代码默认 {code_def.get(k, ('?',))[0]}"
                print(f"  {k:38s} ← {where}")
            print()
        print("怎么改：① 改 configs/*.json 顶层键（对所有方法生效）；"
              "② 改 methods.<m>.*（只对该方法生效）；③ export 环境变量（临时、优先级最高）")
        return 0

    dead = sorted(k for k in cfg_keys if k not in read_keys)
    unconf = sorted(k for k in read_keys if k not in cfg_keys)
    ok = sorted(k for k in cfg_keys if k in read_keys)

    if not only_unconf:
        print(f"══ ① 配置里的键有没有人读 ══  提供 {len(cfg_keys)} / 有人读 {len(ok)} "
              f"/ **死键 {len(dead)}**")
        if dead:
            for k in dead:
                print(f"  ✗ {k:38s} ← {','.join(sorted(set(cfg_keys[k]['files'])))}")
        else:
            print("  ✓ 无死键")
        print()

    if not only_dead:
        print(f"══ ② 代码读、但配置里没写的旋钮 ══  **{len(unconf)} 个**")
        print("  （不写进配置 = 只能靠改代码或 export 环境变量；配置里看不见、"
              "manifest 里也没记，事后无法复现）")
        for k in unconf:
            d = code_def.get(k)
            print(f"  · {k:38s} 代码默认 {d[0] if d else '（非 config.py 常量）':16s}"
                  f" 读于 {','.join(read_keys[k])}")
        print()

    if not only_dead and not only_unconf:
        print("══ ③ 小结 ══")
        print(f"  configs/*.json 提供 {len(cfg_keys)} 键，其中 {len(dead)} 个是死键")
        print(f"  代码读 {len(read_keys)} 键，其中 {len(unconf)} 个配置里没写")
        print(f"  用法：`--method <名字>` 看该方法真正生效的参数；`--dead` / `--unconfigured` 单看一类")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())