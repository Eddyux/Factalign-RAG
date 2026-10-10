"""配置文件 → 环境变量（在 import config / adapter **之前**生效）。

为什么需要：现在参数散在三处 —— `config.py` 的常量、各 adapter 顶部的
`os.environ.get("XXX_...")`、以及 command line。用户要的是「参数通过配置文件管理」。

做法（**不动任何现有代码的读取方式**）：配置文件的扁平键注入 `os.environ`，
于是 `config.py` 与所有 adapter 的 `os.environ.get(...)` 照常工作。优先级：

    CLI 显式参数  >  真实环境变量（已是环境变量）  >  配置文件  >  代码默认值

即：配置文件**不覆盖**已经存在的环境变量（同一个键），但会被 CLI 覆盖。
必须在 `import config`（和任何 adapter）之前调用 `apply_config_file()`。

配置文件格式（JSON，也接受 YAML 若装了 pyyaml）：

```jsonc
{
  "chat_model": "Qwen/Qwen2.5-7B-Instruct",      // → CHAT_MODEL
  "chunking": {"tokens": 400, "overlap": 40},    // → CHUNKING_TOKENS / CHUNKING_OVERLAP
  "methods": {                                   // 按方法分组（只在跑该方法时注入）
     "proprag": {"top_docs": 20, "syn_threshold": 0.8}   // → PROPRAG_TOP_DOCS ...
  },
  "defaults": {"workers": 8, "sample": 500}      // 非 env 键：只作为 run 默认值
}
```
键名规则：`a_b` → `A_B`（全大写），嵌套用 `_` 连接；`methods.<m>.<k>` → `<M>_<K>` 大写。
值：bool → "1"/"0"，list → 逗号连接，None → 跳过（不注入）。
"""

import json
import os
import sys
from pathlib import Path

from common.paths import config_path, rel_to_root

# 非 env 键（run 级默认值，不注入 os.environ）
_NON_ENV_KEYS = {"defaults", "method_prefixes"}


def method_prefix_for(cfg: dict, method: str) -> str:
    """`methods.<m>.*` 应该长成哪个前缀。默认 = 方法名大写。

    为什么需要覆盖：**方法名 ≠ 环境变量前缀**。`hypergraphrag` 的旋钮叫 `HGR_*`、
    `ircot` 的叫 `IRCOT_*`。前缀对不上时那些键会被判为"污染全局命名空间"而
    **跳过并警告**（`config_to_env` 的行为），于是配置文件里写了一堆看似合理的键、
    实际一个都没生效。在配置里显式声明即可：

    ```jsonc
    { "method_prefixes": {"hypergraphrag": "HGR"},
      "methods": {"hypergraphrag": {"tokenizer": "tiktoken"}} }   // → HGR_TOKENIZER
    ```
    """
    mp = (cfg or {}).get("method_prefixes") or {}
    if isinstance(mp, dict) and method in mp:
        return str(mp[method]).upper().replace("-", "_")
    return method.upper().replace("-", "_")


def load_config_file(path) -> dict:
    """读 JSON（或 YAML，装了 pyyaml 的话）。文件不存在返回 {} 并给出提示路径。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"配置文件不存在：{p}\n  → 现有配置：{[x.name for x in config_path('.').parent.glob('*.json')]}")
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml                                     # type: ignore
            return yaml.safe_load(text) or {}
        except Exception as e:                              # noqa: BLE001
            raise RuntimeError(f"读 YAML 配置失败（{e}）；也可以直接用 JSON") from e
    return json.loads(text)


def _flatten(d: dict, prefix: str = "") -> dict:
    """嵌套 dict → 扁平 key：**先按原样拼接**，最后整体 `.upper()`。

    不能在递归里 upper：那样 `methods.proprag.top_docs` 会变成 `METHODS_PROPRAG_TOP_DOCS`，
    而方法私有键的判据是「以 `PROPRAG_` 开头」。整体 upper 后调用方再判前缀即可。

    `_` 开头的键（任意层级）都是**注释键**，直接跳过：`"_note": "..."` 曾经会被当成
    真参数注入成 `XXX__NOTE`（多一道下划线），白白往环境里塞垃圾。
    """
    out = {}
    for k, v in (d or {}).items():
        if str(k).startswith("_"):
            continue
        key = f"{prefix}_{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def config_to_env(cfg: dict, method: str = "") -> dict:
    """配置文件 → {ENV_NAME: str}（只做键映射，不落 os.environ）。

    - 顶层键（与 `defaults.*`）一律注入；
    - `methods.<m>.*` **只在 method 匹配时**注入，且要求扁平成的大写名以
      `<M>_` 开头（如 `methods.proprag.top_docs` → `PROPRAG_TOP_DOCS`），
      否则说明这个键会污染全局命名空间 —— 跳过并收集到 `skipped` 里警告。
    - 真实环境变量已存在的键**不覆盖**（尊重 `XXX=1 python run.py` 这种临时覆盖）。
    """
    if not isinstance(cfg, dict):
        return {}
    env: dict = {}
    for key, val in cfg.items():
        if str(key).startswith("_"):        # `_readme` / `_note_*`：配置文件的注释键
            continue
        if key in _NON_ENV_KEYS:
            # `defaults.*` 是 run 级默认值（workers/sample/...），**不注入 env**：
            # 注入会变成全局 `WORKERS`/`SAMPLE`，既没人读又污染环境。
            continue
        if key == "methods":
            if method and isinstance(val, dict) and method in val:
                prefix = method_prefix_for(cfg, method)
                want_prefix = prefix + "_"
                # 关键：把**方法名替换成真正的环境变量前缀**再扁平化。
                # 例如 hypergraphrag + prefix="HGR" → {"HGR": {"chunk_tokens": 1200}}
                # → HGR_CHUNK_TOKENS。直接 `_flatten({method: ...})` 只能得到
                # HYPERGRAPHRAG_XXX，前缀判据必然不过（写了一堆键却全被跳过）。
                for mk, mv in _flatten({prefix: val[method]}).items():
                    mk = mk.upper().replace("-", "_")
                    if not mk.startswith(want_prefix):
                        print(f"[config] ⚠ 跳过 methods.{method}.{mk}：扁平名不以 "
                              f"{want_prefix} 开头，会污染全局命名空间"
                              f"（方法私有键必须长成 {want_prefix}XXX）", flush=True)
                        continue
                    env[mk] = mv
            continue
        if isinstance(val, dict):
            env.update({k.upper(): v for k, v in _flatten(val).items()})
        else:
            env[str(key).upper()] = val

    out = {}
    for k, v in env.items():
        if v is None:
            continue
        if isinstance(v, bool):
            s = "1" if v else "0"
        elif isinstance(v, (list, tuple)):
            s = ",".join(str(x) for x in v)
        else:
            s = str(v)
        out[k] = s
    return out


def _assert_not_frozen(source: str) -> None:
    """配置文件必须在 `import config` **之前**注入，否则注入静默失效。

    为什么值得一次硬检查：`config.py` 与各 adapter 的常量都是「import 时读一次
    os.environ」。若某个模块在被注入之前就把 `config` 拉进来了，注入就成了一场空——
    参数还是默认值，**不报错、不告警**，只有跑完看结果才发现口径不对。
    （真实踩过：`common/paths.py` 顶层 `from config import ROOT`，而
    `run.py --config` 的注入路径正好经过 `common.paths`。）
    """
    if "config" in sys.modules:
        raise RuntimeError(
            f"配置文件注入太晚：`config` 已经被 import（触发点：{source}）。\n"
            f"  config.py 的常量在 import 时读一次 os.environ，之后再注入 os.environ 无效，\n"
            f"  参数会静默停在各代码默认值上。\n"
            f"  修法：确保 apply_config_file() 在任何 `import config` 之前调用——\n"
            f"  入口脚本自身只能 import stdlib + common.config_file，不能先 import\n"
            f"  common.paths / common.llm / methods.* 等会间接拉起 config 的模块。")


def apply_config_file(path, method: str = "", verbose: bool = True) -> dict:
    """把配置文件注入 os.environ（**已存在的不覆盖**）。返回 {已设置 / 已存在的键}。"""
    _assert_not_frozen(rel_to_root(path))
    cfg = load_config_file(path)
    wanted = config_to_env(cfg, method=method)
    set_keys, kept = {}, []
    for k, v in wanted.items():
        if k in os.environ:                 # 真实环境变量优先（用户显式 export 的更权威）
            kept.append(k)
            continue
        os.environ[k] = v
        set_keys[k] = v
    if verbose:
        print(f"[config] {rel_to_root(path)}"
              + (f"（method={method}）" if method else "")
              + f" → 注入 {len(set_keys)} 个环境变量"
              + (f"；{len(kept)} 个已由环境变量给定、不覆盖：{kept}" if kept else ""),
              flush=True)
        for k, v in sorted(set_keys.items()):
            print(f"[config]   {k}={v}", flush=True)
    return {"set": set_keys, "kept_env": kept, "file": rel_to_root(path), "raw": cfg}


def config_snapshot(path=None, method: str = "") -> dict:
    """读配置原文（写进 build/config.json 与 trace 头，回答"这次用的哪份配置"）。"""
    out = {"config_file": rel_to_root(path) if path else None, "method": method}
    if path and Path(path).exists():
        try:
            out["values"] = load_config_file(path)
        except Exception as e:                              # noqa: BLE001
            out["error"] = f"{type(e).__name__}: {e}"
    return out
