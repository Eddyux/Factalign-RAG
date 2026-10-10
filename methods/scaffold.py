"""新注册 baseline 的脚手架：源码未接入时的统一「安全熔断」。

新注册的 4 个方法（unirag / adaptive_rag / raptor / adarague）在官方
源码接入前**必须拒绝出数**。半实现的基线一旦跑出数字被当成「真实基线结果」写进
论文表格，比跑不出来危险得多（已知坑 #6：检索饱和/2-3 点差值可能不显著，
但伪造基线数字是不可挽回的错误）。

因此所有「等官方源码」的步骤统一抛 SourceNotReady：
- run.py 在 build() 阶段捕获 → 打印 `[方法] 不可用：<原因>` → 直接返回，不写任何结果行
- 错误信息里带「哪个文件、哪个放置区、缺什么」，填好即自动生效

用法（适配器内）::

    from methods.scaffold import SourceNotReady, require_source

    UNIRAG_DECOMPOSE_PROMPT = None      # ★ 官方源码放置区
    ...
    prompt = require_source(UNIRAG_DECOMPOSE_PROMPT, self.name,
                           "entity-grounded decompose prompt",
                           "methods/unirag_adapter.py::UNIRAG_DECOMPOSE_PROMPT")
"""


class SourceNotReady(RuntimeError):
    """官方源码 / 官方 prompt 尚未接入。

    与「依赖没装」「vLLM 挂了」区分开：这是「人还没把源码放进来」，
    不是环境故障，重跑多少次都不会变。
    """


def require_source(value, method: str, what: str, where: str):
    """value 为空 → 抛 SourceNotReady；否则原样返回（便于链式赋值）。"""
    if value is None or (isinstance(value, str) and not value.strip()):
        raise SourceNotReady(
            f"[{method}] 官方源码未接入：缺 {what}\n"
            f"  → 请在 {where} 的「★ 官方源码放置区」填入官方实现/prompt 后重跑\n"
            f"  → 判断依据：本方法未接线前不产出任何结果行（避免半实现基线污染表格）"
        )
    return value


def require_path(path, method: str, what: str, hint: str = ""):
    """外部仓库/权重文件缺失 → 抛 SourceNotReady（带 clone 命令提示）。"""
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        raise SourceNotReady(
            f"[{method}] 官方源码未就位：找不到 {what}\n  期望路径: {p}\n"
            + (f"  {hint}\n" if hint else "")
        )
    return p


def find_repo(candidates, min_py: int = 1):
    """在候选目录里找「真的 clone 过」的仓库（至少含 min_py 个 .py 文件）。

    为什么不能只看目录存在：仓库未 clone 时，我们会在 baselines/<名字>/ 放一个
    「源码放置说明」README——只判目录存在会把这种占位目录误判成「已就位」，
    熔断就失效了。真 clone 的仓库必然有 .py。
    """
    from pathlib import Path
    for c in candidates:
        p = Path(c)
        if p.is_dir() and len(list(p.rglob("*.py"))) >= min_py:
            return p
    return None


def require_repo(candidates, method: str, what: str, hint: str = ""):
    """同 find_repo，但找不到就抛 SourceNotReady。"""
    repo = find_repo(candidates)
    if repo is None:
        from pathlib import Path
        tried = "\n".join(f"    - {Path(c)}" for c in candidates)
        raise SourceNotReady(
            f"[{method}] 官方源码未就位：找不到 {what}（候选目录里没有 .py）\n"
            f"{tried}\n" + (f"  {hint}\n" if hint else "")
        )
    return repo
