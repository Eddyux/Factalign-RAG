"""方法注册表（懒加载）。

为什么懒加载：adapter 文件可能还没同步到本机（例如 graphrag/unirag/... 的
*_adapter.py 不在），旧写法在 import 注册表时就 ModuleNotFoundError，导致 run.py
连 --help 都跑不起来。现在只有真正用到某个方法时才 import 它的 adapter；
缺文件时给出「本机少哪个文件」的明确报错，其余方法照常可用。

已移除（不再进表，adapter / baseline 一并删除）：
  graphrag / unirag / adaptive_rag / raptor / adarague / proh / structrag
  （structrag 已按要求删掉：adapter、6 个官方 prompt、上游仓库目录、配置项、
    全部文档与脚本引用一起移除，不再参与任何跑批）
源码接入前调用会被 methods.scaffold.SourceNotReady 拦下（run.py 只打印
「不可用」、不写结果行）：
  s2grag
"""

from collections.abc import Mapping

from methods.base import BaseMethod
from methods.standard_rag import StandardRAG
from methods.factalign import FactAlign

# 方法名 → (模块名, 类名)。只有被取用到的才会真正 import。
_REGISTRY: dict[str, tuple[str, str]] = {
    "standardrag": ("methods.standard_rag", "StandardRAG"),
    "factalign": ("methods.factalign", "FactAlign"),   # 本方法（HyperfactRAG 退役）
    "lightrag": ("methods.lightrag_adapter", "LightRAG"),
    "hypergraphrag": ("methods.hypergraphrag_adapter", "HyperGraphRAG"),
    "hipporag": ("methods.hipporag_adapter", "HippoRAG"),
    "ircot": ("methods.ircot_adapter", "IRCoT"),
    "zeroshot": ("methods.zeroshot", "NoRetrieval"),
    # ── 新注册基线 ──
    "s2grag": ("methods.s2grag_adapter", "S2GRAG"),
    "selfask": ("methods.selfask_adapter", "SelfAsk"),
    "proprag": ("methods.proprag_adapter", "PropRAG"),
}

# 已在本模块顶部直接 import 的（不需要再动态 import）
_EAGER = {"standardrag": StandardRAG, "factalign": FactAlign}


class AdapterMissing(RuntimeError):
    """adapter 源码文件不在本机（未同步），与「官方源码未就位」区分开。"""


def _load_class(name: str):
    """按需 import 并返回方法类（同一次进程内只 import 一次）。"""
    if name in _EAGER:
        return _EAGER[name]
    mod_name, cls_name = _REGISTRY[name]
    try:
        mod = __import__(mod_name, fromlist=[cls_name])
    except ModuleNotFoundError as e:
        raise AdapterMissing(
            f"[{name}] 本机缺少 adapter 文件：{mod_name.replace('.', '/')}.py（{e}）\n"
            f"  → 从有该文件的机器同步 methods/{mod_name.split('.')[-1]}.py 后重跑；\n"
            f"  → 本机当前 methods/ 里没有这个文件，不是环境依赖问题"
        ) from e
    return getattr(mod, cls_name)


class _MethodRegistry(Mapping):
    """`METHODS` 的懒加载视图。

    兼容三种老用法：`name in methods.METHODS`、`methods.METHODS[name]`、
    `list(methods.METHODS)`。取值时**返回类**（不是实例），与旧 `dict{name: 类}`
    语义一致；但只有真被取到的方法才 import 它的 adapter（缺文件的 adapter 不会
    连累 run.py 启动）。
    """

    def __init__(self, names):
        self._names = tuple(names)

    def __getitem__(self, name):
        if name not in _REGISTRY:
            raise KeyError(name)
        return _load_class(name)

    def __iter__(self):
        return iter(self._names)

    def __len__(self):
        return len(self._names)


METHODS = _MethodRegistry(_REGISTRY)


def get_method(name: str) -> BaseMethod:
    if name not in _REGISTRY:
        raise KeyError(f"未知方法 {name!r}，可选: {list(_REGISTRY)}")
    return _load_class(name)()


def make_method(name: str, **kwargs) -> BaseMethod:
    """带参数实例化（run.py 的 factalign 旋钮用）。"""
    if name == "factalign":
        return FactAlign(**kwargs)
    return get_method(name)
