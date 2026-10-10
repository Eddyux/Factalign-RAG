#!/usr/bin/env python
"""诊断 `<ERROR: The truth value of an array with more than one element is ambiguous>`（hipporag）。

## 背景

hipporag 每题 3 次重试都报同一个错（`run.py:590` 打印 `重试 N/3: <e>`），500 题全退化成
`<ERROR: ...>`。而 `run.py:579,587` 只记 `str(e)`、**不记 traceback** → 日志里只有那一句话。

已用形状推演收敛到**唯一机制**（`np` 实测推演，不是猜）：官方 `HippoRAG.py` 的

    phrase_weights  = np.zeros(len(self.graph.vs['name']))   # :2030 1D
    passage_weights = np.zeros(len(self.graph.vs['name']))   # :2031 1D
    node_weights = phrase_weights + passage_weights          # :2091 本应 1D
    if sum(node_weights) <= 0:                               # :2097 ← 抛 ambiguous

**只有 `node_weights` 变成 2D 时**，`sum()` 才返回多元素数组、这一行才报 ambiguous。
即：某处把**数组**赋进了 1D 权重数组的单个槽位。嫌疑（官方 `HippoRAG.py`）：

  · `:2040-2041`  `fact_score = query_fact_scores[top_k_fact_indices[rank]] if ndim>0 else ...`
  · `:2084-2086`  `passage_dpr_score = normalized_dpr_sorted_scores[i]` → `passage_weights[...] = ...`
  · `:2051/2054`  `weighted_fact_score = fact_score` / `weighted_fact_score /= len(...)`

三者都要求右值是**标量**；一旦我们注入的 embedding（`_BgeEmbedding.batch_encode`）或
DPR 重排返回 `(1, ...)` / `(k, ...)` 形状，广播就把权重数组撑成 2D。

## 本脚本做什么（**不改项目任何文件**）

1. 用**已建好的图**跑前 N 题（默认 3），把每次异常转成**完整 traceback**；
2. 逐帧打印局部变量里 **numpy 数组的 shape**（`node_weights` / `phrase_weights` /
   `query_fact_scores` / `fact_score` … 哪个是多维的一眼可见）；
3. 先探测我们注入的 embedding 对**单条**查询返回的形状（最容易被忽略的根因）。

## 用法（服务器上，项目根目录）

    python tools/diag_hipporag_ambiguous.py --dataset hotpotqa --sample 500 --limit 3

把输出（尤其是 `★ 出错帧` 那几段）反馈即可定位。
"""

from __future__ import annotations

import argparse
import sys
import traceback
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _shape_of(v) -> str:
    try:
        import numpy as np
        if isinstance(v, np.ndarray):
            return f"ndarray{v.shape} {v.dtype}"
        if isinstance(v, (np.floating, np.integer)):
            return f"numpy 标量 {type(v).__name__}={v}"
        if isinstance(v, (list, tuple)):
            import numpy as np2
            try:
                return f"{type(v).__name__} len={len(v)} → np.asarray{v} of {np2.asarray(v).shape}"
            except Exception:                                    # noqa: BLE001
                return f"{type(v).__name__} len={len(v)}"
    except Exception:                                            # noqa: BLE001
        pass
    return type(v).__name__


def _dump_frames(tb: types.TracebackType) -> None:
    """逐帧打印：文件:行、源码行、以及局部变量里的 numpy 形状。"""
    import numpy as np
    frames = []
    while tb is not None:
        frames.append(tb)
        tb = tb.tb_next
    print(f"   ★ 出错帧共 {len(frames)} 层；下面只看**含 numpy 数组**的帧：", flush=True)
    for fr in frames:
        code = fr.tb_frame.f_code
        fname = Path(code.co_filename).name
        lineno = fr.tb_lineno
        try:
            import linecache
            src = (linecache.getline(code.co_filename, lineno) or "").strip()
        except Exception:                                        # noqa: BLE001
            src = ""
        arrs = {}
        for k, v in fr.tb_frame.f_locals.items():
            if isinstance(v, np.ndarray):
                arrs[k] = f"ndarray{v.shape} {v.dtype}"
        # 只打印「含多维数组」的帧，或 hipporag/methods 里的帧（其余是无关的库帧）
        has_multi = any("ndarray(" in d and "," in d.split("ndarray(")[1].split(")")[0]
                        for d in arrs.values())
        if not has_multi and "hipporag" not in code.co_filename \
                and "methods" not in code.co_filename:
            continue
        print(f"     {fname}:{lineno}  {src[:90]}", flush=True)
        for k, desc in sorted(arrs.items()):
            inner = desc.split("ndarray(")[1].split(")")[0] if "ndarray(" in desc else ""
            multi = "," in inner
            print(f"         {k} = {desc}{'   ← **多维数组**' if multi else ''}", flush=True)


def _check_embedding_shape() -> None:
    import numpy as np
    import common.llm as L
    for name, call in (("embed('…')", lambda: L.embed("probe query")),
                       ("embed_batch(['a','b'])", lambda: L.embed_batch(["a", "b"]))):
        try:
            v = np.asarray(call())
            print(f"[diag] common.llm.{name:26s} → shape={v.shape} dtype={v.dtype}", flush=True)
        except Exception as e:                                   # noqa: BLE001
            print(f"[diag] ⚠ {name} 探测失败：{type(e).__name__}: {e}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="诊断 hipporag 的 ambiguous 数组错误")
    ap.add_argument("--dataset", default="hotpotqa")
    ap.add_argument("--sample", type=int, default=500)
    ap.add_argument("--graph-version", type=int, default=1)
    ap.add_argument("--limit", type=int, default=3, help="只跑前 N 题（默认 3）")
    ap.add_argument("--no-build", action="store_true",
                    help="跳过 build()（只适合 _rag 已在内存里的场景）；正常别加")
    args = ap.parse_args()

    print("══ hipporag ambiguous 诊断 ══", flush=True)
    import numpy as np
    print(f"[diag] numpy = {np.__version__} | python = {sys.version.split()[0]}", flush=True)
    _check_embedding_shape()

    from common.data import iter_questions
    from methods import get_method

    try:
        m = get_method("hipporag")
    except Exception as e:                                       # noqa: BLE001
        print(f"✗ 加载适配器失败：{type(e).__name__}: {e}", flush=True)
        return 1
    # ★ 复刻 run.py 的调用顺序（适配器**没有** prepare_for）：
    #   run.py:437-438 设 dataset → run.py:454 build(merged_contexts(...), working_dir=graph_dir)
    #   图目录已存在时 build() 内部会**校验完整性并直接加载**（不重建），所以这里安全。
    if hasattr(m, "dataset"):
        m.dataset = args.dataset
    if not args.no_build:
        try:
            from common.result import graph_dir
            from corpus import merged_contexts
            gr_dir = str(graph_dir("hipporag", args.dataset, args.graph_version))
            print(f"[diag] build() 载入语料（图目录 {gr_dir}）…", flush=True)
            m.build(merged_contexts(args.dataset, args.sample), working_dir=gr_dir)
            print("[diag] build() 返回（图存在则只是加载）", flush=True)
        except Exception as e:                                   # noqa: BLE001
            print(f"[diag] ⚠ build() 失败（{type(e).__name__}: {e}）"
                  f"\n[diag] 继续尝试 query；若 _rag 为空则加 --no-build 并先用 run.py 建好图",
                  flush=True)
            traceback.print_exc()

    rag = getattr(m, "_rag", None)
    if rag is not None:
        try:
            print(f"[diag] graph 节点数 = {len(rag.graph.vs['name'])}", flush=True)
            for attr in ("fact_embeddings", "passage_embeddings"):
                v = getattr(rag, attr, None)
                if v is not None:
                    print(f"[diag] {attr} shape={np.asarray(v).shape}", flush=True)
        except Exception as e:                                   # noqa: BLE001
            print(f"[diag] ⚠ 形状探测失败：{type(e).__name__}: {e}", flush=True)

    items = list(iter_questions(args.dataset, args.sample))[:max(1, args.limit)]
    bad = 0
    for i, it in enumerate(items, 1):
        print(f"\n── 第 {i}/{len(items)} 题 {it.get('id')}：{str(it.get('question'))[:70]}",
              flush=True)
        try:
            r = m.query(it["question"])
            print(f"   ✓ 通过：answer={str(r.get('answer'))[:60]!r} "
                  f"evidence={len(r.get('evidence') or [])}", flush=True)
        except Exception as e:                                   # noqa: BLE001
            bad += 1
            print(f"   ✗ {type(e).__name__}: {e}", flush=True)
            tb = sys.exc_info()[2]
            _dump_frames(tb)
            print("   ── 原始 traceback ──", flush=True)
            traceback.print_exc()
    print(f"\n══ 结束：{len(items)} 题里 {bad} 题抛异常 ══", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
