"""检索工具：FAISS 向量索引。

★ 本文件曾包含整套**旧检索设计**（`search_graph` / `search_graph_ppr` / `plan` / `replan`
+ 4 维加权重排 + 实体/时间/地域约束匹配 + BM25 混合），**已于 2025 审查删除**：

  · 生产路径**从未调用**它们（`run.py` 与 `methods/corpus_index.py` 只 import
    `build_faiss_index`）；只有 `_backup_scripts/test_ppr.py` 用过，而那个脚本已随本次
    清理一起删除。
  · 那条链的配置常量（`config.DEFAULT_WEIGHTS` / `ANSWER_TYPE_WEIGHT` / `ENABLE_HYBRID` /
    `HYBRID_BM25_WEIGHT` / `CONFIDENCE_THRESHOLD` / `MAX_ROUNDS`）也一并删除，见 config.py。
  · 现在各方法的检索都在自己的适配器里：standardrag 纯稠密、factalign 命题 dense 拆解、
    ircot/selfask 用 `standard_rag._bm25`、s2grag 用 `corpus_index` 的稠密+BM25。

保留的唯一函数 `build_faiss_index` 被 3 处使用：
  `run.py`（standardrag 的语料级索引）、`methods/corpus_index.py`（FAISS）、
  `_backup_scripts/smoke_test.py`（旧冒烟）。
"""

import numpy as np


def build_faiss_index(vecs: np.ndarray):
    """faiss IndexFlatIP 向量索引（向量已归一化，内积即余弦）。

    Python 循环暴力余弦 → C++/SIMD 内积检索，语料级每查询从百 ms 级降到 ~10ms。
    语料级建一次、各题复用；不传则回退 numpy 暴力余弦（对齐 hotpotqa）。
    """
    import faiss
    arr = np.ascontiguousarray(vecs, dtype=np.float32)
    idx = faiss.IndexFlatIP(arr.shape[1])
    idx.add(arr)
    return idx
