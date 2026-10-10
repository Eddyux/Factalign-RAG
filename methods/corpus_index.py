"""语料级检索底座（新注册 baseline 共用：unirag / adaptive_rag / raptor / adarague）。

设计原则：**新增基线和老基线只在「机制」上不同，不在「检索底座」上不同。**

- 切块：`methods.standard_rag.build_chunks`（默认**不切块**：0 → CHUNK_NO_SPLIT=2000、overlap 0，
  bge tokenizer；旧口径 400/40 可用环境变量回退）
- 稀疏：`methods.standard_rag._bm25`（k1=1.5, b=0.75）
- 稠密：`common.llm.embed_batch`（passage 侧不加指令）/ `embed`（query 侧加 bge 指令）
- 融合：RRF `1/(60+rank)`，与 `standard_rag.query_chunks` 逐字一致

本文件**不改动任何既有方法**：standardrag / ircot 仍走各自原路径，
这里只是把同一套底座对象化，供新方法复用；mode="rrf" 且 levels=None 时，
检索结果与 `standardrag` 在该语料上的 chunk 召回等价（可用来做口径自检）。

在 IRCoT 上的口径差异（有意为之）：IRCoT 官方用纯 BM25（`mode="bm25"`），
本项目 IRCoT 适配器也如此，别顺手改成 rrf，否则基线数值会漂。

用法::

    idx = CorpusIndex.from_contexts(merged_contexts(dataset, sample), dataset=dataset)  # 叶子块（level 0）
    idx.add_unit(text="cluster summary", level=1, meta={"children": [...]})  # 树上其它层
    idx.build_dense()                                                 # 向量 + FAISS，只建一次
    hits = idx.search(query, k=10, mode="rrf")                        # → 单元下标
"""

import numpy as np

from common.llm import EMBED_DIM, embed as llm_embed, embed_batch, phase
from methods.standard_rag import build_chunks, _bm25, _tokenize
from search import build_faiss_index

RRF_K = 60  # standard_rag.query_chunks 里的 60，别改


class CorpusIndex:
    """语料级检索单元集合（叶子块 + 可选的树上节点），带 dense/BM25/RRF 三种打分。

    units: list[dict]，每个单元 = {"text", "title", "locs", "level", "meta"}
    - level=0      → 整段 passage（叶子，与 standardrag/ircot 同粒度）
    - level>=1     → 派生节点（RAPTOR 的簇摘要等）
    """

    def __init__(self, units=None):
        self.units: list[dict] = list(units or [])
        self.tokens: list[list[str]] = []
        self.vecs = None            # (N, EMBED_DIM) float32，已归一化
        self.faiss = None           # IndexFlatIP（不可用时回退 numpy 暴力余弦）
        self._tokens_ready = False

    # ── 构造 ──
    @classmethod
    def from_contexts(cls, contexts: list[dict], level: int = 0,
                      dataset: str = "") -> "CorpusIndex":
        """N 题上下文合成语料 → 按数据集口径切块（triviaqa 1200/100，其余不切块）。

        `dataset` 非空时以它为准（`config.dataset_chunking`）；不传则退回全局口径。
        """
        units = []
        for title, _start, text, locs in build_chunks(contexts, dataset=dataset):
            units.append({"text": text, "title": title, "locs": list(locs),
                          "level": level, "meta": {}})
        return cls(units)

    def add_units(self, texts, level: int = 1, metas=None, titles=None,
                  embed: bool = False) -> list[int]:
        """批量追加派生节点（摘要/结构化条目），返回下标列表。

        embed=True：**只嵌入新节点**并拼到已有向量后面（RAPTOR 逐层建树时不能
        把整语料重嵌一遍——那是 O(层数 × 语料) 的无谓开销）；
        embed=False：向量置空，之后必须 build_dense()（否则 search 会报错）。
        """
        texts = list(texts)
        metas = list(metas or [{} for _ in texts])
        titles = list(titles or ["" for _ in texts])
        start = len(self.units)
        for t, m, ti in zip(texts, metas, titles):
            self.units.append({"text": t, "title": ti, "locs": [],
                               "level": level, "meta": dict(m or {})})
        self._tokens_ready = False
        self.faiss = None
        if embed and texts:
            with phase("build"):
                new = embed_batch(texts)
            if self.vecs is not None and len(self.vecs) == start:
                self.vecs = np.vstack([self.vecs, new])
            else:
                self.vecs = new          # 长度对不上（没建过/被清空）→ 只留新块，需重建
        else:
            self.vecs = None
        return list(range(start, len(self.units)))

    def add_unit(self, text: str, title: str = "", locs=None, level: int = 1,
                 meta: dict | None = None, embed: bool = False) -> int:
        """追加单个派生节点。返回下标。"""
        idx = self.add_units([text], level=level, metas=[meta], titles=[title],
                             embed=embed)[0]
        if locs:
            self.units[idx]["locs"] = list(locs)
        return idx

    def invalidate(self):
        """单元集合变更后清派生缓存。"""
        self._tokens_ready = False
        self.vecs = None
        self.faiss = None

    def __len__(self) -> int:
        return len(self.units)

    def texts(self) -> list[str]:
        return [u["text"] for u in self.units]

    def by_level(self, level: int) -> list[int]:
        return [i for i, u in enumerate(self.units) if u["level"] == level]

    def levels(self) -> list[int]:
        return sorted({u["level"] for u in self.units})

    # ── 建向量索引（语料级一次，逐题复用）──
    def build_dense(self) -> "CorpusIndex":
        with phase("build"):
            if not self.units:
                self.vecs = np.zeros((0, EMBED_DIM), dtype=np.float32)
                self.faiss = None
                return self
            self.vecs = embed_batch([u["text"] for u in self.units])
        return self.rebuild_faiss()

    def rebuild_faiss(self) -> "CorpusIndex":
        """按当前 vecs 重建 FAISS 索引（vecs 为空则清空索引）。"""
        if self.vecs is None:
            self.faiss = None
            return self
        try:
            self.faiss = build_faiss_index(self.vecs)
        except Exception as e:          # faiss 未装 → 退 numpy 暴力余弦（慢但一致）
            print(f"[corpus_index] faiss 不可用（{type(e).__name__}: {e}），"
                  f"回退 numpy 暴力余弦", flush=True)
            self.faiss = None
        return self

    def _ensure_tokens(self):
        if not self._tokens_ready:
            self.tokens = [_tokenize(u["text"]) for u in self.units]
            self._tokens_ready = True

    # ── 打分（与 standard_rag 同实现）──
    def dense_scores(self, query: str) -> np.ndarray:
        if self.vecs is None:
            raise RuntimeError("CorpusIndex.build_dense() 未调用")
        qv = llm_embed(query)
        if self.faiss is not None:
            qv32 = np.ascontiguousarray(qv, dtype=np.float32).reshape(1, -1)
            sims, ids = self.faiss.search(qv32, len(self.units))
            dense = np.zeros(len(self.units), dtype=np.float32)
            dense[ids[0]] = sims[0]
            return dense
        return np.array([float(np.dot(qv, v) / (np.linalg.norm(qv) * np.linalg.norm(v) + 1e-9))
                         for v in self.vecs], dtype=np.float32)

    def sparse_scores(self, query: str, k1=None, b=None) -> list[float]:
        self._ensure_tokens()
        return _bm25(query, self.tokens, k1=k1, b=b)

    # ── 检索 ──
    def search(self, query: str, k: int, mode: str = "rrf", exclude=None,
               levels=None, bm25_k1=None, bm25_b=None) -> list[int]:
        """返回 top-k 单元下标。

        mode:
          "dense" → 纯稠密（standardrag 现在的口径）
          "bm25"  → 纯稀疏（IRCoT 官方口径，也是 Adaptive-RAG strategy C 的默认）
          "rrf"   → dense+BM25 的 RRF 融合（标准 RAG 的常见做法；standardrag 本轮已不用）
        levels: 限定只在这些 level 上检索（RAPTOR 的 collapsed tree = None(全层)，
                leaf-only = levels={0}）
        exclude: 跳过的单元下标（迭代检索里累积去重）
        bm25_k1 / bm25_b: 覆盖全局 BM25 打分参数（官方各方法取值不同，如 S2G-RAG
                是 0.9/0.4）；不传则用 `standard_rag.BM25_K1/BM25_B`。
        """
        n = len(self.units)
        if n == 0 or k <= 0:
            return []
        exclude = set(exclude or ())
        allowed = [i for i in range(n)
                   if i not in exclude and (levels is None or self.units[i]["level"] in levels)]
        if not allowed:
            return []

        if mode == "bm25":
            sparse = self.sparse_scores(query, k1=bm25_k1, b=bm25_b)
            order = sorted(allowed, key=lambda i: -sparse[i])
        elif mode == "dense":
            dense = self.dense_scores(query)
            order = sorted(allowed, key=lambda i: -dense[i])
        elif mode == "rrf":
            dense = self.dense_scores(query)
            sparse = self.sparse_scores(query, k1=bm25_k1, b=bm25_b)
            dense_rank = {i: r for r, i in enumerate(sorted(allowed, key=lambda i: -dense[i]))}
            sparse_rank = {i: r for r, i in enumerate(sorted(allowed, key=lambda i: -sparse[i]))}
            order = sorted(allowed, key=lambda i: -(1.0 / (RRF_K + dense_rank[i])
                                                    + 1.0 / (RRF_K + sparse_rank[i])))
        else:
            raise ValueError(f"未知 mode={mode!r}（可选 dense/bm25/rrf）")
        return order[:k]

    # ── 取证据（generate_answer 的统一入参）──
    def evidence(self, idxs) -> list[dict]:
        """只给 text：与 standardrag / ircot 喂给 generate_answer 的形态逐字一致
        （带上 source 会多出 "Source: ..." 行，生成 prompt 口径就不同了）。"""
        return [{"text": self.units[i]["text"]} for i in idxs]

    def locs(self, idxs) -> list:
        out = []
        for i in idxs:
            out.extend(self.units[i]["locs"])
        return out

    def window_texts(self, idxs) -> list[str]:
        return [self.units[i]["text"] for i in idxs]
