"""Standard RAG 基线：分块（滑动窗口）+ **纯稠密**检索（无 BM25、无 RRF），取 top-k。

分块粒度与建图窗口一致（EXTRACT_CHUNK_TOKENS / EXTRACT_CHUNK_OVERLAP），
保证与其他方法在「语料单元」口径上可比。
"""

import math
import os
import re

import numpy as np

from common.llm import embed as llm_embed, embed_batch, phase, split_by_tokens
from methods.base import BaseMethod, generate_answer
from config import EXTRACT_CHUNK_TOKENS, EXTRACT_CHUNK_OVERLAP, dataset_chunking

# ── standardrag 的检索口径 ──
# **纯稠密**：关掉 BM25、关掉 RRF 融合，直接按余弦相似度取 top-k。
# k 用独立旋钮：standardrag 的检索口径要能单独调，不被别处的默认值牵动。
# （原先这里解释成"不跟 `SEARCH_TOP_K_CHUNKS`"——那个旋钮**已删除**，因为生产路径
#  无人消费它，只让人把它误读成生成预算。）
STANDARDRAG_TOP_K = int(os.environ.get("STANDARDRAG_TOP_K", "5"))

# ── BM25 英文分析器 / 打分参数（全局唯一一份，5 个方法共用同一实现）──
# 两处官方口径**都是**「英文词干 + 英文停用词 + 去所有格」：
#   · S2G-RAG：Lucene `EnglishAnalyzer`（bm25_search/bm25_init_hotpotqa.py:8）
#   · IRCoT  ：ES `analyzer: "english"`（retriever_server/build_index.py:247-255）
# 我们原来只有 `re.findall(r"\w+", text.lower())` —— 无词干、无停用词，与两边**都不一致**：
# `runs`/`running` 命中不了 `run`；停用词还会稀释 IDF、把长段落的分数摊薄。
# Lucene/ES 的 EnglishAnalyzer = StandardTokenizer + 去 's + LowerCase
#                              + StopFilter(ENGLISH_STOP_WORDS_SET) + PorterStemFilter
BM25_ANALYZER = os.environ.get("BM25_ANALYZER", "1") == "1"
BM25_K1 = float(os.environ.get("BM25_K1", "1.5"))
BM25_B = float(os.environ.get("BM25_B", "0.75"))

# Lucene `EnglishAnalyzer.ENGLISH_STOP_WORDS_SET`（33 词）。**照抄官方词表**，
# 不用 nltk 的停用词表 —— 那张表内容不同（且需额外下载语料），会让口径再偏一次。
_LUCENE_EN_STOP = frozenset((
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in",
    "into", "is", "it", "no", "not", "of", "on", "or", "such", "that", "the",
    "their", "then", "there", "these", "they", "this", "to", "was", "will", "with",
))

_STEMMER = None
_STEM_TRIED = False


def _stem_fn():
    """Porter 词干器（snowballstemmer 优先，退到 nltk）；都装不上则返回 None（不词干）。"""
    global _STEMMER, _STEM_TRIED
    if not _STEM_TRIED:
        _STEM_TRIED = True
        try:
            import snowballstemmer
            _eng = snowballstemmer.stemmer("english")
            _STEMMER = lambda toks: _eng.stemWords(list(toks))     # list -> list
        except Exception:
            try:
                from nltk.stem import SnowballStemmer
                _s = SnowballStemmer("english")
                _STEMMER = lambda toks: [_s.stem(t) for t in toks]
            except Exception:
                _STEMMER = False
    return _STEMMER or None


def _tokenize(text: str) -> list[str]:
    """English analyzer（对齐官方）：小写 → 去所有格 → 切词 → 去停用词 → Porter 词干。

    `BM25_ANALYZER=0` 回到旧行为（纯 `\\w+` 小写）—— 仅用于与历史 run 对账。
    """
    if not BM25_ANALYZER:
        return re.findall(r"\w+", text.lower())
    text = re.sub(r"'s\b", "", text.lower())        # Lucene EnglishPossessiveFilter
    toks = [t for t in re.findall(r"\w+", text) if t not in _LUCENE_EN_STOP]
    stem = _stem_fn()
    return stem(toks) if stem else toks


def build_chunks(contexts: list[dict], chunk_tokens: int | None = None,
                 overlap: int | None = None, dataset: str = "") -> list[tuple]:
    """分块：按 token 切块（尺子 = embedding 模型同一 tokenizer，与建图同粒度）。

    ★ 默认**不切块**（`EXTRACT_CHUNK_TOKENS=0` → 归一成 `CHUNK_NO_SPLIT=2000`、overlap 0）
      = 一个 context 一个整段块 —— 全项目语料单元口径，见 `config.py:48-62`。
      要回退旧的 400/40 就设 `EXTRACT_CHUNK_TOKENS=400 EXTRACT_CHUNK_OVERLAP=40`。
    ★ **`dataset` 非空时以它为准**（`config.dataset_chunking`）：triviaqa 的 passage 是整篇
      维基文章，要切成 1200/100，其余数据集不切块。**切块的地方必须把 dataset 传进来**，
      否则 triviaqa 会静默退回 2000/0，与已建好的图对不上。
      显式传 `chunk_tokens` / `overlap` 时以显式值为准（工具/消融用）。

    返回 [(title, start_char, chunk_text, sent_locs)]，
    sent_locs = 与块字符区间相交的句子 (title, sent_idx)。

    ★ 索引文本 = `title\\ntext`。**这是官方口径**，三条独立证据：
      · 官方 S2G-RAG：`contents = f"{title} {text}"`
        （baselines/S2G-RAG-main/bm25_search/fullwiki_resources.py:49-51）
      · 官方 IRCoT：`document_type="title_paragraph_text"`，title 与正文双字段检索
        （baselines/ircot-main/retriever_server/elasticsearch_retriever.py:49-52）
      · 本项目建图方法一直这么做（hipporag_adapter.py:1000、hypergraphrag_adapter.py:338、
        lightrag_adapter.py:450、proprag_adapter.py:585）
    不拼的后果是系统性的：S2G-RAG/IRCoT 下一轮 query 是桥实体/属性短语，这些词高度集中
    在**标题**上，官方能在 title 上命中，我们不能。
    """
    chunks = []
    if chunk_tokens is None or overlap is None:
        _t, _o = dataset_chunking(dataset) if dataset else (EXTRACT_CHUNK_TOKENS, EXTRACT_CHUNK_OVERLAP)
        chunk_tokens = _t if chunk_tokens is None else chunk_tokens
        overlap = _o if overlap is None else overlap
    for ctx in contexts:
        title = (ctx.get("title") or "").strip()
        sents = ctx.get("sentences") or re.split(r"(?<=[.!?])\s+", ctx.get("text", ""))
        sents = [s.strip() for s in sents if s.strip()]
        if not sents:
            continue
        # 索引文本 = 标题 + 正文。spans 必须按这个整体算偏移（标题占 len(head) 字符），
        # 否则切块后「与块相交的句子」会错位、evidence_locs 全歪。
        head = f"{title}\n" if title else ""
        text = head + " ".join(sents)
        spans = []
        pos = len(head)
        for s in sents:
            spans.append((pos, pos + len(s)))
            pos += len(s) + 1
        for chunk_text, cs, ce in split_by_tokens(text, chunk_tokens, overlap):
            if len(chunk_text) >= 2:
                locs = [(title, i) for i, (s0, s1) in enumerate(spans)
                        if s1 > cs and s0 < ce]
                chunks.append((title, cs, chunk_text, locs))
    return chunks


def _bm25(query: str, docs: list[list[str]], k1=None, b=None) -> list[float]:
    """BM25。`k1`/`b` 省略时用全局旋钮 `BM25_K1` / `BM25_B`（默认 1.5/0.75）。

    官方各方法取值不同（S2G-RAG 0.9/0.4；IRCoT 走 ES 内建默认），所以保留按方法
    传值的入口；不传就是全局口径。
    """
    k1 = BM25_K1 if k1 is None else k1
    b = BM25_B if b is None else b
    N = len(docs)
    if N == 0:
        return []
    avgdl = sum(len(d) for d in docs) / N or 1.0
    df = {}
    for d in docs:
        for t in set(d):
            df[t] = df.get(t, 0) + 1
    q_terms = set(_tokenize(query))
    scores = []
    for d in docs:
        tf = {}
        for t in d:
            tf[t] = tf.get(t, 0) + 1
        s = 0.0
        for t in q_terms:
            if t in tf:
                idf = math.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * (tf[t] * (k1 + 1)) / (tf[t] + k1 * (1 - b + b * len(d) / avgdl))
        scores.append(s)
    return scores


def query_chunks(question: str, chunks: list, k: int = STANDARDRAG_TOP_K, vecs=None,
                 faiss_index=None, dense_only=False, dataset: str = "") -> dict:
    """chunks: build_chunks 的输出。dense+BM25 RRF 检索 + 生成。

    `k` 默认 **`STANDARDRAG_TOP_K`(=5)**，与"生成只喂检索到的那几块"一致。
    ★ 用**独立旋钮**而不是跟检索侧候选池：把"检索深度"和"喂给模型的证据数"混成一个
    旋钮的话，改检索深度就静默改了生成证据量。`SEARCH_TOP_K_CHUNKS` 那个旧旋钮**已删除**
    （生产路径无人消费）。
    （现有调用方 `run.py:549` / `StandardRAG.run()` / `smoke_test.py` 都显式传 `k`，
    所以改这个默认值对既有结果无影响。）

    vecs：预计算的分块向量（语料级 embed 一次、逐题复用，避免每题重嵌）。
    faiss_index：预建的 IndexFlatIP（有则走 C++ 内积检索，否则 numpy 暴力余弦）。
    dense_only：True 时关掉 BM25，纯 dense 按余弦取 top-k（「chunk dense」基线，
    与「命题 dense」只在检索单元上不同，其余同口径）。
    """
    with phase("query"):
        texts = [c[2] for c in chunks]
        if vecs is None:
            vecs = embed_batch(texts) if texts else np.zeros((0, 1024))
        qv = llm_embed(question)

        if faiss_index is not None:
            qv32 = np.ascontiguousarray(qv, dtype=np.float32).reshape(1, -1)
            _sims, _ids = faiss_index.search(qv32, len(chunks))
            dense = np.zeros(len(chunks), dtype=np.float32)
            dense[_ids[0]] = _sims[0]
        else:
            dense = np.array([float(np.dot(qv, v) / (np.linalg.norm(qv) * np.linalg.norm(v) + 1e-9))
                              for v in vecs])

        k = min(k, len(chunks))
        if dense_only:
            # 纯 dense：直接按余弦相似度取 top-k，无 BM25、无 RRF 融合
            top = [(float(dense[i]), i) for i in np.argsort(-np.array(dense))[:k]]
        else:
            sparse = _bm25(question, [_tokenize(t) for t in texts])
            dense_rank = {i: r for r, i in enumerate(np.argsort(-np.array(dense))[:len(chunks)])}
            sparse_rank = {i: r for r, i in enumerate(np.argsort(-np.array(sparse))[:len(chunks)])}
            fused = []
            for i in range(len(chunks)):
                dr = dense_rank.get(i, len(chunks))
                sr = sparse_rank.get(i, len(chunks))
                fused.append((1.0 / (60 + dr) + 1.0 / (60 + sr), i))
            fused.sort(reverse=True)
            top = fused[:k]

        evidence = [{"text": texts[i]} for _, i in top]
        answer = generate_answer(question, evidence, dataset=dataset)
        locs = []
        for _, i in top:
            locs.extend(chunks[i][3])
    return {"answer": answer,
            "evidence": [e["text"] for e in evidence],
            "evidence_locs": locs}


class StandardRAG(BaseMethod):
    name = "standardrag"

    def run(self, question, contexts):
        # ⚠️ 这不是 `run.py` 的路径：那边 standardrag 走的是**语料级** `chunks`
        # （`run.py:350-352` 用 `build_chunks(merged_contexts(...))` 建一次，再
        # `run.py:478-480` 用 `query_chunks(..., vecs, faiss_index, dense_only=True)` 逐题查）。
        # 这里只按传入的 contexts 现场分块，供 `smoke_new_baselines.py` 等单元测试用。
        chunks = build_chunks(contexts, dataset=getattr(self, "dataset", ""))
        # 纯稠密 top-k：关 BM25、关 RRF（口径见文件头与 STANDARDRAG_TOP_K 注释）
        return query_chunks(question, chunks, STANDARDRAG_TOP_K, dense_only=True,
                            dataset=getattr(self, "dataset", ""))
