"""FactAlign-RAG：命题（自包含事实句）索引 + 证据条件化的开放槽命题查询。

方法主线（EXPERIMENT_PLAN.md §〇）：
    多跳检索中，第 t+1 跳的**查询形态与索引单元形态不一致**是第二跳失败的一阶原因。
    把语料索引为自包含命题（命题 = 一句完整事实 + 原始窗口），并把下一跳需求表达成
    **同形态的开放槽命题**（`"Cleopatra IV's father is ___"`），使下一跳检索退化为
    对单条事实的近复述匹配。

本文件是**语料级方法**实现（run.py 的 `--method factalign`），**只含主方法口径**
（`q_stmt_he` = 开放槽命题 × 命题索引，见 EXPERIMENT_PLAN.md §三）：
    1. 召回：全量命题语义检索（dense）→ 命中命题进池
    2. 拆解：LLM 判定「池中是否已能直接回答」；不够 → 生成**开放槽命题** → 再检索
       （命题只进不出，命中分记 decomp_score）
    3. 精排：池内 dense / +实体描述侧面分（`--factalign-rerank`）
    4. 跨跳融合：`final = max(final, normalize(decomp_score))` —— 桥命题对原问题
       语义分低，靠子查询命中分兜底才进得了生成端
    5. 生成：top 命题**本身**（`FACTALIGN_EVIDENCE_PROPS`，默认 20 = 4 轮累积的全部）
       + 各自 window_text 去重后的**前 5 个**原文窗口（`FACTALIGN_EVIDENCE_WINDOWS`）
       → 编号连续 [1..25] → 统一 `generate_answer`（与其余基线同生成 prompt）

★ **消融不在本文件**（用户要求：消融放 `test_layer.py`）。原先这里的消融开关
  （`--no-decompose` 单轮召回、`--single-recall rrf/bm25/dense` 融合检索、
  `--query-form desc/inst`、`--decompose-recall chunk`）**已删除**；
  对应能力在 `test_layer.py` 里有各自独立的实现与 CLI（`--recall` 默认 rrf 等）。

用法（经 run.py）:
    python run.py --method factalign --dataset 2wiki --sample 300 --workers 16

消融请走实验台（同管线，带全部开关）:
    python test_layer.py --dataset 2wiki --sample 300 --decompose --decompose_stmt \
        --rerank hybrid --tag q_stmt_he

═══ 统一口径声明（与其它基线可比的前提，论文里要写）═══
  1. **切块 = 本项目语料单元**（`config.EXTRACT_CHUNK_TOKENS/OVERLAP`，默认**不切块**：
     `0` → 归一成 `CHUNK_NO_SPLIT=2000`、overlap 0 = 一个 context 一个整段块；旧口径为
     400 bge token / 40 重叠，可用环境变量回退），
     与 standardrag/ircot/外部基线同一把 bge 尺子；外部框架内部若自带 tokenizer，
     适配器会运行时替换成 bge，见各 adapter 顶部「与官方的差异清单」）。
  2. **生成**：统一 `methods.base.generate_answer`（`prompts/rag_prompt.txt`），
     证据 small-to-big 映射回原文窗口；与所有对比方法同生成口径。
  3. **命题索引**：`data/corpus/{dataset}/v{n}/`（entities/propositions/pvecs），
     由 `extract.py` 建图，build 只加载/重建该缓存；实体描述向量与窗口向量缓存同目录。
  4. **检索单元 = 命题（自包含命题）**，精排与跨跳融合的公式见上文步骤 3/4。
"""

import os

import numpy as np

from common.llm import chat as llm_chat, embed as llm_embed, embed_batch, phase
from methods.base import BaseMethod, generate_answer
from methods.standard_rag import build_chunks
from config import (EXTRACT_CHUNK_TOKENS, EXTRACT_CHUNK_OVERLAP, EMBED_DIM)

# ── 生成阶段的证据预算（**本项目设计，非官方参数**；对照表见 AUDIT_PARAMS.md §6）──
# 命题：4 轮拆解累积的**全部**（`decomp_top_k` 5 × `decomp_steps` 4 = 20）都发进 prompt ——
#       命题是蒸馏后的事实句，比原文更直接，原先被 `format_evidence` 整个丢掉只用原文窗口。
# 窗口：small-to-big 把命题映射回原文窗口，只发**前 5 个**。
# 0 = 不发该段。两者都只影响**生成 prompt** → 改完重跑逐题即可，**不用重建索引**。
FACTALIGN_EVIDENCE_PROPS = int(os.environ.get("FACTALIGN_EVIDENCE_PROPS", "20"))
FACTALIGN_EVIDENCE_WINDOWS = int(os.environ.get("FACTALIGN_EVIDENCE_WINDOWS", "5"))

# ══════════════════════════════════════════════════════════════════════════════
# 主方法口径常量（消融用的 desc/inst 两版 prompt 已随消融一起删除）
# ══════════════════════════════════════════════════════════════════════════════

# 拆解：判定「是否已能直接回答」+ 生成下一个开放槽命题（**主方法唯一使用的 prompt**）
# ★ 原 `NEXT_PROMPT`（实例化疑问句 inst）与 `NEXT_PROMPT_DESC`（描述式 desc）
#   是消融的两版查询形态，已随消融一起删除（对应消融在 test_layer.py 里有自己的副本）。
NEXT_SYSTEM = "You plan the next retrieval step for a multi-hop QA system."

# 主方法：开放槽命题（陈述式）——把下一跳写成「你希望检索到的那句事实」，未知处留空
NEXT_PROMPT_OPEN_SLOT = """Question: {question}

Evidence gathered so far:
{evidence}

Judge strictly:
1) Does any evidence fact DIRECTLY state the answer to the question? Only a fact that
   itself names the answer counts.
2) If NOT directly answered: the knowledge base stores SELF-CONTAINED FACT sentences, so
   write what you need to look up AS A FACT SENTENCE — the sentence you hope to find,
   with the unknown part left as ___. Reuse the names that appear in the evidence.
   Example: evidence says "X was the son of Antiochus IX and his wife Cleopatra IV" and
   you need Cleopatra IV's father → write "Cleopatra IV's father is ___".
   If the evidence does not name the intermediate entity, put a short description in its
   place, e.g. "the father of Antiochus X's mother is ___".

Return ONLY JSON: {{"direct_answer": false, "next_question": "Cleopatra IV's father is ___"}}"""

# 消融用：LLM 证据过滤（仿 HippoRAG recognition memory）
FILTER_SYSTEM = "You filter retrieved facts for a multi-hop QA system."
FILTER_PROMPT = """Question: {question}

Candidate facts:
{facts}

Select up to {keep} facts that are MOST USEFUL for answering the question above.
Judge each fact against the QUESTION (not against other facts):
- KEEP a fact if it contains the answer, or states a key relation needed for the
  reasoning chain (bridge facts, e.g. "X is the mother of Y"), or constrains the answer
  (time/place/role that the question asks about).
- DROP facts about unrelated entities or tangential topics, even if they share words
  with the question.

Order the kept facts from MOST to LEAST useful. Return ONLY JSON: {{"keep": [3, 1, 7]}}"""


def cosine(a, b) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _norm(x) -> np.ndarray:
    """max 归一化（全 0 时返回全 0）。"""
    x = np.asarray(x, dtype=np.float32)
    m = float(x.max()) if len(x) else 0.0
    return x / m if m > 0 else np.zeros_like(x)


def _tok_hit(a, b) -> bool:
    """token 全包含：一侧集合完全包含另一侧即命中（同 search._match_entity）。"""
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return False
    return len(sa & sb) >= min(len(sa), len(sb))


def evidence_filter(question: str, props: list[dict], keep_top: int = 0):
    """LLM 证据过滤：返回 (保留的命题列表, 说明)。失败/空结果回退原列表。"""
    if len(props) <= 1:
        return props, "too_few"
    _keep = keep_top if keep_top > 0 else len(props)
    lines = "\n".join(f"[{i+1}] {(h.get('summary') or '')[:200]}" for i, h in enumerate(props))
    out = llm_chat(FILTER_SYSTEM,
                   FILTER_PROMPT.format(question=question, facts=lines, keep=_keep),
                   json_mode=True, timeout=120)
    if not out:
        return props, "llm_fail"
    keep = out.get("keep")
    if not isinstance(keep, list):
        return props, "bad_json"
    idx = []
    for k in keep:
        try:
            i = int(k) - 1
        except (TypeError, ValueError):
            continue
        if 0 <= i < len(props) and i not in idx:
            idx.append(i)
    if not idx:
        return props, "empty_keep"
    if keep_top > 0:
        idx = idx[:keep_top]
    return [props[i] for i in idx], f"kept {len(idx)}/{len(props)}"


def ppr_pool(cand_list: list[dict], sem: np.ndarray, eid_name: dict,
             alpha: float = 0.5, iters: int = 20,
             seed_mask: np.ndarray | None = None) -> np.ndarray:
    """池内 PPR：在「共享实体」的命题-命题图上带重启游走（种子 = 语义分）。

    共享检测用实体名 token 而非 gid 精确相等 —— 实体分裂会断图边。
    """
    n = len(cand_list)
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    name_sets = []
    for h in cand_list:
        toks: set = set()
        for eid in h.get("entity_ids", []):
            toks.update(_tokenize(eid_name.get(eid, "")))
        name_sets.append(toks)
    P = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        if not name_sets[i]:
            continue
        for j in range(i + 1, n):
            if not name_sets[j]:
                continue
            ov = len(name_sets[i] & name_sets[j])
            if ov > 0:
                P[i, j] = P[j, i] = ov
    rs = P.sum(axis=1)
    for i in range(n):
        if rs[i] > 0:
            P[i, :] /= rs[i]
    seed = np.maximum(np.asarray(sem, dtype=np.float64), 0.0)
    if seed_mask is not None:
        seed = seed * seed_mask.astype(np.float64)
    ss = seed.sum()
    if ss <= 0:
        seed = np.maximum(np.asarray(sem, dtype=np.float64), 0.0)
        ss = seed.sum()
    if ss <= 0:
        seed = np.ones(n, dtype=np.float64) / n
    else:
        seed /= ss
    p = seed.copy()
    for _ in range(iters):
        p = (1.0 - alpha) * seed + alpha * (P.T @ p)
    s = p.sum()
    return (p / s).astype(np.float32) if s > 0 else p.astype(np.float32)


class FactAlign(BaseMethod):
    """FactAlign-RAG 语料级方法：build(语料) 一次 → 逐题 query。"""

    name = "factalign"

    # 主方法默认口径（EXPERIMENT_PLAN.md §三 第 5 格 q_stmt_he = 开放槽命题 × 命题索引）
    # ★ 消融开关已删（放 test_layer.py）：`query_form` / `recall`（含 **rrf** 融合检索）/
    #   `decompose` / `hyper_recall` / `recall_chunks` / `bm25_w`。
    #   注意 `rerank` / `hope` / `ppr_*` **不在**本次删除范围（用户明确只要删 rrf 那条融合），
    #   它们的取值语义见下：rerank=hybrid 是主口径；hope/ppr_w 默认 0 = 关闭。
    def __init__(self, decomp_top_k: int = 5, decomp_steps: int = 4,
                 rerank: str = "hybrid", gen_top: int = 0,
                 hope: int = 0, hop_k: int = 8, ppr_w: float = 0.0, ppr_alpha: float = 0.5,
                 evidence_filter_on: bool = False, filter_keep: int = 10,
                 filter_pre: int = 0, debug: int = 0):
        self.decomp_top_k = decomp_top_k
        self.decomp_steps = decomp_steps
        self.rerank = rerank
        self.gen_top = gen_top
        self.hope = hope
        self.hop_k = hop_k
        self.ppr_w = ppr_w
        self.ppr_alpha = ppr_alpha
        self.evidence_filter_on = evidence_filter_on
        self.filter_keep = filter_keep
        self.filter_pre = filter_pre
        self.debug = debug
        self.dataset = ""
        self._ready = False
        self.stats = {"n": 0, "decomp_steps_used": 0, "stop_reasons": {}}

    # ── 语料级准备 ──────────────────────────────────────────────────────────
    def build(self, corpus_contexts: list[dict], working_dir: str = "") -> "FactAlign":
        """一次建好：语料图（命题 + 窗口）+ 实体描述向量 + 窗口向量。

        working_dir = run.py 传的 data/{method}/{dataset}/v{n}。语料图本身仍在
        data/corpus/{dataset}/v{n}（与其余方法共用，只建一次）。
        """
        from corpus import get_or_build, _corpus_dir

        self._ctx = corpus_contexts
        with phase("build"):
            entities, propositions, chunks, pvecs = get_or_build(
                self.dataset, self._sample, self._graph_version)
            self.entities, self.propositions, self.chunks = entities, propositions, chunks
            self.pvecs = np.asarray(pvecs, dtype=np.float32)

            # 题面窗口（与建图同口径 = 整段）：生成端 small-to-big 用它
            self.windows = build_chunks(corpus_contexts, dataset=self.dataset)

            # 结构索引
            self.prop_by_id = {h["id"]: h for h in self.propositions}
            self.prop_idx = {h["id"]: i for i, h in enumerate(self.propositions)}
            self.ent_props: dict[str, set] = {}
            for h in self.propositions:
                for eid in h.get("entity_ids", []):
                    self.ent_props.setdefault(eid, set()).add(h["id"])

            # 实体描述向量（语料级一次，缓存复用）
            cdir = _corpus_dir(self.dataset, self._sample, self._graph_version)
            self.ent_vecs = self._load_or_build_ent_vecs(cdir)
            # 窗口向量（缓存复用）
            self.win_vecs = self._load_or_build_win_vecs(cdir)
            # 一跳扩展用的全库实体矩阵
            self.ent_list = list(self.ent_vecs.keys())
            self.ent_mat = (np.stack([self.ent_vecs[e] for e in self.ent_list])
                            if self.ent_list else np.zeros((0, EMBED_DIM), dtype=np.float32))
        self._ready = True
        print(f"[factalign] 命题 {len(self.propositions)} | 窗口 {len(self.windows)} | "
              f"实体描述 {len(self.ent_vecs)} | 查询形态=open_slot（命题索引）"
              f"（{self.decomp_steps}步×{self.decomp_top_k}条） 精排={self.rerank}",
              flush=True)
        return self

    def _embed_cache_ok(self, cdir, tag: str) -> bool:
        """缓存向量是否属于**当前** embedding 口径（换 embedder 后必须重算）。

        为什么不能只看"文件在不在"：换 embedder（bge→Qwen3 都截到 1024 维）后
        `.npy` 的形状一模一样，旧向量会被静默复用 → 实体向量与查询向量不在同一空间，
        召回全错但没有任何报错。用 `components.json` 里的指纹判定。
        """
        import json
        from common.run_record import embed_fingerprint
        try:
            p = cdir / f"{tag}_components.json"
            if not p.exists():
                return False
            return json.loads(p.read_text(encoding="utf-8")).get("embed_fingerprint") \
                == embed_fingerprint()
        except Exception:                                        # noqa: BLE001
            return False

    def _write_embed_cache(self, cdir, tag: str) -> None:
        import json
        from common.run_record import embed_fingerprint
        try:
            cdir.mkdir(parents=True, exist_ok=True)
            (cdir / f"{tag}_components.json").write_text(json.dumps({
                "embed_fingerprint": embed_fingerprint(),
                "embed_dim": EMBED_DIM, "chunk_tokens": EXTRACT_CHUNK_TOKENS,
                "chunk_overlap": EXTRACT_CHUNK_OVERLAP,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:                                        # noqa: BLE001
            pass

    def _load_or_build_ent_vecs(self, cdir) -> dict:
        import json
        cache, ids_cache = cdir / "ent_desc_vecs.npy", cdir / "ent_desc_ids.json"
        if cache.exists() and ids_cache.exists() and self._embed_cache_ok(cdir, "ent"):
            ids = json.loads(ids_cache.read_text(encoding="utf-8"))
            v = np.load(str(cache))
            return {eid: v[i] for i, eid in enumerate(ids)}
        texts, ids = [], []
        for e in self.entities:
            desc = (e.get("description") or "").strip()
            if desc:
                texts.append(f"{e.get('name','')}: {desc}")
                ids.append(e["id"])
        if not texts:
            print("[factalign] 无实体描述", flush=True)
            return {}
        v = embed_batch(texts)
        cdir.mkdir(parents=True, exist_ok=True)
        np.save(str(cache), v.astype(np.float32))
        ids_cache.write_text(json.dumps(ids), encoding="utf-8")
        self._write_embed_cache(cdir, "ent")
        print(f"[factalign] 实体描述 embedding 已算并缓存: {len(ids)} 条", flush=True)
        return {eid: v[i] for i, eid in enumerate(ids)}

    def _load_or_build_win_vecs(self, cdir) -> np.ndarray:
        cache = cdir / "window_vecs.npy"
        if cache.exists() and self._embed_cache_ok(cdir, "window"):
            return np.load(str(cache))
        v = embed_batch([w[2] for w in self.windows]) if self.windows \
            else np.zeros((0, EMBED_DIM), dtype=np.float32)
        cdir.mkdir(parents=True, exist_ok=True)
        np.save(str(cache), v.astype(np.float32))
        self._write_embed_cache(cdir, "window")
        print(f"[factalign] 窗口向量已算并缓存: {len(self.windows)} 条", flush=True)
        return v

    def prepare_for(self, dataset: str, sample: int, graph_version: int = 1) -> "FactAlign":
        """run.py 用：记录 (dataset, sample, v) 供 build() 定位语料图。"""
        self.dataset, self._sample, self._graph_version = dataset, sample, graph_version
        return self

    # ── 逐题查询 ────────────────────────────────────────────────────────────
    def query(self, question: str) -> dict:
        if not self._ready:
            raise RuntimeError("请先 build()（或经 run.py 传入 dataset/sample）")
        with phase("query"):
            return self._query(question)

    def _query(self, q: str) -> dict:
        qv = llm_embed(q)

        cand_props: dict[str, dict] = {}
        decomp_log: list = []
        decomp_score: dict[str, float] = {}

        # ── 1) 拆解 + 迭代检索（主方法：开放槽命题 × 命题索引）──
        # ★ 原 `if self.decompose:` 分支已去掉：主方法**总是**拆解（单轮召回是消融，
        #   放 test_layer.py）。同理去掉了 `_use_chunk`（--decompose-recall chunk 消融）。
        _next_prompt = NEXT_PROMPT_OPEN_SLOT
        sub_q = q
        seen_subs = {sub_q.lower()}
        for step in range(1, self.decomp_steps + 1):
            qv_s = llm_embed(sub_q)
            added, hits = 0, []
            sims = self.pvecs @ qv_s
            for i in np.argsort(-sims):
                h = self.propositions[int(i)]
                if h["id"] in cand_props:
                    decomp_score[h["id"]] = max(decomp_score.get(h["id"], 0.0),
                                                float(sims[int(i)]))
                    continue
                cand_props[h["id"]] = h
                decomp_score[h["id"]] = float(sims[int(i)])
                # ★ 记**命题**（不是原文段落）：`summary` 就是命题全文，见 `extract.py:291`。
                #   上限 200 与 `common/trace.py:32` 的 `HEAD_CHARS` 同档 —— 这里截多少，
                #   trace 的 `text_head` 就是多少，两者必须一致，否则"trace 看到的"和
                #   "实际检索到的"对不上账。原来这里是 `[:70]`，导致：
                #     · `text_head` 只有 70 字、句子被砍断，看起来不像完整命题；
                #     · trace 里的 `chars` 恒为 70 是**假值**（sha256 也是 70 字的哈希，
                #       与 `propositions.json` 里的真命题对不上）。
                #   ★ 传 dict（不是 tuple）是关键：`trace.py:182-190` 的 dict 分支会消费
                #     `text` 并保留其余键，于是 `prop_id` 能落盘、可反查命题；tuple 分支
                #     （`trace.py:362`）只能拿到两个值，id 无处可放。
                #   ★ 不带 `title`：那需要把 `build_chunks` 的 `chunk_text`（含 `title\n`
                #     前缀）与 `window_text`（纯正文）做**逐字**匹配，口径未实测，风险不值。
                hits.append({"text": h.get("summary", "")[:200],
                             "prop_id": h["id"],
                             "sem": float(sims[int(i)])})
                added += 1
                if added >= self.decomp_top_k:
                    break
            decomp_log.append(("SEARCH", sub_q, hits))
            if step == self.decomp_steps:
                self._bump("max_steps")
                break
            ev_lines = "\n".join(f"- {h.get('summary','')}"
                                 for h in list(cand_props.values())[:15])
            nxt_out = llm_chat(NEXT_SYSTEM,
                               _next_prompt.format(question=q, evidence=ev_lines),
                               json_mode=True, timeout=90)
            if not nxt_out:
                decomp_log.append(("STOP", "LLM 失败", []))
                self._bump("llm_fail")
                break
            if nxt_out.get("direct_answer"):
                decomp_log.append(("ENOUGH", "", []))
                self._bump("direct_answer")
                break
            nxt = str(nxt_out.get("next_question") or "").strip()
            if not nxt:
                decomp_log.append(("STOP", "没给出下一问", []))
                self._bump("empty_next")
                break
            if nxt.lower() in seen_subs:
                decomp_log.append(("STOP", f"查询重复: {nxt[:50]}", []))
                self._bump("repeat")
                break
            seen_subs.add(nxt.lower())
            sub_q = nxt

        # （原「2) 非拆解：单轮召回」与「3) 消融：拆解检索原文窗口」两段已删除：
        #   它们只在 --no-decompose / --single-recall rrf|bm25|dense /
        #   --decompose-recall chunk 这些**消融开关**下才会走到，属 test_layer.py 的职责。）

        # ── 4) 可选：一跳实体近邻扩展 ──
        hop_added = 0
        if self.hope > 0 and cand_props:
            for _ in range(self.hope):
                anchor = {eid for h in cand_props.values()
                          for eid in h.get("entity_ids", []) if eid in self.ent_vecs}
                if not anchor:
                    break
                A = np.stack([self.ent_vecs[e] for e in anchor])
                sim = A @ self.ent_mat.T
                k = min(self.hop_k, self.ent_mat.shape[0])
                topn = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
                nb = {self.ent_list[j] for row in topn for j in row} - anchor
                added = {}
                for eid in nb:
                    for pid in self.ent_props.get(eid, ()):
                        if pid not in cand_props:
                            added[pid] = self.prop_by_id[pid]
                if not added:
                    break
                cand_props.update(added)
                hop_added += len(added)

        cand_list = list(cand_props.values())
        if not cand_list:
            return self._result(q, generate_answer(q, [], dataset=getattr(self, "dataset", "")), [], decomp_log, note="empty")

        # ── 5) 池内精排 ──
        sem = np.array([cosine(qv, self.pvecs[self.prop_idx[h["id"]]]) for h in cand_list],
                       dtype=np.float32)
        final = _norm(sem)
        seed_mask = None
        if self.filter_pre > 0:
            _n = min(self.filter_pre, len(cand_list))
            kept, _ = evidence_filter(q, cand_list[:_n])
            kept_ids = {h["id"] for h in kept}
            seed_mask = np.array([h["id"] in kept_ids for h in cand_list], dtype=bool)
        if self.ppr_w > 0:
            ppr = ppr_pool(cand_list, sem, {e["id"]: e["name"] for e in self.entities},
                           alpha=self.ppr_alpha, seed_mask=seed_mask)
            final = (1.0 - self.ppr_w) * _norm(sem) + self.ppr_w * _norm(ppr)
        elif self.rerank == "hybrid":
            ent_scores = np.zeros(len(cand_list), dtype=np.float32)
            for ci, h in enumerate(cand_list):
                best = 0.0
                for eid in h.get("entity_ids", []):
                    v = self.ent_vecs.get(eid)
                    if v is not None:
                        best = max(best, cosine(qv, v))
                ent_scores[ci] = best
            final = 0.7 * final + 0.3 * _norm(ent_scores)

        # ── 6) 跨跳分数融合：子查询命中分兜底（论文的 max-pooling 融合）──
        # 主方法总是拆解，所以这里恒为可走；`decomp_score` 为空（4 轮都没命中新命题）时跳过。
        if decomp_score:
            dsc = np.array([decomp_score.get(h["id"], 0.0) for h in cand_list],
                           dtype=np.float32)
            final = np.maximum(final, _norm(dsc))

        order = (np.arange(len(cand_list)) if self.rerank == "none"
                 else np.argsort(-final))
        _ntop = self.gen_top if self.gen_top > 0 else FACTALIGN_EVIDENCE_PROPS
        top = [cand_list[i] for i in order[: _ntop]]
        if self.evidence_filter_on:
            top, _ = evidence_filter(q, top, keep_top=self.filter_keep)
        ans = generate_answer(q, top,
                              windows=FACTALIGN_EVIDENCE_WINDOWS,
                              dataset=getattr(self, "dataset", ""))
        return self._result(q, ans, [h.get("window_text") or h.get("summary", "") for h in top],
                            decomp_log)

    def _bump(self, reason: str):
        st = getattr(self, "stats", None)
        if isinstance(st, dict):
            st["stop_reasons"][reason] = st["stop_reasons"].get(reason, 0) + 1

    def _result(self, q, answer, evidence, decomp_log, locs=None, note="") -> dict:
        return {"answer": answer,
                "evidence": [e for e in evidence if e],
                "evidence_locs": locs or [],
                "decomp_log": decomp_log,
                "note": note}

    # 语料级方法不走 per-question run()
    def run(self, question, contexts):
        raise NotImplementedError("FactAlign 走语料级：run.py 里先 build(语料) 再逐题 query")
