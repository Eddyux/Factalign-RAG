"""语料级建图：把数据集 N 题的上下文合并，直接建「一张」图（单一实体注册表）。

与 Graph-R1 / HippoRAG 同款：图建在语料上，题目作为查询。
"""

import json
import re

import numpy as np

from common.data import iter_questions
from extract import build_graph_for_question
from config import (CORPUS_DIR, EXTRACT_VERSION, CHAT_MODEL, EMBED_DIM, MERGED_DIR,
                    EXTRACT_CHUNK_TOKENS, EXTRACT_CHUNK_OVERLAP, dataset_chunking)


def _slug(s: str) -> str:
    """模型名 → 文件系统安全片段（做 meta 指纹）。"""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", s)


def _corpus_dir(dataset: str, n: int, graph_version: int = 1):
    """语料图缓存目录：data/corpus/{dataset}/v{n}/。

    v = 第几次建图（--graph-version）：换图参数想保留旧图就开新 v；
    同一 v 内建图参数（管线版本/chat 模型/**题数**）变化会由 meta 校验触发重建。

    ⚠️ 目录名里**不含题数**（历史布局），所以 meta 里记了 `sample`：不同 `--sample`
    绝不能共用同一张语料图（300 题和 500 题的语料是两个不同的语料）。
    """
    return CORPUS_DIR / dataset / f"v{graph_version}"


def _meta(n: int, dataset: str = "") -> dict:
    # sample 必须记：目录名不含题数，没有它就无法判断这张图是按几百题建的
    # embed_fingerprint 必须记：换 embedder/尺子后 pvecs 形状可能一致，旧向量会被静默复用
    # chunking 必须记：目录名也不含切块口径，而 `chunks.json` 是按它切出来的 —— 不记的话
    #   改了数据集切块口径（如 triviaqa 1200/100）会**静默复用按旧口径切好的语料单元**。
    from common.run_record import embed_fingerprint
    _ct, _co = dataset_chunking(dataset) if dataset else (EXTRACT_CHUNK_TOKENS, EXTRACT_CHUNK_OVERLAP)
    return {"extract_version": EXTRACT_VERSION, "chat_model": CHAT_MODEL, "sample": n,
            "embed_fingerprint": embed_fingerprint(), "embed_dim": EMBED_DIM,
            "chunking": {"tokens": _ct, "overlap": _co}}


def _merged_from_samples(dataset: str, n: int) -> list[dict]:
    """从 `data/samples/{dataset}_{n}.json` 现场拼语料（每条带 qid 记来源题）。

    同一段被多道题共用时只留一份（按 `(title, text)` 精确去重）。
    这是 `tools/build_merged_corpus.py` 落盘快照时用的**唯一真源**。
    """
    items = list(iter_questions(dataset, n))
    all_contexts = []
    seen = set()
    for item in items:
        for c in item["contexts"]:
            key = (c.get("title", ""), c.get("text", ""))
            if key in seen:
                continue
            seen.add(key)
            c2 = dict(c)
            c2["qid"] = item["id"]  # 记来源题，供以后算跨题污染
            all_contexts.append(c2)
    return all_contexts


def merged_contexts(dataset: str, n: int) -> list[dict]:
    """取语料，供建图 / 外部 baseline 建索引。

    优先读 `data/merged/{dataset}_{n}/corpus.jsonl` —— 那是 `tools/build_merged_corpus.py`
    固化的快照，也正是 `checksums.json` 里 sha256 认证的那一份。这样「跑实验用的语料」
    与「被认证的语料」是同一份，改了语料构造也不会两边对不上。
    快照不存在时回退到现场拼。
    """
    snap = MERGED_DIR / f"{dataset}_{n}" / "corpus.jsonl"
    if snap.exists():
        rows = [json.loads(ln) for ln in snap.read_text(encoding="utf-8").splitlines()
                if ln.strip()]
        if rows:
            return rows
    print(f"[corpus] 未找到语料快照 {snap} → 现场拼（未落盘、不参与 sha256 认证）。"
          f"要固化：python tools/build_merged_corpus.py --dataset {dataset} --sample {n}",
          flush=True)
    return _merged_from_samples(dataset, n)


def build_corpus(dataset: str, n: int, graph_version: int = 1):
    """把 N 题上下文拼成一个 corpus，建一张图，存盘。返回 (entities, propositions, chunks, pvecs)。"""
    all_contexts = merged_contexts(dataset, n)

    # ★ dataset 必须传下去：切块口径按数据集解析（triviaqa=1200/100，其余不切块）。
    entities, propositions, chunks, pvecs = build_graph_for_question("corpus", all_contexts,
                                                                     dataset=dataset)

    outdir = _corpus_dir(dataset, n, graph_version)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "entities.json").write_text(json.dumps(entities, ensure_ascii=False, indent=2), encoding="utf-8")
    (outdir / "propositions.json").write_text(json.dumps(propositions, ensure_ascii=False, indent=2), encoding="utf-8")
    (outdir / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    np.save(str(outdir / "pvecs.npy"), pvecs.astype(np.float32))
    (outdir / "meta.json").write_text(json.dumps(_meta(n, dataset), ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    print(f"语料建图完成: {len(entities)}e {len(propositions)}h -> {outdir}"
          f"（{n} 题语料，已把题数写进 meta.json）")
    return entities, propositions, chunks, pvecs


def load_corpus(dataset: str, n: int, graph_version: int = 1):
    outdir = _corpus_dir(dataset, n, graph_version)
    if not (outdir / "pvecs.npy").exists():
        raise FileNotFoundError(f"未找到 {outdir}，请先 build_corpus()")
    # meta 校验：同一 v 内建图参数变了 → 旧图作废（换参数请开新 v 或删目录重建）
    if (outdir / "meta.json").exists():
        old = json.loads((outdir / "meta.json").read_text(encoding="utf-8"))
        if old.get("extract_version") != EXTRACT_VERSION or old.get("chat_model") != CHAT_MODEL:
            raise FileNotFoundError(
                f"{outdir} 的建图参数已变化（meta: {old.get('chat_model')}/"
                f"{old.get('extract_version')} vs 当前 {CHAT_MODEL}/{EXTRACT_VERSION}）。"
                f"换参数请开新 v（--graph-version 2）或删掉该目录重建")
        # ★ 题数校验：目录名不含题数，只能靠 meta。300 题和 500 题的语料是**两个不同语料**，
        #   混用会让"语料错、结果全错"且毫无提示。
        if old.get("sample") != n:
            raise FileNotFoundError(
                f"{outdir} 是按 **{old.get('sample')} 题**语料建的，当前要的是 **{n} 题**"
                f"（`--sample {n}`）——两者语料不同，不能共用同一张图。\n"
                f"  要保留旧图：换 --graph-version；要覆盖：让它重建（会自动重建并打印）")
        # ★ 向量口径校验：换 embedding 模型/尺子后 pvecs 的形状常常**完全一致**
        #   （比如 bge→Qwen3 都截到 1024 维），旧向量会被静默复用 → 新旧向量空间混用，
        #   检索结果不可信却没有一行报错。只能靠指纹拦。
        from common.run_record import embed_fingerprint
        fp_now = embed_fingerprint()
        # ★ 切块口径校验：`chunks.json` 是按切块参数切出来的，而目录名不含该参数。
        #   不校验的话，改了数据集切块口径（如 triviaqa 2000/0 → 1200/100）后旧语料单元
        #   会被**静默复用** —— 检索单元与声称的口径不一致，且没有任何报错。
        _ct, _co = dataset_chunking(dataset)
        old_ck = old.get("chunking") or {}
        if (old_ck.get("tokens") or old_ck.get("overlap")) is not None and (
                old_ck.get("tokens") != _ct or old_ck.get("overlap") != _co):
            raise FileNotFoundError(
                f"{outdir} 是按另一套切块口径建的（meta: {old_ck.get('tokens')}/"
                f"{old_ck.get('overlap')} vs 当前 {_ct}/{_co}）—— `chunks.json` 里的语料单元"
                f"与新口径不一致。\n"
                f"  换参数请开新 v（--graph-version 2）或删掉该目录重建")
        if old.get("embed_fingerprint") not in (None, fp_now):
            raise FileNotFoundError(
                f"{outdir} 是按另一套 embedding 口径建的（meta: "
                f"{old.get('embed_fingerprint')}/{old.get('embed_dim')} vs 当前 "
                f"{fp_now}/{EMBED_DIM}）——旧向量不在同一个向量空间，不能复用。\n"
                f"  换 embedder 请开新 --graph-version（旧 v 留着做对照）")
    else:
        print(f"[corpus] 警告：{outdir} 没有 meta.json，无法确认它是按几百题建的。"
              f"不同 --sample 的语料不能混用——建议删掉该目录或换 --graph-version 重建。",
              flush=True)
    entities = json.loads((outdir / "entities.json").read_text(encoding="utf-8"))
    propositions = json.loads((outdir / "propositions.json").read_text(encoding="utf-8"))
    chunks = json.loads((outdir / "chunks.json").read_text(encoding="utf-8"))
    pvecs = np.load(str(outdir / "pvecs.npy"))
    return entities, propositions, chunks, pvecs


def get_or_build(dataset: str, n: int, graph_version: int = 1):
    """有缓存就读，没有就建。

    注意「参数变了」这一路：`load_corpus` 会因为 meta 指纹不符抛 FileNotFoundError，
    这里接住并**在同一个 v 目录上重建（覆盖旧图）**。以前是静默发生的，现在会先打印一行
    说明——否则"同一 v 号下的图被换掉了"没人知道。要保留旧图请开新 `--graph-version`。
    """
    try:
        return load_corpus(dataset, n, graph_version)
    except FileNotFoundError as e:
        p = _corpus_dir(dataset, n, graph_version)
        if (p / "pvecs.npy").exists():
            print(f"[corpus] {p} 已有旧图，但 {e}\n"
                  f"[corpus] → 在**同一个 v 目录上重建并覆盖**（想保留旧图请换 --graph-version）",
                  flush=True)
        return build_corpus(dataset, n, graph_version)
