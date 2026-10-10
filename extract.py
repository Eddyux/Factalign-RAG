"""实体 + 自包含命题抽取。

一次 LLM 调用同时产出实体与命题；命题 = 一句能单独看懂的事实句（检索单元）。
prompt 放在 prompts/extract_graph.txt（改文件即生效，无需改代码）。

图按题缓存到 CACHE_DIR：缓存 key = 抽取 prompt + 问题 + 上下文 的 sha1，因此
改抽取 prompt 会自动失效旧缓存，不会用到脏图。
"""

import hashlib
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from common.llm import EMBED_DIM, chat as llm_chat, embed_batch, phase, split_by_tokens
from config import (EXTRACT_CHUNK_TOKENS, EXTRACT_CHUNK_OVERLAP,
                     CACHE_DIR, EMBED_MODEL, CHAT_MODEL, EXTRACT_WORKERS, EXTRACT_VERSION,
                     PROMPTS_DIR, dataset_chunking)

_EXTRACT_PROMPT_PATH = PROMPTS_DIR / "extract_graph.txt"
if not _EXTRACT_PROMPT_PATH.exists():
    raise RuntimeError(
        f"构图 prompt 文件缺失：{_EXTRACT_PROMPT_PATH}\n"
        f"改构图 prompt 请直接编辑该文件（改完重跑即生效，缓存按 prompt 内容自动失效）")
EXTRACT_SYSTEM = _EXTRACT_PROMPT_PATH.read_text(encoding="utf-8").strip()

# ★ 抽取失败重试口径（全项目统一，对齐 hipporag 的 `HIPPORAG_OPENIE_TRIES=3`）：
#   **解析失败最多再抽 2 次、封顶 3 次，仍失败就丢弃该窗口**（该窗口不进图）。
# 同口径：hipporag 3 / hypergraphrag 3 / lightrag 3。只有「解析不出结果」才重发；
# 解析成功但实体为空**不算失败**（那是模型的正常输出，重发也白费）。
EXTRACT_TRIES = int(os.environ.get("EXTRACT_TRIES", "3"))


def make_id(prefix: str, seed: str) -> str:
    return prefix + hashlib.sha1(seed.encode("utf-8")).hexdigest()[:10]


def norm(s: str) -> str:
    return s.strip().lower()


def _coerce_str(x) -> str:
    """LLM 可能返回 str 也可能返回 list（如多个年份），统一压成 str 落盘。"""
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple)):
        return " ".join(str(i) for i in x)
    return str(x) if x else ""


def split_sentences(text: str) -> list[str]:
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sents if s.strip()]


def _embed_fingerprint() -> str:
    """抽取缓存的 embedding 口径标识：**basename + 全部嵌入旋钮**。

    为什么不能只拿 `EMBED_MODEL` 全部入 key：同一个模型用**不同字符串**指向时
    （`export EMBED_MODEL=/root/.../Qwen3-Embedding-0.6B` vs 默认值，或 HF 名 vs 本地目录）
    会被当成两个模型 → 同一语料算出两个缓存 key → 抽取缓存全失效、白抽一遍。
    取 basename 后这些情况都归一到同一个名字，跨机器/跨路径也稳定。

    但只取 basename 又不够：换 embedder（如 `Qwen3-Embedding-8B` 的 basename 与 0.6B 不同，
    能挡住），可只改 `EMBED_TOKENIZER`（换 tokenizer）或 `CHUNK_TOKENIZER`（tiktoken）时
    basename 不变而**切块口径变了** → 旧缓存被复用，图和向量口径就混了。所以这里用
    `common.run_record.embed_fingerprint()` 的口径（模型 + 维度 + 上限 + 两把尺子 + 查询格式）
    拼上 basename。
    """
    from common.run_record import embed_fingerprint
    name = os.path.basename(str(EMBED_MODEL).rstrip("/\\")) or str(EMBED_MODEL)
    return f"{name}|{embed_fingerprint()}"


def _cache_key(question: str, contexts: list[dict], dataset: str = "") -> str:
    h = hashlib.sha1()
    h.update(EXTRACT_SYSTEM.encode("utf-8"))  # 抽取 prompt 变了自动失效缓存
    h.update(_embed_fingerprint().encode("utf-8"))  # embedding 变了也自动失效缓存
    h.update(CHAT_MODEL.encode("utf-8"))      # 换抽取 LLM 也自动失效缓存（避免复用旧模型抽的图）
    h.update(EXTRACT_VERSION.encode("utf-8"))  # 建图管线版本号：合并/关联逻辑变更后自动失效旧缓存
    # 切块参数进 key。★ 用**按数据集解析后**的值（triviaqa=1200/100，其余=2000/0）——
    # 写死全局值的话，改了 triviaqa 的切块口径仍会命中旧缓存、静默复用切法不同的图。
    _ct, _co = (dataset_chunking(dataset) if dataset
                else (EXTRACT_CHUNK_TOKENS, EXTRACT_CHUNK_OVERLAP))
    h.update(f"{_ct}|{_co}".encode("utf-8"))
    h.update(question.encode("utf-8"))
    for c in contexts:
        h.update(c.get("title", "").encode("utf-8"))
        h.update(c.get("text", "").encode("utf-8"))
    return h.hexdigest()[:16]


def build_graph_for_question(question: str, contexts: list[dict], use_cache: bool = True,
                             dataset: str = ""):
    """返回 (entities, propositions, chunks, pvecs)。默认按题缓存并复用。

    `dataset` 只用于**解析切块口径**（triviaqa=1200/100，其余不切块）：它同时进缓存 key
    和真正的切块调用 —— 两处必须用同一个值，否则会出现"按 A 口径取缓存、按 B 口径新建"。
    """
    key = _cache_key(question, contexts, dataset)
    cdir = CACHE_DIR / key
    if use_cache and (cdir / "pvecs.npy").exists():
        entities = json.loads((cdir / "entities.json").read_text(encoding="utf-8"))
        propositions = json.loads((cdir / "propositions.json").read_text(encoding="utf-8"))
        chunks = json.loads((cdir / "chunks.json").read_text(encoding="utf-8"))
        pvecs = np.load(str(cdir / "pvecs.npy"))
        return entities, propositions, chunks, pvecs

    entity_registry: dict[str, dict] = {}
    alias_index: dict[str, str] = {}
    all_propositions: list[dict] = []
    chunk_records: list[dict] = []

    # ── 1. 建 sentence chunk + 切窗任务（chunk 带 qid，语料级下不同题的句子不撞 id）──
    # 窗口按 token 切（embedding 模型同一 tokenizer）。口径按数据集解析：
    #   triviaqa → 1200/100（它的 passage 是整篇维基文章）；其余 → 全局值（默认 2000/0 不切块）
    CHUNK, OVERLAP = (dataset_chunking(dataset) if dataset
                      else (EXTRACT_CHUNK_TOKENS, EXTRACT_CHUNK_OVERLAP))
    tasks = []
    for ctx in contexts:
        title = ctx.get("title", "")
        qid = str(ctx.get("qid", ""))
        sents = ctx.get("sentences") or split_sentences(ctx.get("text", ""))
        if not sents:
            continue
        sent_records = []
        for si, sent in enumerate(sents):
            sc = {
                "id": make_id("C", f"{qid}|{title}|{si}|{sent[:60]}"),
                "title": title, "sent_idx": si, "text": sent,
                "entity_ids": [], "proposition_ids": [],
            }
            sent_records.append(sc)
            chunk_records.append(sc)
        full_text = " ".join(sc["text"] for sc in sent_records)
        # 每句在 full_text 中的字符区间，供切块后找相交句子
        spans = []
        pos = 0
        for sc in sent_records:
            spans.append((pos, pos + len(sc["text"])))
            pos += len(sc["text"]) + 1
        for slice_text, cs, ce in split_by_tokens(full_text, CHUNK, OVERLAP):
            idxs = [i for i, (s0, s1) in enumerate(spans) if s1 > cs and s0 < ce]
            if slice_text.strip() and idxs:
                tasks.append((title, slice_text, idxs, sent_records))

    # ── 2. 并发抽取（LLM 调用并发，结果顺序处理，对齐 hotpotqa） ──
    def extract_one(title, text):
        user = f"Title: {title}\n\nText:\n{text}"
        # phase 是**线程本地**的：调用方（test_layer/run.py）的 phase("build") 只作用于
        # 主线程，抽取跑在线程池里，不在这里各设一次的话，_cur_phase() 会回退成 "query"
        # → 建图的抽取 token 全被记进 query 桶，build/query 分桶就错了。
        with phase("build"):
            # 解析失败 → 最多重发 EXTRACT_TRIES 次（封顶 3 次），仍失败返回 None → 该窗口
            # 被下面 `if not extracted: continue` 丢弃（进不了图）。
            # 解析成功（拿到 dict）就收，**不看实体是否为空**——空实体是模型的正常输出。
            for _attempt in range(EXTRACT_TRIES):
                try:
                    out = llm_chat(EXTRACT_SYSTEM, user, json_mode=True, timeout=180)
                except Exception:                        # noqa: BLE001  传输/超时 → 也算一次
                    out = None
                if isinstance(out, dict):
                    return out
            return None

    # 增量断点：每 20 窗保存一次成功结果；中断重跑只补没完成/失败的窗口。
    # 只存非 None 结果（失败的下次重试）；最终缓存写完后删掉 checkpoint。
    ckpt_file = cdir / "checkpoint.json"
    results_map = {}
    if ckpt_file.exists():
        try:
            results_map = {int(k): v for k, v in
                           json.loads(ckpt_file.read_text(encoding="utf-8")).items()}
            print(f"  恢复抽取进度：已完成 {len(results_map)}/{len(tasks)} 窗口", flush=True)
        except Exception:
            results_map = {}

    pending_tasks = [(i, t) for i, t in enumerate(tasks) if i not in results_map]
    done = len(results_map)
    _bar_len = 0
    if pending_tasks:
        with ThreadPoolExecutor(max_workers=EXTRACT_WORKERS) as executor:
            futures = {executor.submit(extract_one, t[0], t[1]): i for i, t in pending_tasks}
            for f in as_completed(futures):
                i = futures[f]
                try:
                    results_map[i] = f.result()
                except Exception:
                    results_map[i] = None
                done += 1
                if done % 20 == 0 or done == len(tasks):
                    # 单行进度条（\r 覆盖），不再逐行刷屏
                    line = f"  抽取 {done}/{len(tasks)} 窗口"
                    sys.stderr.write("\r" + line + " " * max(0, _bar_len - len(line)))
                    sys.stderr.flush()
                    _bar_len = len(line)
                    good = {str(k): v for k, v in results_map.items() if v}
                    cdir.mkdir(parents=True, exist_ok=True)
                    ckpt_file.write_text(json.dumps(good, ensure_ascii=False),
                                         encoding="utf-8")
        if _bar_len:
            sys.stderr.write("\n")
            sys.stderr.flush()

    for i, (title, slice_text, idxs, sent_records) in enumerate(tasks):
        extracted = results_map.get(i)
        if not extracted:
            continue
        local_to_global: dict[str, str] = {}
        chunk_eids: list[str] = []          # 本窗涉及实体的 gid（用于窗口内 sentence 关联）
        gid_entities: dict[str, dict] = {}  # gid -> registry 实体（含别名，关联时用）

        for ent in extracted.get("entities", []):
            if not isinstance(ent, dict):
                continue
            name = str(ent.get("name", "")).strip()
            if not name:
                continue
            key = norm(name)
            if key in entity_registry:
                target = key
            elif key in alias_index:
                target = alias_index[key]
            else:
                target = None
                for a in ent.get("aliases", []):
                    ak = norm(a)
                    if ak in alias_index:
                        target = alias_index[ak]
                        break
                if target is None:
                    target = key
            if target not in entity_registry:
                entity_registry[target] = {
                    "id": make_id("E", target), "name": name,
                    "aliases": [a.strip() for a in ent.get("aliases", []) if a.strip() and a.strip() != name],
                    "description": ent.get("description", ""),
                }
                alias_index.setdefault(key, target)
                for a in ent.get("aliases", []):
                    alias_index.setdefault(norm(a), target)
            else:
                ex = entity_registry[target]
                if norm(ex["name"]) != key and name not in ex["aliases"]:
                    ex["aliases"].append(name)
                for a in ent.get("aliases", []):
                    a = a.strip()
                    if a and a != name and a not in ex["aliases"]:
                        ex["aliases"].append(a)
                alias_index.setdefault(key, target)
                # 同一实体多次出现时保留更长的 description
                if len(ent.get("description", "")) > len(ex.get("description", "")):
                    ex["description"] = ent["description"]
            lid = str(ent.get("local_id", "")).strip()
            if lid:
                gid = entity_registry[target]["id"]
                local_to_global[lid] = gid
                if gid not in chunk_eids:
                    chunk_eids.append(gid)
                    gid_entities[gid] = entity_registry[target]

        chunk_he_ids: list[str] = []
        prop_by_id: dict[str, dict] = {}
        for prop in extracted.get("propositions", []):
            if not isinstance(prop, dict):
                continue
            summary = str(prop.get("summary", "")).strip()
            if not summary:
                continue
            eids, enames = [], []
            for lid in prop.get("entity_ids", []):
                gid = local_to_global.get(str(lid).strip())
                if gid and gid not in eids:
                    eids.append(gid)
                    # 用 id 反查实体名（entity_registry 的 key 是"归一化名"，不是 id）
                    ent_obj = gid_entities.get(gid)
                    if ent_obj is None:
                        for _e in entity_registry.values():
                            if _e["id"] == gid:
                                ent_obj = _e
                                break
                    enames.append(ent_obj["name"] if ent_obj else gid)
            if not eids:
                continue
            pid = make_id("H", summary + "|" + "|".join(sorted(eids)))
            he_rec = {
                "id": pid, "summary": summary,
                "entity_ids": eids, "entity_names": enames,
                "window_text": slice_text,  # small-to-big：生成时映射回原文窗口
            }
            all_propositions.append(he_rec)
            chunk_he_ids.append(pid)
            prop_by_id[pid] = he_rec

        # 窗口内关联 sentence（实体名/别名关键词重叠，对齐 hotpotqa：
        # 只在抽取窗口范围内关联，避免语料级下 O(H×C) 全量扫描和跨题污染）
        for si in idxs:
            sc = sent_records[si]
            sent_lower = sc["text"].lower()
            for eid in chunk_eids:
                e = gid_entities[eid]
                matched = e["name"].lower() in sent_lower
                if not matched:
                    for alias in e.get("aliases", []):
                        if alias.lower() in sent_lower:
                            matched = True
                            break
                if matched and eid not in sc["entity_ids"]:
                    sc["entity_ids"].append(eid)
            for pid in chunk_he_ids:
                he_rec = prop_by_id[pid]
                matched = False
                for ename in he_rec.get("entity_names", []):
                    if ename.lower() in sent_lower:
                        matched = True
                        break
                if not matched:
                    for eid in he_rec.get("entity_ids", []):
                        e = gid_entities.get(eid)
                        if e:
                            for alias in e.get("aliases", []):
                                if alias.lower() in sent_lower:
                                    matched = True
                                    break
                            if matched:
                                break
                if matched and pid not in sc["proposition_ids"]:
                    sc["proposition_ids"].append(pid)

    # ── 3. 命题去重（sentence 关联已在窗口内完成）──
    seen = {}
    for prop in all_propositions:
        k = prop["summary"]
        if k not in seen:
            seen[k] = prop
        else:
            ex = seen[k]
            ex["entity_ids"] = sorted(set(ex["entity_ids"]) | set(prop["entity_ids"]))
            ex["entity_names"] = sorted(set(ex["entity_names"]) | set(prop["entity_names"]))
    propositions = list(seen.values())

    entities = sorted(entity_registry.values(), key=lambda e: e["id"])
    htexts = [h["summary"] for h in propositions]
    pvecs = embed_batch(htexts) if htexts else np.zeros((0, EMBED_DIM))

    # ── 存盘缓存，下次直接复用 ──
    if not propositions:
        # 抽取全失败（多半是 LLM 服务挂了）：不能把空图写进缓存，
        # 否则下次会"命中"这个垃圾缓存，出现删了缓存却还在用缓存的假象
        print(f"[WARN] 抽取 0 命题（LLM 服务不可用？），不写缓存", flush=True)
        return entities, propositions, chunk_records, pvecs
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "entities.json").write_text(json.dumps(entities, ensure_ascii=False, indent=2), encoding="utf-8")
    (cdir / "propositions.json").write_text(json.dumps(propositions, ensure_ascii=False, indent=2), encoding="utf-8")
    (cdir / "chunks.json").write_text(json.dumps(chunk_records, ensure_ascii=False, indent=2), encoding="utf-8")
    np.save(str(cdir / "pvecs.npy"), pvecs.astype(np.float32))
    ckpt_file.unlink(missing_ok=True)  # 完整缓存已写，checkpoint 使命完成
    return entities, propositions, chunk_records, pvecs
