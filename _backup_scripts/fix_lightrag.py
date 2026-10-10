"""补抽 lightrag 缺失块（不重跑全量索引）。

用途：建图时部分块抽取失败（超时等）进了 failed_docs，且重试轮被
doc_status 的 content_hash 判重挡掉后，直接在已有索引上补插缺失块：
  1. 从样本重新切块（与 build() 同口径）→ 按 kv_store_full_docs 找缺失块
  2. 删掉 doc_status 里这些块的旧记录（含 r1_/r2_/r3_ 重试残留，清判重依据）
  3. 换 fix_ 前缀的新 id/file_paths 重插缺失块
  4. 验证缺失清零

用法:
  python fix_lightrag.py --dataset hotpotqa --sample 300 --graph-version 1 --workers 16
"""

import argparse
import json
import sys
from pathlib import Path

from common.llm import split_by_tokens
from config import CHAT_MODEL, EMBED_MODEL, EXTRACT_CHUNK_TOKENS, EXTRACT_CHUNK_OVERLAP
from corpus import merged_contexts
from common.result import graph_dir


def _make_docs(dataset: str, sample: int):
    """与 lightrag_adapter.build() 完全一致的切块/编号逻辑。"""
    docs, ids, file_paths = [], [], []
    for c in merged_contexts(dataset, sample):
        title = (c.get('title') or '').strip()
        qid = str(c.get('qid', ''))
        full = f"{title}\n{c.get('text','')}".strip()
        for chunk_text, _cs, _ce in split_by_tokens(full, EXTRACT_CHUNK_TOKENS,
                                                    EXTRACT_CHUNK_OVERLAP):
            if chunk_text.strip():
                docs.append(chunk_text)
                doc_id = f"{qid}_{title}_{len(docs)}"
                ids.append(doc_id)
                file_paths.append(doc_id)
    return docs, ids, file_paths


def _missing(docs, working_dir: Path):
    store = working_dir / "kv_store_full_docs.json"
    if not store.exists():
        return list(range(len(docs)))
    try:
        stored = {str(v.get("content", "")) for v in
                  json.loads(store.read_text(encoding="utf-8")).values()}
    except Exception:
        return list(range(len(docs)))
    return [i for i, d in enumerate(docs)
            if d not in stored and d.strip() not in stored]


def _drop_doc_status(miss_fps: set[str], working_dir: Path) -> int:
    """删 doc_status 里缺失块（含 r{n}_ 重试残留）的旧记录，清掉判重依据。"""
    p = working_dir / "kv_store_doc_status.json"
    if not p.exists():
        return 0
    data = json.loads(p.read_text(encoding="utf-8"))

    def _match(fp):
        if fp in miss_fps:
            return True
        for n in range(1, 4):
            pre = f"r{n}_"
            if fp.startswith(pre) and fp[len(pre):] in miss_fps:
                return True
        return False

    to_del = [k for k, v in data.items() if _match(str(v.get("file_path", "")))]
    for k in to_del:
        del data[k]
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(to_del)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--sample", type=int, default=300)
    ap.add_argument("--graph-version", type=int, default=1)
    args = ap.parse_args()

    wdir = graph_dir("lightrag", args.dataset, args.graph_version)
    if not (wdir / "kv_store_full_docs.json").exists():
        print(f"索引不存在：{wdir}", flush=True)
        sys.exit(1)

    docs, ids, file_paths = _make_docs(args.dataset, args.sample)
    miss = _missing(docs, wdir)
    if not miss:
        print(f"[fix_lightrag] 没有缺失块，无需补抽", flush=True)
        return
    print(f"[fix_lightrag] {len(miss)} 个缺失块：", flush=True)
    for i in miss[:20]:
        print(f"  - {file_paths[i]} | {docs[i][:60]!r}", flush=True)

    # 1) 删 doc_status 旧记录（第一轮失败记录 + r{n}_ 重试残留）
    n_del = _drop_doc_status({file_paths[i] for i in miss}, wdir)
    print(f"[fix_lightrag] 已删 doc_status 记录 {n_del} 条", flush=True)

    # 2) 换 fix_ 前缀重插缺失块（绕过 filename 判重；content_hash 已清）
    from methods.lightrag_adapter import LightRAG as LR
    m = LR()
    m.load(str(wdir))
    fix_ids = [f"fix_{ids[i]}" for i in miss]
    fix_fps = [f"fix_{file_paths[i]}" for i in miss]
    print(f"[fix_lightrag] 重插 {len(miss)} 块（并发 {m._rag.llm_model_max_async}）...", flush=True)
    m._dispatch(m._rag.ainsert([docs[i] for i in miss],
                               ids=fix_ids, file_paths=fix_fps))

    # 3) 验证
    still = _missing(docs, wdir)
    if still:
        print(f"[fix_lightrag] ⚠️ 仍缺 {len(still)} 块：", flush=True)
        for i in still[:10]:
            print(f"  - {file_paths[i]}", flush=True)
    else:
        print(f"[fix_lightrag] ✅ 缺失清零，索引完整", flush=True)


if __name__ == "__main__":
    main()
