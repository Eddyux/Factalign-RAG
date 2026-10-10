"""体检：各方法**实际索引的块**，和我们喂进去的整段窗口是不是同一批文本。

为什么需要：我们按项目口径切块（**默认不切块** = 整段 passage；triviaqa 例外，切成
1200/100）——尺子是 embedding 模型的 tokenizer；LightRAG / HyperGraphRAG 内部的
chunk_token_size 用 tiktoken 记账。数字口径不同，但**落到文本上必须一致**——否则
"同一语料、同一粒度"就不再是单变量对照（审稿人一定会问这一条）。

做法：重建"我们喂给框架的 docs"（各适配器用的是同一段代码：`title + text`，按
`config.dataset_chunking()` 的口径切），再和框架自己落盘的块集合比对：
  - LightRAG / HyperGraphRAG : kv_store_text_chunks.json 的 content
  - HippoRAG                 : openie_results_ner_*.json 的 passage

用法：
  python tools/audit_chunk_parity.py --dataset 2wiki --sample 300
退出码：0=全部一致（或没找到框架落盘）；1=有不一致，并打印差异样例。
"""

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def our_docs(dataset: str, sample: int) -> list[str]:
    """重建喂给框架的文档（与 lightrag/hypergraphrag/hipporag 适配器逐字一致）。"""
    from corpus import merged_contexts
    from common.llm import split_by_tokens
    from config import EXTRACT_CHUNK_TOKENS, EXTRACT_CHUNK_OVERLAP

    docs: list[str] = []
    for c in merged_contexts(dataset, sample):
        full = f"{c.get('title', '')}\n{c.get('text', '')}".strip()
        for chunk_text, _cs, _ce in split_by_tokens(full, EXTRACT_CHUNK_TOKENS,
                                                    EXTRACT_CHUNK_OVERLAP):
            if chunk_text.strip():
                docs.append(chunk_text.strip())
    return docs


def _load_json(p: pathlib.Path):
    try:
        return json.loads(p.read_text(encoding="utf-8-sig"))
    except Exception as e:  # noqa: BLE001
        print(f"  ! 读不了 {p}: {type(e).__name__}: {e}")
        return None


def _framework_of(p: pathlib.Path) -> str:
    """从落盘路径猜是哪个方法（顺序要紧：hypergraphrag 里含 "graphrag" 子串）。"""
    s = str(p).lower()
    for k in ("hypergraphrag", "hipporag", "lightrag", "s2grag"):
        if k in s:
            return k
    return p.parent.name


def collect(dataset: str) -> list[tuple[pathlib.Path, str, set]]:
    """扫 data/ 下各框架落盘的块集合。"""
    out = []
    data = ROOT / "data"
    for p in sorted(data.glob(f"**/{dataset}/**/kv_store_text_chunks.json")):
        d = _load_json(p)
        if isinstance(d, dict):
            theirs = {str(v.get("content", "")).strip() for v in d.values()
                      if isinstance(v, dict)}
            out.append((p, "kv_store_text_chunks", theirs - {""}))
    for p in sorted(data.glob(f"**/{dataset}/**/openie_results_ner_*.json")):
        d = _load_json(p)
        if isinstance(d, dict):
            theirs = {str(x.get("passage", "")).strip() for x in d.get("docs", [])
                      if isinstance(x, dict)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="2wiki")
    ap.add_argument("--sample", type=int, default=300)
    ap.add_argument("--show", type=int, default=3)
    ap.add_argument("--all", action="store_true",
                    help="把所有 data/samples/*.json 都扫一遍，最后打一张汇总表")
    a = ap.parse_args()

    if a.all:
        targets = []
        for p in sorted((ROOT / "data" / "samples").glob("*.json")):
            stem = p.stem                      # 如 2wiki_300 / hotpotqa_gr1_128
            ds, _, n = stem.rpartition("_")
            if ds and n.isdigit():
                targets.append((ds, int(n)))
        if not targets:
            print("data/samples/ 下没有题集")
            return 0
    else:
        targets = [(a.dataset, a.sample)]

    bad = 0
    summary = []
    for ds, n in targets:
        docs = our_docs(ds, n)
        ours = set(docs)
        print(f"=== {ds}_{n}：我们的窗口 {len(docs)} 条（去重 {len(ours)}）===")
        rows = collect(ds)
        found = {}
        for p, kind, theirs in rows:
            rel = p.relative_to(ROOT)
            if not theirs:
                print(f"  - {rel}  [{kind}] 跳过（没解析到块）")
                continue
            fw = _framework_of(p)
            same = theirs == ours
            found[fw] = (len(theirs), same)
            print(f"  - {rel}  [{kind}] 块={len(theirs)} 交集={len(theirs & ours)} "
                  f"{'✓' if same else '✗ 有差异'}")
            if not same:
                bad += 1
                for t in sorted(theirs - ours)[:a.show]:
                    print(f"      [框架独有] {t[:90]!r}")
                for t in sorted(ours - theirs)[:a.show]:
                    print(f"      [我们独有] {t[:90]!r}")
        summary.append(((ds, n), len(docs), len(ours), found))
        print()

    print("=== 汇总 ===")
    print(f"{'数据集':<18}{'窗口':>7}{'去重':>7}   各方法索引块数（✓=与窗口逐条一致）")
    for (ds, n), ndoc, nuniq, found in summary:
        cells = "  ".join(
            f"{fw}={cnt}{'✓' if ok else '✗'}"
            for fw, (cnt, ok) in sorted(found.items())) or "（未找到框架索引）"
        print(f"{ds + '_' + str(n):<18}{ndoc:>7}{nuniq:>7}   {cells}")

    print()
    if bad:
        print(f"✗ 有 {bad} 处块集合不一致 —— 不是单一变量对照，"
              f"要么改参数重跑，要么在论文里写明差异")
        return 1
    print("✓ 找到的框架索引都与我们的整段窗口逐条一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
