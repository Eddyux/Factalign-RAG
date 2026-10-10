"""inspect_isolated.py — 查语料图里的孤立节点（度=0），看它们是实体还是超边、为什么孤立。

用法:
  python inspect_isolated.py --dataset hotpotqa_time --sample 100
"""

import argparse
import json
import re

import networkx as nx

from config import CORPUS_DIR, EXTRACT_VERSION, CHAT_MODEL


def _corpus_dir(dataset, n):
    slug = re.sub(r"[^A-Za-z0-9_.-]", "_", CHAT_MODEL)
    return CORPUS_DIR / f"{dataset}_{n}_{EXTRACT_VERSION}_{slug}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--sample", type=int, default=100)
    args = ap.parse_args()

    d = _corpus_dir(args.dataset, args.sample)
    entities = json.loads((d / "entities.json").read_text(encoding="utf-8"))
    hyperedges = json.loads((d / "hyperedges.json").read_text(encoding="utf-8"))

    ent_by_id = {e["id"]: e for e in entities}

    # 统计每个实体被多少超边引用
    ent_degree = {e["id"]: 0 for e in entities}
    he_degree = {}
    for h in hyperedges:
        eids = h.get("entity_ids", [])
        he_degree[h["id"]] = len(eids)
        for eid in eids:
            if eid in ent_degree:
                ent_degree[eid] += 1

    iso_ent = [eid for eid, deg in ent_degree.items() if deg == 0]
    iso_he = [hid for hid, deg in he_degree.items() if deg == 0]

    print(f"实体总数: {len(entities)}, 超边总数: {len(hyperedges)}")
    print(f"孤立实体(度=0): {len(iso_ent)}")
    print(f"孤立超边(度=0): {len(iso_he)}")
    print()

    print("=== 孤立实体样例（前 30 个）===")
    for eid in iso_ent[:30]:
        e = ent_by_id[eid]
        print(f"  [{e.get('type','?')}] {e.get('name','')!r}  desc={e.get('description','')[:50]!r}")

    print()
    print("=== 孤立超边样例（前 30 个）===")
    he_by_id = {h["id"]: h for h in hyperedges}
    for hid in iso_he[:30]:
        h = he_by_id[hid]
        print(f"  summary={h.get('summary','')[:70]!r}  entity_ids={h.get('entity_ids', [])}")


if __name__ == "__main__":
    main()
