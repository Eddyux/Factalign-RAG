"""analyze_hop.py — 调试：某个锚实体能否通过「超边一跳」够到目标超边。

打印链路细节，定位桥断在哪：
  1. 锚实体的直接超边 + 各自包含的实体
  2. 目标关键词的超边 + 各自包含的实体
  3. 一跳可达集合里有没有目标超边
  4. 锚簇与目标簇的实体交集（桥）

用法:
  python analyze_hop.py --dataset 2wiki --sample 300 --anchor "Antiochus X" --target "Ptolemy"
  python analyze_hop.py --dataset 2wiki --sample 300 --anchor "Man at Bath" --target "Cahiers"
"""

import argparse
import json
import re

from corpus import get_or_build, _corpus_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--sample", type=int, required=True)
    ap.add_argument("--anchor", required=True, help="锚实体关键词（问题里的主体）")
    ap.add_argument("--target", required=True, help="目标关键词（答案相关）")
    args = ap.parse_args()

    entities, hyperedges, _chunks, _hvecs = get_or_build(args.dataset, args.sample)
    he_by = {h["id"]: h for h in hyperedges}
    eid_name = {e["id"]: e["name"] for e in entities}
    # 实体 -> 含它的超边
    ent_he: dict[str, set] = {}
    for h in hyperedges:
        for eid in h.get("entity_ids", []):
            ent_he.setdefault(eid, set()).add(h["id"])

    def _show(hes, label):
        print(f"\n== {label}: {len(hes)} 条 ==")
        for h in hes:
            names = [eid_name.get(e, "?") for e in h.get("entity_ids", [])]
            print(f"  [{', '.join(names[:5])}]  {h['summary'][:110]}")

    # 1. 锚直接超边
    anchor_hes = [h for h in hyperedges if args.anchor.lower() in h["summary"].lower()]
    _show(anchor_hes, f"锚「{args.anchor}」直接超边")

    # 锚实体集合（锚超边涉及的实体）
    anchor_eids = set()
    for h in anchor_hes:
        anchor_eids.update(h.get("entity_ids", []))
    print(f"\n锚簇实体数: {len(anchor_eids)}")
    for eid in sorted(anchor_eids, key=lambda x: eid_name.get(x, "")):
        print("   ", eid_name.get(eid, "?"))

    # 2. 目标超边
    target_hes = [h for h in hyperedges if args.target.lower() in h["summary"].lower()]
    _show(target_hes, f"目标「{args.target}」超边")
    target_eids = set()
    for h in target_hes:
        target_eids.update(h.get("entity_ids", []))

    # 3. 一跳可达：锚超边实体的全部超边
    reach = set()
    for h in anchor_hes:
        for eid in h.get("entity_ids", []):
            reach.update(ent_he.get(eid, set()))
    hit = reach & {h["id"] for h in target_hes}
    print(f"\n一跳可达超边: {len(reach)} (直接 {len(anchor_hes)} + 新增 {len(reach - {h['id'] for h in anchor_hes})})")
    print(f"目标超边是否在一跳可达内: {'✅ 是' if hit else '❌ 否'}")
    if hit:
        _show([he_by[i] for i in hit], "一跳命中的目标超边")

    # 4. 桥分析
    bridge = anchor_eids & target_eids
    print(f"\n锚簇 ∩ 目标簇 共享实体(直接桥): {len(bridge)}")
    for eid in bridge:
        print("   ", eid_name.get(eid, "?"))
    # 一跳后目标簇实体是否进锚可达（看需要几步）
    print(f"\n目标簇实体数: {len(target_eids)}")


if __name__ == "__main__":
    main()
