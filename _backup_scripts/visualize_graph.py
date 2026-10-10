"""visualize_graph.py — 加载语料图，检查连通性 + 可选画子图。

图是「实体 ↔ 超边」二部图。核心回答两个问题：
1. 图是不是「连成一大团」？（连通分量数、最大分量占比）
   —— 如果是 1 个大分量，说明跨题实体被大量合并（同名实体串成一片，即 #5 跨题混同）
2. 可选：画某个实体的局部邻域子图（--draw + --entity 名字）

用法:
  python visualize_graph.py --dataset hotpotqa --sample 300
  python visualize_graph.py --dataset hotpotqa --sample 300 --draw --entity "Kathmandu"
"""

import argparse
import json
import re

import networkx as nx

from config import CORPUS_DIR, EXTRACT_VERSION, CHAT_MODEL


def _corpus_dir(dataset, n):
    slug = re.sub(r"[^A-Za-z0-9_.-]", "_", CHAT_MODEL)
    return CORPUS_DIR / f"{dataset}_{n}_{EXTRACT_VERSION}_{slug}"


def load_graph(dataset, n):
    d = _corpus_dir(dataset, n)
    entities = json.loads((d / "entities.json").read_text(encoding="utf-8"))
    hyperedges = json.loads((d / "hyperedges.json").read_text(encoding="utf-8"))
    return entities, hyperedges


def build_bipartite(entities, hyperedges):
    G = nx.Graph()
    for e in entities:
        G.add_node("ent:" + e["id"], kind="entity", label=e.get("name", ""))
    for h in hyperedges:
        G.add_node("he:" + h["id"], kind="hyperedge", label=h.get("summary", "")[:40])
        for eid in h.get("entity_ids", []):
            G.add_edge("he:" + h["id"], "ent:" + eid)
    return G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--sample", type=int, default=300)
    ap.add_argument("--draw", action="store_true", help="画子图")
    ap.add_argument("--entity", type=str, default="", help="画这个实体的邻域（默认画最大分量的子集）")
    args = ap.parse_args()

    entities, hyperedges = load_graph(args.dataset, args.sample)
    G = build_bipartite(entities, hyperedges)

    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    n_node = len(G.nodes())
    n_edge = len(G.edges())

    print(f"总节点: {n_node}  (实体 {len(entities)} + 超边 {len(hyperedges)})")
    print(f"总边: {n_edge}")
    print(f"连通分量数: {len(comps)}")
    print(f"最大分量: {len(comps[0])} 节点 ({len(comps[0])/n_node*100:.1f}%)")
    if len(comps) > 1:
        sizes = [len(c) for c in comps]
        print(f"第2大: {sizes[1]} 节点")
        print(f"孤立节点(大小=1)的分量数: {sum(1 for s in sizes if s == 1)}")

    if len(comps) == 1:
        print("\n>>> 整张图连成一个大分量 —— 实体被大量跨题合并，存在跨题混同")
    elif len(comps[0]) / n_node > 0.8:
        print(f"\n>>> 最大分量占 {len(comps[0])/n_node*100:.0f}% —— 大部分连在一起，跨题混同较严重")
    else:
        print("\n>>> 图较分散，多个分量 —— 实体合并没那么激进")

    if args.draw:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if args.entity:
            # 找该实体节点，画它的 2 跳邻域
            seed = None
            for e in entities:
                if e.get("name", "").lower() == args.entity.lower():
                    seed = "ent:" + e["id"]
                    break
            if seed is None:
                print(f"找不到实体 {args.entity!r}")
                return
            sub = nx.ego_graph(G, seed, radius=1)
            title = f"邻域: {args.entity} ({len(sub.nodes())} 节点)"
        else:
            biggest = comps[0]
            if len(biggest) > 60:
                biggest = set(list(biggest)[:60])
            sub = G.subgraph(biggest)
            title = f"最大分量子集 ({len(sub.nodes())} 节点)"

        pos = nx.spring_layout(sub, seed=42, k=0.8)
        colors = ["#4C9AFF" if G.nodes[n]["kind"] == "entity" else "#FFB74D"
                  for n in sub.nodes()]
        plt.figure(figsize=(14, 10))
        nx.draw_networkx(sub, pos, node_color=colors, node_size=120,
                         edge_color="#ddd", with_labels=False, width=0.4)
        # 只标实体名
        labels = {n: G.nodes[n]["label"] for n in sub.nodes()
                  if G.nodes[n]["kind"] == "entity"}
        nx.draw_networkx_labels(sub, pos, labels, font_size=7)
        plt.title(title)
        plt.axis("off")
        out = f"graph_{args.dataset}.png"
        plt.savefig(out, dpi=120, bbox_inches="tight")
        print(f"图已存: {out}")


if __name__ == "__main__":
    main()
