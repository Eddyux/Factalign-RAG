"""draw_graph.py — 用 networkx 画语料图的局部子图（实体邻域）。

networkx 自带绘图（底层 matplotlib）。全图 3 万节点画不了，所以只画子图。

用法:
  python draw_graph.py --dataset hotpotqa --sample 300 --entity "Kathmandu"
  python draw_graph.py --dataset hotpotqa --sample 300 --top 60   # 画最大分量前 60 节点
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
    ap.add_argument("--entity", type=str, default="", help="画这个实体的邻域")
    ap.add_argument("--top", type=int, default=0, help="画最大分量前 N 节点（缺省 entity 时用）")
    ap.add_argument("--out", type=str, default="", help="输出文件名")
    args = ap.parse_args()

    entities, hyperedges = load_graph(args.dataset, args.sample)
    G = build_bipartite(entities, hyperedges)

    if args.entity:
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
        comps = sorted(nx.connected_components(G), key=len, reverse=True)
        biggest = comps[0]
        n = args.top or min(60, len(biggest))
        biggest = set(list(biggest)[:n])
        sub = G.subgraph(biggest)
        title = f"最大分量子集 ({len(sub.nodes())} 节点)"

    pos = nx.spring_layout(sub, seed=42, k=0.8)
    ent_nodes = [n for n in sub.nodes() if G.nodes[n]["kind"] == "entity"]
    he_nodes = [n for n in sub.nodes() if G.nodes[n]["kind"] == "hyperedge"]

    nx.draw_networkx_nodes(sub, pos, nodelist=ent_nodes, node_color="#4C9AFF",
                           node_size=150)
    nx.draw_networkx_nodes(sub, pos, nodelist=he_nodes, node_color="#FFB74D",
                           node_size=80)
    nx.draw_networkx_edges(sub, pos, edge_color="#ccc", width=0.5)
    labels = {n: G.nodes[n]["label"] for n in ent_nodes}
    nx.draw_networkx_labels(sub, pos, labels, font_size=7)

    import matplotlib.pyplot as plt
    plt.title(title)
    plt.axis("off")
    out = args.out or f"graph_{args.dataset}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"已存: {out}")


if __name__ == "__main__":
    main()
