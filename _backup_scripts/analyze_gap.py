"""analyze_gap.py — 分析结果文件，按差距类型分类错题。

有 better（如 hipporag2）文件时：分类「better 对而你错」的题。
无 better 时：分类你自己的全部错题（同样能定位瓶颈）。

差距来源三分：
  1. recall ：答案词在语料图里但不在你的证据里（检索不够）
  2. generate：答案词在你的证据里但你答错（生成/推理不够）
  3. data   ：答案词根本不在语料图里（数据缺陷/建图漏，不算方法差距）

用法:
  python analyze_gap.py --better results/hipporag_2wiki.jsonl \
                        --mine results/test_layer_hyperedge_dense_2wiki.jsonl \
                        --dataset 2wiki --sample 300
  python analyze_gap.py --mine results/test_layer_hyperedge_dense_2wiki.jsonl \
                        --dataset 2wiki --sample 300
"""

import argparse
import json
from pathlib import Path

from corpus import get_or_build


def load_rows(path: Path) -> dict:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
            rows[r["id"]] = r
        except json.JSONDecodeError:
            pass
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--better", default="", help="更强的那个方法的 jsonl（如 hipporag2，可省略）")
    ap.add_argument("--mine", required=True, help="你的方法 jsonl")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--sample", type=int, required=True)
    args = ap.parse_args()

    mine = load_rows(Path(args.mine))
    if args.better:
        better = load_rows(Path(args.better))
        common = [i for i in better if i in mine]
        target_ids = [i for i in common
                      if better[i].get("em", 0) == 1 and mine[i].get("em", 0) == 0]
        label = f"{Path(args.better).name} 对而 {Path(args.mine).name} 错"
    else:
        target_ids = [i for i in mine if mine[i].get("em", 0) == 0]
        label = f"{Path(args.mine).name} 全部错题"

    entities, hyperedges, _ch, _h = get_or_build(args.dataset, args.sample)
    all_text = []
    for h in hyperedges:
        all_text.append(h.get("summary", "").lower())
        wt = h.get("window_text", "") or ""
        all_text.append(wt.lower())
    blob = "\n".join(all_text)

    def in_corpus(ans: str) -> bool:
        toks = [t for t in ans.lower().replace("'", "").split() if len(t) > 2]
        return sum(1 for t in toks if t in blob) >= max(1, int(len(toks) * 0.6))

    def in_evidence(ans: str, row: dict) -> bool:
        ev = json.dumps(row.get("evidence", []), ensure_ascii=False).lower()
        toks = [t for t in ans.lower().replace("'", "").split() if len(t) > 2]
        if not toks:
            return False
        return sum(1 for t in toks if t in ev) >= max(1, int(len(toks) * 0.5))

    cls = {"recall": 0, "generate": 0, "data": 0}
    rows = []
    for i in target_ids:
        r = mine[i]
        gold = r["gold"]
        in_c = in_corpus(gold)
        in_ev = in_evidence(gold, r)
        if not in_c:
            c = "data"
        elif in_ev:
            c = "generate"
        else:
            c = "recall"
        cls[c] += 1
        rows.append((c, r))

    print(f"分析: {label} -> {len(target_ids)} 题")
    for c, r in rows[:25]:
        in_c = in_corpus(r["gold"])
        in_ev = in_evidence(r["gold"], r)
        print(f"[{c:<8}] {r['question'][:65]}")
        print(f"         gold={r['gold']!r}  pred={r['pred'][:40]!r}  "
              f"in_corpus={in_c} in_evidence={in_ev}")

    print(f"\n=== 差距分类（共 {len(target_ids)} 题）===")
    for k in ("recall", "generate", "data"):
        note = {"recall": "答案在语料但不在你证据 → 检索不够",
                "generate": "答案在你证据但仍答错 → 生成/推理不够",
                "data": "答案不在语料 → 数据缺陷，不算方法差距"}[k]
        print(f"  {k:<8}: {cls[k]:<4}题  ({note})")
    if cls["recall"] + cls["generate"] + cls["data"] == 0:
        print("  （无错题）")


if __name__ == "__main__":
    main()
