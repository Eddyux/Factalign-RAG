"""数据集体检：答案到底在不在语料里（假阴性率）。

为什么需要：语料级协议下，如果某题的标准答案根本不在合并语料中，**任何方法都答不对**
→ 该数据集的 EM/F1 上限被语料锁死，方法间差异被压缩，不能用来比方法。
（历史结论：popqa / triviaqa_gr1 就是这样被废弃的。）

判定：把该题所有上下文拼成小写文本，看 golden answer 是否为子串（多答案时分别统计）。

用法:
  python tools/check_answer_in_corpus.py                      # 体检常用数据集
  python tools/check_answer_in_corpus.py triviaqa_gr1_128 hotpotqa_300
"""

import json
import pathlib
import sys

SAMPLES = pathlib.Path(__file__).resolve().parent.parent / "data" / "samples"
DEFAULT = ["triviaqa_gr1_128", "nq_gr1_128", "popqa_gr1_128", "hotpotqa_gr1_128",
           "2wiki_gr1_128", "musique_gr1_128", "hotpotqa_300"]


def items_of(d) -> list:
    """样本文件结构有几种写法，挨个探测。"""
    if isinstance(d, list):
        return d
    for k in ("data", "questions", "items", "samples"):
        if isinstance(d.get(k), list):
            return d[k]
    for v in d.values():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            return v
    return []


def ctx_blob(ctxs) -> str:
    parts = []
    for c in ctxs or []:
        if isinstance(c, dict):
            parts.append(str(c.get("text", "")))
            sents = c.get("sentences") or []
            parts.append(" ".join(str(s) for s in sents))
        else:
            parts.append(str(c))
    return " ".join(parts).lower()


def check(name: str) -> str:
    p = SAMPLES / f"{name}.json"
    if not p.exists():
        return f"{name:20s} 缺文件 {p.name}"
    items = items_of(json.loads(p.read_text(encoding="utf-8")))
    if not items:
        return f"{name:20s} 结构未识别（顶层键: {list(json.loads(p.read_text(encoding='utf-8')))[:6]}）"
    n = hit_any = hit_all = no_ans = 0
    for it in items:
        ans = it.get("golden_answers") or it.get("answers") or it.get("answer") or []
        if isinstance(ans, str):
            ans = [ans]
        ans = [str(a) for a in ans if str(a).strip()]
        if not ans:
            no_ans += 1
            continue
        n += 1
        blob = ctx_blob(it.get("contexts") or it.get("context") or [])
        hits = [a for a in ans if a.lower() in blob]
        hit_any += bool(hits)
        hit_all += len(hits) == len(ans)
    if not n:
        return f"{name:20s} 无可判题（缺 golden answer）"
    return (f"{name:20s} n={n:4d}  答案(任一)在语料内 {hit_any / n:6.1%}"
            f"   全部答案都在 {hit_all / n:6.1%}   假阴性 {1 - hit_any / n:6.1%}"
            + (f"   [无答案字段 {no_ans}]" if no_ans else ""))


def main() -> int:
    names = sys.argv[1:] or DEFAULT
    for nm in names:
        print(check(nm))
    return 0


if __name__ == "__main__":
    sys.exit(main())
