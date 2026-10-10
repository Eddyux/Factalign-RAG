"""统一评测指标：EM / ContainsMatch / F1 / R@k。

★ 打分口径 = **HippoRAG 2 / S2G-RAG / SubQRAG / SQuAD 官方**（四家逐字一致），不是本项目自创：

  1. `normalize_answer` 与下面三份**逐字相同**（源头都是 SQuAD 官方 `evaluate-v2.0.py`）：
       · `baselines/HippoRAG-main/src/hipporag/utils/eval_utils.py`
       · `baselines/S2G-RAG-main/utils/evaluation.py`
       · `baselines/SubQRAG-main/eval/query_eval.py`
     口径：lower → 删 `string.punctuation`（**含 `_` 与 `-`**）→ 删 a/an/the → 压空白。
     注：早期版本用 `re.sub(r"[^\\w\\s]", "", s)` 归一化，`\\w` 会把下划线留下、
     把 `-` 当普通字符删掉（`Weston-super-Mare` → `westonsupermare`）；已废弃。
  2. EM = 归一化后**整串相等**；F1 = 归一化后按空白切 token，`Counter` **多重集**交集
     （不是 `set`），照抄 HippoRAG 2 `evaluation/qa_eval.py::QAF1Score.compute_f1`
     = S2G-RAG `utils/evaluation.py::compute_f1_multiple`。
  3. 多答案：`*_aliases` 把 gold + `answer_aliases` 当成 **gold 列表**，逐条打分取 **max**
     （HippoRAG 2 的 `aggregation_fn=np.max`）。TriviaQA 每题中位 11 个别名，
     这正是官方「接受答案集」的口径。
  4. F1 的 yes/no 守卫（官方 `hotpot_evaluate_v1.py::f1_score` 同款）：
     gold 或 pred 任一方是 yes/no/noanswer 而两者不相等 → 整题 F1=0，**不给 token 重叠分**。
     不这么写，"no idea" 这种答案对 `gold=no` 会白拿 0.67，恰好掩盖 yes/no 答反。
     ⚠️ 官方对 EM 不加这个守卫（EM 本来就是整串相等，加了也不改变结果）。

⚠️ 换口径会让**已有 results/*.jsonl 的 em/f1 列按旧口径**，与新行不可直接混比：
   重跑或按新口径重算一遍再画表。

`token_f1` / `recall_at_k` / `precision_at_k` 是**检索侧**指标（对齐 hotpotqa
`_token_f1` 与 supporting-facts 命中），与本文件上半部分的**答案侧**口径无关，未改动。
"""

import re
import string
from collections import Counter

# yes/no 型答案：官方视为**整体判定**而非 token 重叠
YES_NO_TOKENS = {"yes", "no", "noanswer"}


def normalize_answer(s: str) -> str:
    """SQuAD / HippoRAG 2 官方归一化（逐字对齐，勿改）。"""
    s = (s or "").lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def normalize(s: str) -> str:
    """`normalize_answer` 的别名（旧调用点兼容）。"""
    return normalize_answer(s)


def _tokens(s: str) -> list:
    """F1 的 token 序列：归一化后按空白切（**list，保留重复**；多重集交集的输入）。"""
    return normalize_answer(s).split()


def exact_match(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)


def contains_match(pred: str, gold: str) -> bool:
    n_gold = normalize_answer(gold)
    return bool(n_gold) and n_gold in normalize_answer(pred)


def f1_score(pred: str, gold: str) -> float:
    """HippoRAG 2 / SQuAD 官方 F1（+ HotpotQA 的 yes/no 守卫）。"""
    n_pred, n_gold = normalize_answer(pred), normalize_answer(gold)
    # yes/no 守卫：答错就整题 0，不给部分分
    if n_pred in YES_NO_TOKENS and n_pred != n_gold:
        return 0.0
    if n_gold in YES_NO_TOKENS and n_pred != n_gold:
        return 0.0
    gold_tokens, pred_tokens = n_gold.split(), n_pred.split()
    if not gold_tokens or not pred_tokens:
        # 官方 SQuAD：任一方为空 → 1.0 当且仅当两边都空
        return 1.0 if gold_tokens == pred_tokens else 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)          # 多重集，不是 set
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _gold_list(gold: str, aliases=None) -> list:
    return [str(g).strip() for g in ([gold] + list(aliases or [])) if str(g).strip()]


def exact_match_aliases(pred: str, gold: str, aliases=None) -> bool:
    """EM：命中 gold 或任一别名（= HippoRAG 2 对 gold 列表取 max）。"""
    golds = _gold_list(gold, aliases)
    if not golds:
        return exact_match(pred, gold)
    return any(exact_match(pred, g) for g in golds)


def f1_score_aliases(pred: str, gold: str, aliases=None) -> float:
    """F1：pred 对 gold + aliases 逐条打分，取 max（HippoRAG 2 的 np.max 聚合）。"""
    golds = _gold_list(gold, aliases)
    if not golds:
        return f1_score(pred, gold)
    return max(f1_score(pred, g) for g in golds)


# ── 检索侧指标（答案侧口径无关，未改） ──

def token_f1(a: str, b: str) -> float:
    """token 重叠 F1（对齐 hotpotqa _token_f1，检索 P@k/R@k 文本匹配用，阈值 0.6）。

    注意：这里刻意用 `set`（检索侧口径），与答案侧 `f1_score` 的 `Counter` 不同。
    """
    ta = set(re.findall(r"\w+", a.lower()))
    tb = set(re.findall(r"\w+", b.lower()))
    if not ta or not tb:
        return 0.0
    common = ta & tb
    if not common:
        return 0.0
    p = len(common) / len(ta)
    r = len(common) / len(tb)
    return 2 * p * r / (p + r)


def recall_at_k(evidence_locs: list, supporting_facts: list) -> float:
    """evidence_locs: 证据对应的 (title, sent_idx) 集合；supporting_facts: [(title, sent_idx)]。"""
    if not supporting_facts:
        return 0.0
    gold = set((t.strip(), int(s)) for t, s in supporting_facts)
    hits = len(set(evidence_locs) & gold)
    return hits / len(gold)


def precision_at_k(evidence_locs: list, supporting_facts: list) -> float:
    if not evidence_locs:
        return 0.0
    gold = set((t.strip(), int(s)) for t, s in supporting_facts)
    hits = len(set(evidence_locs) & gold)
    return hits / len(set(evidence_locs))