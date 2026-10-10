#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""selfask `query()` 循环的**离线逐步 trace**（不联网、不调 LLM）。

用假 LLM 跑真实的 `SelfAsk.query()`，把**每次真正发给模型的 prompt**、**每跳状态**、
**state 机分支**打出来，用来核对流程是否与官方 notebook 的 `promptf()` 一致：

    cur_prompt = prompt[0] + question + prompt[1]
    ret_text = call_gpt(cur_prompt, intermediate)
    while followup in get_last_line(ret_text):
        cur_prompt += ret_text                      # ★ 官方在**循环体开头**累积
        question = extract_question(ret_text)
        external_answer = get_answer(question)
        if external_answer is not None:
            cur_prompt += intermediate + ' ' + external_answer + '.'
            ret_text = call_gpt(cur_prompt, intermediate)
        else:
            ...
    if finalans not in ret_text:
        cur_prompt += finalans
        ret_text = call_gpt(cur_prompt, '\n')

要核的几点：
  ① 每轮发给模型的 prompt 里，上一轮的 `Follow up:` + `Intermediate answer:` 是否都在
     （不累积 = 模型看不到链，多跳退化成单跳）；
  ② `ret` 在「循环外用 `or ""` 拿到初值」的写法下，有没有分支把**未经 Intermediate
     answer 的裸 ret** 拼进 prompt；
  ③ **首轮 `subq is None` 兜底分支**（模型第一句就写 `So the final answer is:`）走完后
     `ret` 是什么、下一轮循环拿它做了什么。

用法: python tools/trace_selfask_loop.py [场景]
场景: normal（默认）/ no_followup（首轮直接给最终答案）/ multi（三跳）
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import methods.selfask_adapter as sa  # noqa: E402

NL = "\n"
CALLS: list[dict] = []
RETS: list[str] = []
DEBUG = False
# few-shot 前缀长度（跑之前就按官方 prompt 算好）：官方 prompt 自带 6 处
# "Intermediate answer:"，统计「本题真实累积」必须先把前缀切掉，否则计数永远是 6+。
PREFIX_CUT = {"v": 0}


def _live(prompt: str) -> str:
    """prompt 去掉 few-shot 前缀后的「本题累积链」。"""
    return prompt[PREFIX_CUT["v"]:]


def _fake_llm(system, user, **kw):
    """假 LLM：按 prompt 里**真问题之后**已出现的 Intermediate answer 条数决定下一句。

    真实 text-davinci-002 的续写形如 `" Yes.\\nFollow up: <子问题>"`（首轮带
    `Are follow up questions needed here:` 的答案），随后各轮是 `"\\nFollow up: <子问题>"`。
    """
    marker = f"Question: {kw.get('_q', '')}"
    body = user
    n_int = body.count(sa.INTERMEDIATE + " ")
    CALLS.append({"n_int": n_int, "prompt": user, "kw": kw})
    if n_int == 0:
        return " Yes." + NL + "Follow up: What is the capital of Freedonia?"
    if n_int == 1:
        return NL + "Follow up: In what year was Freedonia founded?"
    return NL + "So the final answer is: 1904"


def _fake_llm_followup_only(system, user, **kw):
    """正常续写：每轮只写**一个** Follow up，绝不越界写最终答案（= stop 生效时的样子）。"""
    n = _live(user).count(sa.INTERMEDIATE + " ")
    CALLS.append({"n_int": n, "prompt": user, "kw": kw})
    if n == 0:
        return " Yes." + NL + "Follow up: What is the capital of Freedonia?"
    if n == 1:
        return NL + "Follow up: In what year was Freedonia founded?"
    return NL + "So the final answer is: 1904"


def _fake_llm_no_followup(system, user, **kw):
    """罕见分支：模型首轮就写最终答案（官方 few-shot 全是 Yes，所以它极少这样）。"""
    n = user.count(sa.INTERMEDIATE + " ")
    CALLS.append({"n_int": n, "prompt": user, "kw": kw})
    if n == 0:
        return " No." + NL + "So the final answer is: 1904"
    return NL + "So the final answer is: 1904"


def run(scenario: str) -> None:
    sa.llm_chat = ({"no_followup": _fake_llm_no_followup,
                    "multi": _fake_llm_followup_only}.get(scenario, _fake_llm))
    if DEBUG:
        _orig = sa.llm_chat

        def _dbg(system, user, **kw):
            r = _orig(system, user, **kw)
            RETS.append(r)
            return r
        sa.llm_chat = _dbg
    sa.generate_answer = lambda q, ev, dataset="", **k: "1904"
    sa.generate_from_context = lambda *a, **k: "Freedonia"

    m = sa.SelfAsk()
    m.dataset = "hotpotqa"
    m._chunks = [(f"Page {i}", 0, f"Page {i} is about Freedonia topic {i}.", [i])
                 for i in range(4)]
    m._texts = [c[2] for c in m._chunks]
    m._toks = [sa._tokenize(t) for t in m._texts]
    m._fewshot, _ = sa.load_official_fewshot()
    QUESTION = "When was the capital of Freedonia founded?"

    # few-shot 前缀长度 = 第一次调用的 prompt（此时本题尚未累积任何内容）
    PREFIX_CUT["v"] = len(m._fewshot.format(QUESTION) + sa.NEEDED)

    r = m.query(QUESTION)

    print(f"\n{'=' * 78}\n场景 = {scenario}\n{'=' * 78}")
    print(f"hops_used={r['hops_used']}  n_trace_rounds={len(r['trace_rounds'])}")
    print(f"sub_questions={r['sub_questions']}")
    print(f"answer={r['answer']!r}  n_evidence={len(r['evidence'])}")
    for rd in r["trace_rounds"]:
        print(f"   第{rd['round']}跳 query={rd['query_used']!r} "
              f"中间答案={rd.get('intermediate_answer')!r} n_chunks={len(rd['chunks'])}")

    print(f"\n── 真实发给模型的 prompt 共 {len(CALLS)} 次 ──")
    for k in range(len(RETS)):
        print(f"   假模型第{k}次**原样返回** = {RETS[k]!r}")
    # few-shot 前缀的尾巴：真问题之前的部分不算「本轮累积」
    for k, c in enumerate(CALLS):
        p = c["prompt"]
        cut = p.find(f"Question: {QUESTION}")
        prefix, live = p[:cut], p[cut:]
        print(f"\n[调用 {k}] prompt_chars={len(p)}  "
              f"（few-shot 前缀 {len(prefix)} + 真问题起 {len(live)}）")
        print(f"   真问题之后的累积链（换行显示）:")
        print("     " + live.replace(NL, "\n     "))

    print(f"\n── 检查 ①：调用 k+1 是否以「调用 k 的 prompt」为前缀（= 正确累积）──")
    for k in range(1, len(CALLS)):
        a, b = CALLS[k - 1]["prompt"], CALLS[k]["prompt"]
        print(f"   调用{k} ⊆ 调用{k+1}: {b.startswith(a)}")

    print(f"\n── 检查 ②：真问题之后的 Follow up / Intermediate answer 是否配对 ──")
    for k, c in enumerate(CALLS):
        p = c["prompt"]
        live = p[p.find(f"Question: {QUESTION}"):]
        n_fu = live.count("Follow up:")
        n_ia = live.count(sa.INTERMEDIATE)
        ok = (n_fu == n_ia) or (n_fu == n_ia + 1)
        print(f"   调用{k}: Follow up={n_fu}  Intermediate answer={n_ia}  "
              f"{'✅' if ok else '❌ 有裸 Follow up 没配答案'}")

    print(f"\n── 检查 ③：`ret` 的初值/兜底分支有没有漏掉累积 ──")
    p0 = CALLS[0]["prompt"]
    seg0 = p0[p0.find(f"Question: {QUESTION}"):]
    if seg0.count("Follow up:") and seg0.count(sa.INTERMEDIATE) == 0:
        print("   ❌ 调用0（首轮强制检索）之后、调用1 之前的 prompt 里 Follow up 没配答案")
    else:
        print("   ✅ 无裸 Follow up")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "normal")
