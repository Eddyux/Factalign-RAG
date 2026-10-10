"""No-Retrieval 基线（zero-shot LLM）：不给任何检索证据，直接让模型凭参数知识回答。

用途：证明检索环节本身的价值（下界基线）。作答**格式与统一 RAG prompt 一致**
（先一行 Thought、再 `Answer:` 行给最短完整答案）、生成器/温度/关思考全部同口径，
唯一差别是 evidence 为空 —— 这样它与有检索的方法之间只差「检索」这一个变量。

prompt 在 `prompts/rag_prompt_nocontext.txt`（走 `methods.base` 的统一加载，
与另外两份带证据的 prompt 放在一起，改动即生效、且 manifest 能查到用的是哪份）。
★ 它**不按数据集分派**：单跳/多跳口径讲的是"从证据里抄哪个 span、抄多长"，
无证据时没有 span 可抄 —— 所以这里只有"有证据/无证据"一个维度，数据集维度对它不存在。

⚠️ 这里**刻意不禁止 "unknown"**（旧版要求 BEST GUESS）：与统一 prompt 一样，不确定时
应当诚实输出 unknown，否则下界基线会因为"蒙对"而虚高，反而掩盖检索的价值。

用法（经 run.py，per-question 路径，不需要语料）:
  python run.py --method zeroshot --dataset 2wiki --sample 300 --workers 16
"""

from methods.base import (BaseMethod, ANSWER_SYSTEM, NO_RAG_PROMPT,
                          extract_final_answer)
from common.llm import chat as llm_chat, strip_thinking


class NoRetrieval(BaseMethod):
    name = "zeroshot"

    def query(self, question: str, dataset: str = "") -> dict:
        # `dataset` 只为了随调用方显式传参而保留：无证据路径**不分数据集**，
        # prompt 恒定取 NO_RAG_PROMPT（见模块 docstring）。
        _ = dataset
        out = llm_chat(ANSWER_SYSTEM, NO_RAG_PROMPT.format(question=question),
                       json_mode=False, timeout=120, thinking=False)
        if out is None:
            return {"answer": "unknown", "evidence": [], "evidence_locs": [], "entities": []}
        text = out if isinstance(out, str) else out.get("text", "")
        text = strip_thinking(text)
        answer = extract_final_answer(text)
        return {"answer": answer if answer else "unknown",
                "evidence": [], "evidence_locs": [], "entities": []}

    def run(self, question, contexts):
        """无检索：contexts 不参与，等价于 query()。"""
        return self.query(question, dataset=getattr(self, "dataset", ""))