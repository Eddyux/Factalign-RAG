"""Self-Ask 适配器 —— **官方算法复现**（Press et al., Findings of EMNLP 2023），接本项目语料。

论文: "Measuring and Narrowing the Compositionality Gap in Language Models"
      (Press, Zhang, Min, Schmidt, Smith, Lewis; Findings of EMNLP 2023)
官方代码: https://github.com/ofirpress/self-ask  →  `baselines/self-ask-main/`
（该仓库只提供 **一个 Colab notebook**：`self-ask_plus_search-engine_demo.ipynb`，
  没有可 import 的 .py 模块；本适配器按 notebook 逐行对齐，官方 prompt 也在 notebook 里。）

═══ 官方算法（notebook `promptf` / `call_gpt` / `google` / `extract_*`）═══
  1. **提示** = 4-shot 前缀（notebook cell 6，逐字：4 个
     `Question/Are follow up questions needed here/Follow up/Intermediate answer/
      So the final answer is:` 例子）+ `{原问题}` + `\nAre follow up questions needed here:`
     （`prompt[0] + question + prompt[1]`）。
  2. **每轮**：让模型续写，`stop=["\\nIntermediate answer:"]`（`call_gpt(cur_prompt, intermediate)`）
     → 输出形如 ` Yes.\nFollow up: <子问题>`（模型自己先判「是否需要追问」）。
  3. **最后一行为 `Follow up:`** → `extract_question` 取 `:` 后、去掉尾句点的子问题，
     交给**搜索引擎**取回 `external_answer`（官方 `google()` / SerpAPI）。
  4. 把 `\nIntermediate answer: {external_answer}.` 拼回 prompt，**回到第 2 步**继续生成。
     ——「搜索引擎给的答案」是模型生成下一条追问的唯一依据（ofir.io/self-ask 的
     search-engine demo 就是这个 loop）。
  5. 最后一行为 `So the final answer is:`（不再有 Follow up）→ `extract_answer` 取答案；
     若模型没写，官方补一句 `\nSo the final answer is:` 再让模型生成一次。
  6. 官方无显式轮数上限（靠模型自己停）；`max_tokens=256`、`temperature=0`。

═══ 本项目怎么接（论文里必须写明的适配取舍）═══
  - **搜索引擎 → 本项目语料**（唯一必要的替换）：官方 `google(query)` 换成
    「在 merged 语料上检索」。**检索底座逐字复用 `standardrag`**：
    `standard_rag.build_chunks`（默认**不切块**：0 → CHUNK_NO_SPLIT=2000、overlap 0，
     bge tokenizer）+ 纯 BM25
    （`k1=1.5, b=0.75`），不建向量、不调 embedding —— Self-Ask 原版就没有稠密检索，
    这样每题只花 LLM token（成本表：嵌入 0）。跨轮用「已取过的块」做 exclude，
    与 IRCoT/S2G-RAG 的累积去重同义。
  - **`Intermediate answer` 由谁写**：官方是搜索引擎的 snippet **原文**（不含推理），
    而那一步**不调 LLM**（`def get_answer(question): return google(question)`）。
    本项目默认 `SELFASK_SNIPPET=snippet` → 从命中块里本地挑**最相关的一句**当 snippet
    （`_snippet`：按子问题对块内句子做 BM25 选句），**零 LLM 调用**，与官方同构；
    `SELFASK_SNIPPET=llm` 则退回"把检索段落喂 LLM 改写成短答案"（每题多 1 次/跳）；
    `SELFASK_GROUND_INTERMEDIATE=0` 则退化成"让模型自己按上文回答该子问题"
    （= 原版 self-ask+search 里搜索引擎无结果时的分支，也等于常见 self-ask 实现），
    这两档要作为消融/差异说明写进论文。
    喂给 `_snippet` 的是**累积**命中块（不是本轮新块）：官方每轮把新段落 `+=` 进
    prompt，跨跳指代（"his birthplace"）要靠前几跳的段落才成立。
  - **最终作答**：默认 `SELFASK_FINAL_GROUNDED=1` —— 用**累积的检索段落**走统一
    `generate_answer`（与其余 baseline 同生成口径，防止模型凭记忆编答案）；
    设 0 则逐字用官方 `So the final answer is:` 的模型输出。
  - **不做 `remove_wh_words` 等改动**（那是 IRCoT 官方的），Self-Ask 的查询就是
    模型写出的子问题原样。
  - **首轮强制检索**（与 S2G-RAG 的 `should_force_first_retrieval` 同思路）：
    4-shot prompt 全是 "Yes"，模型极少数情况下会直接给最终答案；不给检索证据的
    greedy 答案与本项目「答案必须来自语料」的口径冲突 → 证据为空且没反问过时，
    强制先检索原问题一轮再继续。这一轮**没有 `Follow up:` 行**，所以：
      · 模型这一轮的原话仍按官方 `cur_prompt += ret_text` 拼进 prompt；
      · 检索结果**不**写进 `Intermediate answer: X.`，而是用官方「搜索引擎无结果」的
        写法 `\nIntermediate answer:`（只有提示、没有内容）—— 否则 prompt 里会出现
        一个**没有对应 Follow up 的悬空答案**；
      · `trace_rounds[0].query_used` 记**真正检索的那句话**（= 原问题），并标
        `forced_query=true`，否则审计看不到这跳在检索什么。
  - 成本：每题 LLM 调用 = 1（首次续写）+ 每轮 1（续写；**中间答案默认不调 LLM**）
    + 1（最终作答，grounded 时，且模型没自己写出答案时才多一次收尾续写）
    ≈ **2~12 次**（`SELFASK_MAX_HOPS` 上限 10，实际由模型写到
    `So the final answer is:` 自己停），与 IRCoT(2~4) 同量级；检索是本地纯 BM25，
    **embedding 调用 0**。设 `SELFASK_SNIPPET=llm` 则每轮再多 1 次。

用法:
  python run.py --method selfask --dataset hotpotqa --sample 300 --workers 8
  旋钮（环境变量，无需改代码）:
    SELFASK_MAX_HOPS=10           最多几次追问（官方 notebook **无上限**；10 = 实际不约束、仅防跑飞）
    SELFASK_BM25_K=1              每个子问题取几个块（= 官方每跳 1 条，见 _search 注释）
    SELFASK_GROUND_INTERMEDIATE=1 中间答案是否由检索结果决定（0=模型自答）
    SELFASK_SNIPPET=snippet       中间答案怎么来：snippet（默认，本地选句、零 LLM）/ llm
    SELFASK_SNIPPET_FALLBACK=1    选不出有词面重合的句子时是否退回 LLM（0=严格官方口径）
    SELFASK_FINAL_GROUNDED=1      最终答案用统一 generate_answer（0=官方模型输出）
    SELFASK_LLM_TIMEOUT=300       续写/中间作答的单次超时
    SELFASK_MAX_NEW_TOKENS=256    官方 call_gpt 的 max_tokens（text-davinci-002，temperature=0）
    SELFASK_BM25_K1=1.5 / _B=0.75 BM25 打分（官方未开源检索器，本项目自定，见下方注释）
"""

import os
import re
from pathlib import Path

from methods.base import BaseMethod, generate_answer, generate_from_context
from methods.standard_rag import build_chunks, _bm25, _tokenize
from common.llm import chat as llm_chat, role as llm_role
from config import BASELINES_DIR

# ── 官方超参 / 旋钮 ──
SELFASK_MAX_HOPS = int(os.environ.get("SELFASK_MAX_HOPS", "10"))
SELFASK_BM25_K = int(os.environ.get("SELFASK_BM25_K", "1"))
SELFASK_GROUND_INTERMEDIATE = os.environ.get("SELFASK_GROUND_INTERMEDIATE", "1") == "1"
SELFASK_FINAL_GROUNDED = os.environ.get("SELFASK_FINAL_GROUNDED", "1") == "1"
# ★ 中间答案怎么来（对齐官方 `get_answer` = 搜索引擎 snippet，**不调 LLM**）：
#   "snippet"（默认）= 从命中块里本地挑出最相关的一句当 snippet 直接拼进 prompt；
#   "llm"            = 旧行为，额外调一次 LLM 把检索段落改写成中间答案（每题多 1 次/跳）。
#   为什么默认改成 snippet：官方那一步是 `return google(question)`，**零 LLM 调用**；
#   我们原来多调一次，导致 selfask 的 LLM 调用数约为 ircot 的 2 倍（用户实测"selfask 很慢"）。
#   官方唯一的 LLM 调用 `call_gpt` 一律 `max_tokens=256`；本方案把那一跳整体省掉。
SELFASK_SNIPPET = os.environ.get("SELFASK_SNIPPET", "snippet").strip().lower()
#   snippet 全无词面重合时是否退回 LLM（"snippet 没含答案"的兜底）。0 = 严格官方口径。
SELFASK_SNIPPET_FALLBACK = os.environ.get("SELFASK_SNIPPET_FALLBACK", "1") == "1"
# ★ BM25 打分参数：**官方没有可对齐的值** —— `baselines/self-ask-main/` 里只有一个
#   调 Google Search API 的 demo notebook，**没有任何检索/BM25 代码**（论文里的
#   BM25-Wikipedia 检索器没开源）。所以这里是**项目自定的声明值**，不是官方口径。
#   取 1.5/0.75 = `rank_bm25.BM25Okapi` 的库默认（该论文常用实现）。
#   显式声明而不落回全局默认：这样改全局 `BM25_K1` 不会静默改掉这个基线。
SELFASK_BM25_K1 = float(os.environ.get("SELFASK_BM25_K1", "1.5"))
SELFASK_BM25_B = float(os.environ.get("SELFASK_BM25_B", "0.75"))
SELFASK_LLM_TIMEOUT = int(os.environ.get("SELFASK_LLM_TIMEOUT", "300"))
SELFASK_MAX_NEW_TOKENS = int(os.environ.get("SELFASK_MAX_NEW_TOKENS", "256"))

# 官方 notebook / 仓库位置（只用于溯源；**缺文件不影响运行**，见 load_official_fewshot）
SELFASK_REPO_CANDIDATES = [BASELINES_DIR / "self-ask-main", BASELINES_DIR / "self-ask"]
SELFASK_NOTEBOOK = "self-ask_plus_search-engine_demo.ipynb"

# ── 官方 prompt 分隔串（notebook `promptf` 的默认参数，逐字）──
FOLLOWUP = "Follow up:"
FINALANS = "\nSo the final answer is:"
INTERMEDIATE = "\nIntermediate answer:"
NEEDED = "\nAre follow up questions needed here:"

# ── 官方 4-shot 前缀（notebook cell 6 的 `prompt[0]`，逐字）──
# 装载时若官方 notebook 在，用 notebook 里的那份覆盖（`{}` 是待填问题占位，官方是字符串拼接）
SELFASK_FEWSHOT = """Question: Who lived longer, Muhammad Ali or Alan Turing?
Are follow up questions needed here: Yes.
Follow up: How old was Muhammad Ali when he died?
Intermediate answer: Muhammad Ali was 74 years old when he died.
Follow up: How old was Alan Turing when he died?
Intermediate answer: Alan Turing was 41 years old when he died.
So the final answer is: Muhammad Ali 

Question: When was the founder of craigslist born?
Are follow up questions needed here: Yes.
Follow up: Who was the founder of craigslist?
Intermediate answer: Craigslist was founded by Craig Newmark.
Follow up: When was Craig Newmark born?
Intermediate answer: Craig Newmark was born on December 6, 1952.
So the final answer is: December 6, 1952

Question: Who was the maternal grandfather of George Washington?
Are follow up questions needed here: Yes.
Follow up: Who was the mother of George Washington?
Intermediate answer: The mother of George Washington was Mary Ball Washington.
Follow up: Who was the father of Mary Ball Washington?
Intermediate answer: The father of Mary Ball Washington was Joseph Ball.
So the final answer is: Joseph Ball 

Question: Are both the directors of Jaws and Casino Royale from the same country? 
Are follow up questions needed here: Yes. 
Follow up: Who is the director of Jaws? 
Intermediate Answer: The director of Jaws is Steven Spielberg. 
Follow up: Where is Steven Spielberg from? 
Intermediate Answer: The United States. 
Follow up: Who is the director of Casino Royale? 
Intermediate Answer: The director of Casino Royale is Martin Campbell. 
Follow up: Where is Martin Campbell from? 
Intermediate Answer: New Zealand. 
So the final answer is: No

Question: {}"""

# 中间作答的 system：官方这里是「搜索引擎」，不是推理器 —— 只允许照抄检索段落
_INTERMEDIATE_SYSTEM = (
    "You are a search engine. Answer the follow-up question with ONLY the exact "
    "answer stated in the retrieved passages. Do not explain, do not reason, do not "
    "use outside knowledge. If the passages do not state it, output nothing."
)

_PROMPT_CACHE: dict = {"value": None, "note": ""}


def _strip_after_colon(text: str) -> str:
    """官方 `extract_answer`/`extract_question` 的公共部分：取最后一个 ':' 之后、去掉首空格与尾句点。"""
    after = text.split(":")[-1] if ":" in text else text
    if after.startswith(" "):
        after = after[1:]
    if after.endswith("."):
        after = after[:-1]
    return after


def _last_line(text: str) -> str:
    """官方 `get_last_line`：不含 '\\n' 就是整串，否则最后一行。"""
    return text if "\n" not in text else text.split("\n")[-1]


def extract_followup(generated: str) -> str | None:
    """官方 `extract_question`：最后一行的 `Follow up:` 之后就是子问题。

    只认**最后一行**（与官方 `get_last_line` + `'Follow up:' not in last_line` 一致）。
    注意：模型一轮可能同时输出 `Follow up: <子问题>` **和** `Intermediate answer: ...`
    两行（官方 few-shot 里每问一次就自答一次），这时最后一行是 Intermediate answer
    —— **仍要取上一行的 Follow up**，否则 loop 走一跳就停。所以这里在整段文本里
    取**最后一个** Follow up 行（顺序上等于「刚才那个子问题」）。
    """
    text = generated or ""
    if FOLLOWUP not in text:
        return None
    line = next((l for l in reversed(text.split("\n")) if FOLLOWUP in l), "")
    q = _strip_after_colon(line).strip()
    return q or None


def extract_final(generated: str) -> str | None:
    """官方 `extract_answer`：`So the final answer is:` 之后到行尾。"""
    if not generated:
        return None
    if ":" not in generated:
        line = _last_line(generated)
    else:
        line = generated.split(":")[-1]
    ans = _strip_after_colon(line).strip()
    return ans or None


def load_official_fewshot() -> tuple[str, str]:
    """读官方 notebook cell 6 的 4-shot prompt；读不到就用内置逐字副本。

    返回 (few-shot 文本, 来源说明)。**不抛异常**：官方仓库只有 notebook，
    它是「口径溯源」而不是运行依赖（本项目所有 adapter 的 prompt 都必须能溯源）。
    """
    if _PROMPT_CACHE["value"] is not None:
        return _PROMPT_CACHE["value"], _PROMPT_CACHE["note"]
    text, note = SELFASK_FEWSHOT, "内置逐字副本（官方 notebook cell 6）"
    for repo in SELFASK_REPO_CANDIDATES:
        nb = Path(repo) / SELFASK_NOTEBOOK
        if not nb.exists():
            continue
        try:
            import json
            cells = json.loads(nb.read_text(encoding="utf-8")).get("cells", [])
            for cell in cells:
                src = "".join(cell.get("source") or [])
                if "Question: Who lived longer, Muhammad Ali or Alan Turing?" not in src:
                    continue
                m = re.search(r"'''(.*?)'''", src, re.DOTALL)
                if not m:
                    continue
                body = m.group(1).rstrip()
                # 官方 prompt[0] 以 "Question: " 结尾（等拼问题）→ 换成 {question} 占位
                if body.endswith("Question:"):
                    body = body[: -len("Question:")].rstrip() + "\nQuestion: {}"
                text = body
                note = f"官方 notebook: {nb.relative_to(BASELINES_DIR.parent)}"
                break
            if note.startswith("官方"):
                break
        except Exception as e:                      # notebook 坏了不该拖垮实验
            note = f"⚠ 读官方 notebook 失败（{type(e).__name__}: {e}）→ 用内置逐字副本"
    _PROMPT_CACHE["value"], _PROMPT_CACHE["note"] = text, note
    return text, note


class SelfAsk(BaseMethod):
    """Self-Ask 复现（语料级：build() 建块索引，query() 逐题跑追问 loop）。"""

    name = "selfask"

    def __init__(self, max_hops: int = SELFASK_MAX_HOPS,
                 bm25_k: int = SELFASK_BM25_K,
                 ground_intermediate: bool = SELFASK_GROUND_INTERMEDIATE,
                 final_grounded: bool = SELFASK_FINAL_GROUNDED):
        self.dataset = ""
        self.max_hops = max_hops
        self.bm25_k = bm25_k
        self.ground_intermediate = ground_intermediate
        self.final_grounded = final_grounded
        self._fewshot = SELFASK_FEWSHOT
        self._fewshot_note = ""
        self._chunks: list[tuple] = []          # build_chunks 的输出 (title, start, text, locs)
        self._toks: list[list[str]] = []
        self._texts: list[str] = []

    # ── 语料级准备：按数据集口径切块（triviaqa 1200/100，其余整段）+ BM25
    #    （与 standardrag 同底座、同口径）──
    def build(self, corpus_contexts: list[dict], working_dir: str = "") -> "SelfAsk":
        self._chunks = build_chunks(corpus_contexts, dataset=getattr(self, "dataset", ""))
        self._texts = [c[2] for c in self._chunks]
        self._toks = [_tokenize(t) for t in self._texts]
        self._fewshot, self._fewshot_note = load_official_fewshot()
        print(f"[selfask] 检索底座 {len(self._chunks)} 块（整段，与 standardrag 同口径）；"
              f"纯 BM25 top-{self.bm25_k}/跳，max_hops={self.max_hops}；"
              f"中间答案={('检索句 snippet（官方口径，不调 LLM）' if (self.ground_intermediate and SELFASK_SNIPPET != 'llm') else ('检索段落+LLM 改写' if self.ground_intermediate else '模型自答'))}；"
              f"最终作答={'统一 generate_answer' if self.final_grounded else '官方模型输出'}；"
              f"{self._fewshot_note}", flush=True)
        return self

    # ── 官方 `google(query)` 的语料版：BM25 top-k，排除已取过的块 ──
    # ⚠️ 官方**每次只喂 1 条**（notebook cell 10 的 `google()`）：优先 Google `answer_box`，
    #    否则取 `organic_results[0]['snippet']` —— **没有 num / top-k 参数**，第 2 条起不用。
    #    我们 K=**1** 对齐这个**条数**（检索侧 = §0 第③层，该逐方法对齐官方）。
    #    但每跳的**文本量**做不到等价：官方 snippet ≈30-50 词，我们一块 ≈500 词（默认整段
    #    passage 一块，不切块）→ 我们仍是官方的 **10~17 倍**（差异落在对基线有利的那一侧）。
    #    （原默认 5 = 官方的 30~50 倍，属未消掉的不对称，已改。）
    def _search(self, query: str, exclude: list) -> list[int]:
        if not self._texts or not (query or "").strip():
            return []
        scores = _bm25(query, self._toks, k1=SELFASK_BM25_K1, b=SELFASK_BM25_B)
        order = sorted((i for i in range(len(self._texts)) if i not in exclude),
                       key=lambda i: -scores[i])
        return order[:self.bm25_k]

    def _context(self, hits: list[int]) -> str:
        """本轮检索结果拼成 prompt 上下文（编号与 generate_answer 同形）。"""
        return "\n\n".join(f"[{j+1}] {self._texts[i]}" for j, i in enumerate(hits))

    def _snippet(self, subq: str, hits: list[int]) -> str:
        """官方 `get_answer` 的本地等价物：从命中块里挑**最相关的一句**当搜索引擎 snippet。

        为什么需要这一步：官方 snippet 是搜索引擎答案框（**一句话**），而我们每题只给
        1 个整块（可能几百词）——**不能把整块拼进 prompt**（官方 `cur_prompt` 里那一跳
        只占一句）。所以按子问题对块内句子做 BM25 选句，取 top-1。

        返回空串 = 找不到与子问题有词面重合的句子（调用方决定是否退回 LLM）。
        """
        cands: list[str] = []
        # `hits` 是按块 id 排序的（调用方用 sorted(...)），保证同分时选句确定 —— 传
        # `set` 进来会让迭代顺序随 hash 变，`_bm25` 同分时的 argmax 也随之变，同一题
        # 两次跑出不同结果。
        for i in hits:
            text = self._texts[i] if 0 <= i < len(self._texts) else ""
            if not text:
                continue
            # 本地切句（与 `ircot_adapter` 同一条正则）；**不 import extract.split_sentences**
            # —— 那是建图模块，适配器不该依赖它（建图与检索解耦）。
            for s in re.split(r"(?<=[.!?])\s+", text):
                s = s.strip()
                if s:
                    cands.append(s)
        if not cands:
            return ""
        toks = [_tokenize(c) for c in cands]
        scores = _bm25(subq, toks, k1=SELFASK_BM25_K1, b=SELFASK_BM25_B)
        if not scores:
            return ""
        # 同分取**最后**一个 = 累积顺序里**最新**检索到的那一块：`cur_prompt` 里也是
        # 越靠后的段落越新，优先它可与 prompt 的阅读顺序一致。
        best = max(range(len(scores)), key=lambda j: (scores[j], j))
        # BM25 对"一个词都没重合"的句子给 0 分 → 这种 snippet 不含答案，交给调用方兜底
        return cands[best] if scores[best] > 0 else ""

    def _intermediate(self, subq: str, hits: list[int]) -> str:
        """官方 `get_answer(question)`：本轮检索结果 → 一个短答案。

        三种口径：
          · `SELFASK_SNIPPET=snippet`（**默认**，对齐官方）= 本地挑一句当 snippet 直接返回，
            **不调 LLM**（官方 `get_answer` 就是 `return google(question)`，零 LLM 调用）。
            若挑不出有词面重合的句子且 `SELFASK_SNIPPET_FALLBACK=1` → 退回下面那条 LLM 路径。
          · `SELFASK_SNIPPET=llm` = 旧行为：把检索段落喂给 LLM 改写成中间答案（每题多 1 次/跳）。
          · `SELFASK_GROUND_INTERMEDIATE=0` = 不喂检索结果，让模型按上文自答
            （官方「搜索引擎无结果」分支的等价物，消融档）。
        """
        if self.ground_intermediate and hits and SELFASK_SNIPPET != "llm":
            snip = self._snippet(subq, hits)
            if snip:
                return snip
            if not SELFASK_SNIPPET_FALLBACK:
                return "unknown"
        if self.ground_intermediate and hits:
            ctx = self._context(hits)
            sys = _INTERMEDIATE_SYSTEM
        else:
            ctx, sys = "", ""
        # ⚠️ 这里**有意不传 dataset**：本调用出的是每跳的**中间答案**（口径 = 官方
        #    Self-Ask 的「搜索引擎照抄」，system 也是 _INTERMEDIATE_SYSTEM），不是
        #    最终作答；它必须继续用统一 RAG_PROMPT，才能保持中间步骤口径可比。
        #    最终答案那条（下方 query() 里的 generate_answer）才透传 dataset。
        # ⚠️ 这条调用**必须有长度上限**：官方唯一 LLM 调用是 `max_tokens=256`，而这里
        #    预期只出一个短语。不传上限 → vLLM 生成到 `max_model_len - prompt_len`。
        return generate_from_context(subq, ctx, system=sys, max_tokens=SELFASK_MAX_NEW_TOKENS)

    def query(self, question: str) -> dict:
        if not self._chunks:
            raise RuntimeError("请先 build()")
        self._fewshot, _ = load_official_fewshot()

        cur_prompt = self._fewshot.format(question) + NEEDED
        evidence: list[dict] = []
        seen: list[int] = []        # 已取过的块 id，**检索顺序**（去重 + 给 snippet 定序）
        locs: list = []
        generated = ""
        sub_questions: list[str] = []
        hops = 0
        # ★ 每跳单独一条（`common/trace.py:334` 里 `trace_rounds` 优先级最高）。
        #   以前只返回 `sub_questions`，而 `rounds_from_result` 给每跳挂的都是
        #   **最终**证据 → "每跳各检索到什么"看不出来。
        trace_rounds: list[dict] = []

        # 1) 官方 `ret_text = call_gpt(cur_prompt, intermediate)`
        with llm_role("selfask-step"):
            ret = llm_chat("", cur_prompt, json_mode=False, timeout=SELFASK_LLM_TIMEOUT,
                           flatten=False, max_tokens=SELFASK_MAX_NEW_TOKENS,
                           stop=[INTERMEDIATE]) or ""
        generated += ret

        while hops < self.max_hops:
            subq = extract_followup(ret)
            # 首轮防「不检索直接答」（与 S2G-RAG 的 should_force_first_retrieval 同思路）：
            # 模型第一句就写 `So the final answer is:` 时，我们仍然强制先检索**原问题**一轮。
            # ⚠️ 这一支背后**没有 `Follow up:` 行**，所以：① 模型这一轮的原话 `ret` 必须
            #    先拼进 prompt（否则官方那句 `cur_prompt += ret_text` 被漏掉）；② 本轮
            #    检索结果**不能**写成 `\nIntermediate answer: X.` —— 它在 prompt 里会变成
            #    一个**没有对应 Follow up 的悬空答案**（模型看到"凭空冒出来的检索结果"
            #    会跑偏，且实测会让它下一轮回吐 `So the final answer is:` 从而提前结束
            #    正确路径）。用官方的「搜索引擎无结果」写法 `\nIntermediate answer:`
            #    （只有提示、没有内容）代替。
            forced = subq is None
            if forced:
                if hops != 0 or evidence:
                    break                       # 官方行为：既不追问、也没写答案 → 结束
                subq = question

            # 2) 「搜索引擎」：本项目语料 BM25（跨轮去重）
            hits = self._search(subq, seen)
            for i in hits:
                seen.append(i)
                evidence.append({"text": self._texts[i], "title": self._chunks[i][0]})
                locs.extend(self._chunks[i][3])
            sub_questions.append(subq)

            # 3) 官方 `external_answer = get_answer(question)`
            #    ★ 喂**累积**命中块，不是本轮新块：官方循环每轮把新段落 `+=` 进 prompt，
            #      到第 3 跳时 prompt 里已经有前两跳的段落；只看本轮会让 snippet 丢掉
            #      跨跳指代（"his birthplace"）赖以成立的先验段落。
            #    `seen` 是**检索顺序**的列表（不是 set）：`_snippet` 同分时取最后一个，
            #      依赖"越靠后 = 越新检索到"这个确定顺序（set 迭代顺序随 hash 变）。
            ext = self._intermediate(subq, list(seen)) if seen else "unknown"

            # 4) 官方 `cur_prompt += ret_text` + `cur_prompt += intermediate + ' ' + external_answer + '.'`
            #    ret 必须**另起一行**再接 Intermediate answer：官方 stop 串是
            #    "\nIntermediate answer:"，模型续写自带换行；若直接拼，下一轮模型输出的
            #    "Follow up:" 会被粘在 Intermediate answer 行尾，extract_followup
            #    （只看最后一个 Follow up 行）就永远解析不到，loop 只能走一跳。
            cur_prompt = cur_prompt + "\n" + ret
            if forced:
                cur_prompt += INTERMEDIATE          # 无 Follow up → 官方「无结果」写法
            else:
                cur_prompt += f"{INTERMEDIATE} {ext}."

            # 每跳记录：子问题、本跳「搜索引擎」返回的块、本跳的中间答案
            # `query_used` 必须是**真正拿去检索的那句话**：forced 时是原问题，而
            # `extract_followup(ret)` 那时是 None；trace_rounds 优先级高于通用兜底
            # （`common/trace.py:334`），写错了审计就完全看不到这跳在检索什么。
            # `intermediate_answer` 记的是 `get_answer` **返回值本身**（引擎的确切返回），
            # 与「它有没有被写进 prompt 的 Intermediate answer 行」是两件事。
            trace_rounds.append({
                "round": hops, "query_used": subq,
                "chunks": [{"text": self._texts[i], "title": self._chunks[i][0]}
                           for i in hits],
                "entities": [], "paths": [], "sub_questions": [subq],
                "intermediate_answer": str(ext),
                "forced_query": forced})

            hops += 1

            # 5b) 官方 `ret_text = call_gpt(cur_prompt, intermediate)`（下一轮续写）
            #    ★ `ret` 是"本轮模型原话、已经拼进 cur_prompt"，**退出循环时不能清空**：
            #      收尾那步是 `cur_prompt += ret_text + FINALANS`，清空了就等于让模型对着
            #      半截上下文作答（它上一轮明明已经写了 `Follow up:`，prompt 里却直接跳到
            #      `So the final answer is:`）。所以只在**确认还要继续**时才把它清掉。
            if hops >= self.max_hops:
                cur_prompt += INTERMEDIATE          # 触顶收尾：本轮结果仍要进 prompt
                break                               # ret 留给收尾步骤用
            ret = ""
            with llm_role("selfask-step"):
                ret = llm_chat("", cur_prompt, json_mode=False, timeout=SELFASK_LLM_TIMEOUT,
                               flatten=False, max_tokens=SELFASK_MAX_NEW_TOKENS,
                               stop=[INTERMEDIATE]) or ""
            generated += ret

        # 6) 官方：没写 final answer（或写了但抽不出内容）就补一句再生成
        if extract_final(ret) is None:
            cur_prompt = cur_prompt + "\n" + ret + FINALANS
            with llm_role("selfask-final"):
                ret = llm_chat("", cur_prompt, json_mode=False, timeout=SELFASK_LLM_TIMEOUT,
                               flatten=False, max_tokens=SELFASK_MAX_NEW_TOKENS,
                               stop="\n") or ""
            generated += ret

        answer = extract_final(ret) or extract_final(generated) or "unknown"
        # 6) 本项目口径：用累积证据重新生成（与其余 baseline 同 prompt），防止凭记忆作答
        if self.final_grounded and evidence:
            answer = generate_answer(question, evidence, dataset=getattr(self, "dataset", ""))

        top = list(dict.fromkeys(locs))            # 同一句可能被多个块覆盖，去重
        return {"answer": answer,
                "evidence": [e["text"] for e in evidence],
                "evidence_locs": top,
                "sub_questions": sub_questions,
                "trace_rounds": trace_rounds,
                "hops_used": hops,
                "final_grounded": bool(self.final_grounded and evidence),
                "prompt_chars": len(cur_prompt)}

    def run(self, question, contexts):
        """per-question 路径不用（每题语料不同），统一走语料级 build/query。"""
        raise NotImplementedError("Self-Ask 走语料级：先 build(全部上下文) 再逐题 query")
