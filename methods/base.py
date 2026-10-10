"""方法基类 + 统一接口 + 答案生成。"""

import os
import re

from common.llm import chat as llm_chat, strip_thinking, role as llm_role
from config import PROMPTS_DIR

# 生成答案 prompt：统一放 prompts/rag_prompt.txt，改文件即生效（无需改代码）。
# {context} / {question} 是 .format 占位符，文件里不要出现其它花括号。
_RAG_PROMPT_PATH = PROMPTS_DIR / "rag_prompt.txt"
if not _RAG_PROMPT_PATH.exists():
    raise RuntimeError(
        f"生成 prompt 文件缺失：{_RAG_PROMPT_PATH}\n"
        f"改生成答案 prompt 请直接编辑该文件（改完重跑即生效，无需重建索引）")
RAG_PROMPT = _RAG_PROMPT_PATH.read_text(encoding="utf-8").strip()

# TriviaQA 专属作答 prompt（对齐 S2G-RAG 的 triviaqa 分支口径）：
# 单跳事实题 → 只要求**最短精确 span**，并「优先用证据里的 canonical alias，而不是页面标题」。
# 只有 dataset == "triviaqa" 时才用它，其余数据集一律用上面的统一 RAG_PROMPT。
_TRIVIAQA_PROMPT_PATH = PROMPTS_DIR / "rag_prompt_triviaqa.txt"
if not _TRIVIAQA_PROMPT_PATH.exists():
    raise RuntimeError(
        f"TriviaQA 生成 prompt 文件缺失：{_TRIVIAQA_PROMPT_PATH}\n"
        f"改 triviaqa 作答口径请直接编辑该文件（改完重跑即生效，无需重建索引）")
TRIVIAQA_RAG_PROMPT = _TRIVIAQA_PROMPT_PATH.read_text(encoding="utf-8").strip()

# 数据集的作答 prompt 表。键 = `--dataset` 的取值；不在表里的一律用统一 RAG_PROMPT。
# ★ 全项目只有这两条**带证据**的生成路由（4 个数据集 × 9 个检索方法共用）：
#     hotpotqa / musique / 2wiki → `prompts/rag_prompt.txt`（多跳口径，Thought + Answer）
#     triviaqa                   → `prompts/rag_prompt_triviaqa.txt`（单跳口径，Reason + Answer）
#   任何方法都不要再自带一份作答 prompt：方法之间的差异应当只在**检索/推理**上。
RAG_PROMPTS = {"triviaqa": TRIVIAQA_RAG_PROMPT}

# ── 无证据路径（zeroshot）：单独一份，**不按数据集分派** ──
# 为什么不分派：单跳/多跳口径讲的是"从证据里抄哪个 span、抄多长"，无证据时没有 span 可抄、
# 答的是参数知识 —— 给它做 triviaqa 版本是空转。所以这里只有"有证据 / 无证据"两个维度，
# 数据集维度对它不存在。
_NO_RAG_PROMPT_PATH = PROMPTS_DIR / "rag_prompt_nocontext.txt"
if not _NO_RAG_PROMPT_PATH.exists():
    raise RuntimeError(
        f"无证据作答 prompt 文件缺失：{_NO_RAG_PROMPT_PATH}\n"
        f"改 zeroshot 作答口径请直接编辑该文件（改完重跑即生效）")
NO_RAG_PROMPT = _NO_RAG_PROMPT_PATH.read_text(encoding="utf-8").strip()
# 该路径**没有** `{context}`，只有 `{question}`
NO_CONTEXT_METHODS = ("zeroshot",)


def rag_prompt_for(dataset: str = "", method: str = "") -> str:
    """选作答 prompt：先看是不是**无证据**方法（与数据集无关），再按数据集分派。

    为什么 method 要进来：`rag_prompt_for()` 只覆盖"带证据作答"，而 zeroshot 出的是最终
    答案却完全不经过它 → manifest 里登记的 `rag_prompt` 是个**从没用过**的 prompt
    （留档与实际不符）。把无证据路径并进来之后，"每一次出答案"都能在 manifest 里查到
    真正用的那份文件。
    """
    if (method or "").strip().lower() in NO_CONTEXT_METHODS:
        return NO_RAG_PROMPT
    return RAG_PROMPTS.get((dataset or "").strip().lower(), RAG_PROMPT)


# ── 作答输出契约：「先想一行，再在 `Answer:` 行给答案」 ──
# 两份 prompt 都要求最后一行是 `Answer: <答案>`；只有那一行算预测，
# 所以**必须**在这里把答案行切出来，否则整个 "Thought: … Answer: X" 会进 EM/F1 直接判错。
ANSWER_MARKER = "Answer:"
REASON_MARKER = "Reason:"          # triviaqa 那份用的思考行标记
THOUGHT_MARKER = "Thought:"        # 多跳那份用的思考行标记
# 模型把 "So the answer is: X"（HippoRAG 2 IRCoT 口径）当答案的情况也接住
_ANSWER_ALIASES = ("Final answer:", "So the answer is:", "The answer is:")
# 模型原样吐出**模板占位符**（未替换方括号里的说明）→ 当无效答案，不许进 EM/F1。
# 原判据只认 `provide|answer here|your answer` 三种措辞，**漏掉了我们自己 prompt 里的
# 占位符**（`[the exact answer, and nothing else]` / `[one short sentence]` /
# `[that piece of text, copied exactly as it appears]`）→ 实测模型抄占位符时它原样变成 pred。
# 现判据：**整行就是一个方括号短串且内部没有再嵌方括号**（≤80 字符）。
# 为什么安全：LM 引用 `[7]` 这类是行内片段、不是整行；真实答案很少整行被方括号包住却不含内层括号；
# 而 ≤80 的上限把"整段被误包"的情况排除掉。
_PLACEHOLDER_RE = re.compile(r"^\[[^\[\]]{1,80}\]$", re.DOTALL)

# 作答复用的 system 消息：统一放这里，zeroshot 等自带 prompt 的路径也 import 它，
# 保证「只输出 Answer 行」这个契约全项目只有一处定义。
ANSWER_SYSTEM = ("You output ONLY the answer line. Never explain in the answer line. "
                 "Keep the reasoning line (if the format asks for one) to one short "
                 "sentence; the answer itself must be complete even if it is long.")


def extract_final_answer(text: str) -> str:
    """把模型输出切成**最终答案行**（`Answer:` 之后的内容）。

    背景：两份作答 prompt 都要模型「先写一行 Thought/Reason，再写 `Answer: <答案>`」。
    评分只能看答案行，所以这里做切分：
      · 多个标记命中时取**最后一个**（模型可能复述格式、或给多段推理后再收尾）；
      · `Answer:` 后同一行的内容才算答案，且只取该行（后面若还有行，丢弃）；
      · 顺手去掉答案两侧的引号与句末句点（纯展示口径，正则本身会去掉标点——
        去掉句点是为了让 pred 与 gold 看起来一致，不改判分结果）；
      · 找不到任何标记 → 原样返回（老行为兜底：万一模型直接给裸答案）；
      · 只找到思考标记、没有答案标记（多半是生成被截断）→ 返回空串，
        让调用方按「无效答案」处理（EM/F1 诚实记 0，别把 Thought 整段当答案）。
    """
    t = (text or "").strip()
    if not t:
        return ""
    markers = (ANSWER_MARKER, *_ANSWER_ALIASES)
    best = None                                    # (位置, 标记长度)
    for m in markers:
        i = t.rfind(m)
        if i >= 0 and (best is None or i > best[0]):
            best = (i, len(m))
    if best is None:
        # 没有答案标记 → **一律判无效**（返回空串，调用方按 unknown 处理，EM/F1 诚实记 0）。
        #
        # ★ 这里原先有个"老行为兜底"：文本里不含 `Thought:`/`Reason:` 字面量时就**原样返回**
        #   整段——理由是"万一模型直接给裸答案"。实测证明这个兜底是有害的：
        #   模型不按两行格式输出时，写出来的是**关于证据的元描述**
        #   （`The evidence mentions the film's director and production location. New York City`），
        #   原样返回就会把整段描述当成答案 → EM 记 0 但 **F1 会拿到虚假非零分**，
        #   把"格式失败"混进"答错"，污染统计。
        # 对照上游（**语义参考，未复制措辞**）：
        #   · S2G-RAG `utils/text_processing.py:141` 找不到标记时返回哨兵 `"Answer not found"`；
        #   · HippoRAG2 `HippoRAG.py:1158` 直接 `split('Answer:')[1]`，靠 prompt 保证标记存在。
        #   两家都**不把原始输出当答案**，只有我们原来这么干。
        # 与之配套：`prompts/rag_prompt.txt` 已加硬约束「没有 `Answer:` 行则无效记 0」
        # 并给了反例——prompt 说无效、这里就必须真的判无效，两边不能矛盾。
        return ""
    tail = t[best[0] + best[1]:]
    answer = tail.splitlines()[0].strip() if tail.strip() else ""
    answer = answer.strip().strip('"').strip("'").strip()
    if answer.endswith("."):
        answer = answer[:-1].strip()
    if not answer or _PLACEHOLDER_RE.match(answer):
        return ""
    return answer

# ── 生成阶段的证据预算（默认值；调用方可用 `windows=` 覆盖）──
# ① **命题**（evidence 里的 `summary`）：**总是全发**（不设条数上限）。
#    为什么不做成旋钮：只有 factalign 的命题**同时**带 `summary` 与 `window_text`（见下面 167 行
#    的 `if not wt: continue`），而它在 factalign 侧已经由 `FACTALIGN_EVIDENCE_PROPS` 定死了条数
#    （`top` 本身就只是前 N 条）—— 再在通用层加第二个上限既是重复、又容易两边不一致。
#    （历史上这里有个 `EVIDENCE_PROPS`，默认 0 = 不发命题。已删除：它**永远不生效**
#     —— 唯一走得到这条分支的 factalign 显式传了自己的值，其他方法传的证据没有
#     `window_text`，到不了这行。`tools/audit_params.py --unconfigured` 会把这类
#     "读了环境变量但没有配置键"的洞列出来。）
# ② EVIDENCE_WINDOWS = 发几个**原文窗口**（`window_text`）。
#    ★ 默认 **5**。**思路：生成阶段只喂检索到的那 5 个窗口**，与 standardrag 的
#    `STANDARDRAG_TOP_K=5` 对齐 —— 检索候选数不该顺带变成"喂给模型的证据数"。
# 只有 factalign 的命题**同时**带 `summary` 与 `window_text`（extract.py:269 填的），
# 所以下面的窗口预算目前只影响它（它另显式传自己的 `FACTALIGN_EVIDENCE_WINDOWS`）。
EVIDENCE_WINDOWS = int(os.environ.get("EVIDENCE_WINDOWS", "5"))


class BaseMethod:
    name = "base"
    # 当前数据集名。run.py 对**有该属性**的方法对象执行 `method_obj.dataset = args.dataset`，
    # 各 adapter 的作答路由靠它选 prompt（triviaqa → 专属 prompt，其余 → 统一 prompt）。
    # ★ 放在基类上：以前 hipporag/hypergraphrag/lightrag/standardrag 没有这个属性，
    #   run.py 的 hasattr 守卫会静默跳过赋值，它们的 triviaqa 作答就悄悄退回了统一 prompt
    #   （代码没错、结果错，最难查的一类）。
    dataset = ""

    def run(self, question: str, contexts: list[dict]) -> dict:
        """返回 {"answer": str, "evidence": [str], "evidence_locs": [(title, sent_idx)]}。"""
        raise NotImplementedError


def format_evidence(evidence: list[dict],
                    windows: int | None = None) -> list[str]:
    """evidence: list of {"window_text"/"summary"/"text", "time_span", "source"} → prompt 的编号块。

    ── 分支 ①：**有 `window_text`**（只有 factalign 的命题同时给 `summary` + `window_text`）──
    先列**命题**（`summary`，检索到的蒸馏事实，**全发**），再列**原文窗口**（`window_text`，
    受 `windows` 限制）。编号**跨两段连续**（命题 [1..P]，窗口 [P+1..P+W]），中间空一行 →
    `prompts/rag_prompt.txt` 里那些 "Evidence [N] says…" 的指代始终唯一。
      · `windows=0` → 不发窗口
    两段各自去重（同一命题 / 同一窗口只出现一次，按原始顺序保留首次）。

    ── 分支 ②：**无 `window_text`**（standardrag/hipporag/hypergraphrag 直接传原文 text）──
    原样拼，**无封顶** —— 与分支 ① 是不同口径，对照表见 AUDIT_PARAMS.md §6。
    """
    _w = EVIDENCE_WINDOWS if windows is None else windows
    props_out, wins_out, seen_p, seen_w = [], [], set(), set()
    for ev in evidence:
        wt = (ev.get("window_text") or "").strip()
        if not wt:
            continue
        sm = (ev.get("summary") or "").strip()
        if sm and sm not in seen_p:               # 命题全发（条数已由调用方 top-N 决定）
            seen_p.add(sm)
            props_out.append(sm)
        if _w > 0 and wt not in seen_w:
            seen_w.add(wt)
            wins_out.append(wt)
    if props_out or wins_out:
        ws = wins_out[:_w]
        out = []
        if props_out:                             # 命题段：[1..P]
            out.append("\n".join(f"[{i}] {p}" for i, p in enumerate(props_out, 1)))
        if ws:                                    # 窗口段：[P+1..P+W]
            out.append("\n".join(f"[{len(props_out) + j}] {w}" for j, w in enumerate(ws, 1)))
        return out
    # 无窗口信息（standardrag/hipporag/hypergraphrag 直接传原文 text）：原样拼
    parts = []
    for i, ev in enumerate(evidence):
        p = f"[{i+1}] {ev.get('summary', ev.get('text', ''))}"
        if ev.get("time_span"):
            p += f" (time: {ev['time_span']})"
        if ev.get("source"):
            p += f"\n    Source: {ev['source']}"
        parts.append(p)
    return parts


def generate_from_context(question: str, context: str, system: str = "",
                          dataset: str = "", max_tokens: int | None = None) -> str:
    """用统一 RAG_PROMPT 作答（context 由调用方拼好，不再走 evidence 结构）。

    Self-Ask 这类方法每一跳要拿「本轮检索到的段落」回答**中间问题**（不是原问题），
    传 evidence 结构表达不了，所以把「拼 context」与「作答」拆开：
    generate_answer = format_evidence + generate_from_context。
    system 非空时替换默认作答 system（Self-Ask 的中间答案是「搜索引擎照抄」口径，
    不是推理口径）；prompt 主体仍是统一 RAG_PROMPT，口径可比。

    `dataset` 决定用**哪份作答 prompt**：`"triviaqa"` → `rag_prompt_triviaqa.txt`
    （S2G-RAG 的单跳口径），其余/未指定 → 统一 `rag_prompt.txt`。

    `max_tokens`：本次生成的硬上限。**默认 None = 不传**（vLLM 用
    `max_model_len - prompt_len`，即"能生成多少就生成多少"）——最终作答需要这个默认值
    （长答案不能被截）；但**中间步骤必须传**：Self-Ask 每跳的中间答案预期只是一个短语，
    官方那条唯一的 LLM 调用也是 `max_tokens=256`。

    返回值是**答案行**（`extract_final_answer` 切出来的），不是模型原始输出；
    原始输出在逐题 trace 里（`common/trace.py` 的 raw_output）。
    """
    prompt = rag_prompt_for(dataset).format(context=context, question=question)
    # 生成答案阶段关思考：答案本身 5-10 词，思考纯开销且可能触顶截断
    # role("reader")：只影响逐题 trace 的标签，让它一眼看出这条输出来自作答环节
    with llm_role("reader"):
        out = llm_chat(system or ANSWER_SYSTEM,
                       prompt, json_mode=False, timeout=120, thinking=False,
                       max_tokens=max_tokens)
    if out is None:
        return "unknown"
    text = out if isinstance(out, str) else out.get("text", "")
    # 统一去 think（common/llm.py 已清一遍，这里兜底；对齐 hotpotqa）
    text = strip_thinking(text)
    # 只取 `Answer:` 行；被截断（只剩思考行）或占位符 → 无效答案，诚实记 0
    answer = extract_final_answer(text)
    return answer if answer else "unknown"


def generate_answer(question: str, evidence: list[dict],
                    windows: int | None = None,
                    dataset: str = "") -> str:
    """evidence: list of {"window_text"/"summary"/"text", "time_span", "source"}。返回短答案。

    拼 prompt 的口径全在 `format_evidence` 里；`windows` 原样透传给它可以覆盖
    默认的窗口预算（`None` = 用模块默认 `EVIDENCE_WINDOWS`）。**命题条数不在这里控制** ——
    发进来的 `evidence` 是几条就发几条（factalign 在调用前已按 `FACTALIGN_EVIDENCE_PROPS`
    截成 top-N，见 `methods/factalign.py`）。
    `dataset="triviaqa"` 时改用 TriviaQA 专属 prompt（见 `rag_prompt_for`）。

    small-to-big：检索返回命题（，细粒度），生成时既能发命题本身，
    也能把命题映射回它们的**原文窗口**（一个语料单元的完整段落；默认不切块，
    超过 2000 token 的按 2000 切，见 `config.CHUNK_NO_SPLIT`）。

    ⚠️ 各方法的篇数**并不同口径**，别当成一回事：
      · **factalign**：命题 `FACTALIGN_EVIDENCE_PROPS`=20（= `decomp_top_k`5 × `decomp_steps`4
        的全部累积）+ 原文窗口 `FACTALIGN_EVIDENCE_WINDOWS`=5，编号连续 [1..25]
      · **standardrag**：走 `STANDARDRAG_TOP_K` = **5**（纯稠密，无 BM25/RRF），
        且它的 evidence 只有 `"text"` 字段
        → 落进 `format_evidence` 的**无封顶**分支，5 篇全部进 prompt（不发命题）
      · 其余（hipporag / hypergraphrag / s2grag / proprag …）：只传 `"text"`，同上无封顶
    全项目证据预算对照见 AUDIT_PARAMS.md §6。
    """
    return generate_from_context(
        question, "\n\n".join(format_evidence(evidence, windows=windows)),
        dataset=dataset)
