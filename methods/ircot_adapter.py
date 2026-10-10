"""IRCoT 适配器 —— **官方算法复现**（Trivedi et al. ACL 2023），接本项目语料。

与旧版的区别（旧版只是「IRCoT 风格」近似）：本版按 `baselines/ircot-main` 的实现逐条对齐。

═══ 官方算法（`commaqa/inference/ircot.py` + `base_configs/ircot_codex_*.jsonnet`）═══
  1. **初始状态**：`data["paras"] = data["titles"] = []`（config 的 reader 是
     `add_paras/add_gold_paras/add_pinned_paras = false`；`gold_with_2_distractors`
     指的是 **prompt 里 few-shot 演示**的构造方式，不是把金段落钉进上下文）。
  2. **每轮**：
     a. 检索（`retrieve_and_reset_paragraphs`，`ircot.py:339-536`）
        - query = 原问题（首轮）/ `remove_reasoning_sentences` 后的**最后一句已生成 thought**
          （`query_source="question_or_last_generated_sentence"`，`:347-353`）
        - `input_query = remove_wh_words(input_query)`（`:433`；去掉 who/what/when/where/why/
          which/how/does/is —— 官方原样保留，默认开，可用 `IRCOT_REMOVE_WH=0` 关）
        - BM25 = **两个字段各算一次再相加**（`document_type="title_paragraph_text"`，
          `elasticsearch_retriever.py:49-52`）：
          ① `query × paragraph_text` ② `query × title` → 分数求和。
          所以 title 命中不会被正文长度稀释（等价于给 title 命中加权）。
        - **先取前 `bm25_retrieval_count` 名（hotpotqa 配置 = 6），之后才过滤**：
          `elasticsearch_retriever.py:99` `retrieval[:max_hits_count]` → 再由
          `ircot.py:474-487` 丢弃「>600 词」与模糊重复的条目，**最终可能不足 6 篇**。
        - 段落级去重 `is_para_closely_matching`（title+正文 rapidfuzz ≥ **90**，`:111-130`）
        - 全局上限 `global_max_num_paras = **15**`（`:486-487`；`cumulate_titles=true` 所以跨轮累积）
     b. 生成一句 thought（`step_by_step_cot_gen`，`ircot.py:790-...`）
        - prompt = **官方 prompt 文件**（`prompts/{dataset}/{variant}_context_cot_qa_codex.txt`，
          29 个 few-shot，**忽略 `# METADATA:` 行**）+ 当前上下文段落（每段 `para_to_text`
          截到 **350 词**，`:133-140`）+ `Q: {question}` + `A: {已生成 thought 逐句拼接}`
        - 模型续写一句；用 spacy 切成句取第一句（`:846-848`；本适配器用等价正则切句）
        - 命中 `.* answer is:? (.*)` → **结束检索循环**（`:851-855`；官方
          `step_by_step_exit_controller`。注意这只是**停止信号**，不是最终答案）
        - 生成句数达 `max_num_sentences = **10**` → 结束（`:797-798`）
  3. **最终答案** = 用**项目统一生成口径** `generate_answer()`（`prompts/rag_prompt.txt`）
     对累积检索到的段落作答 —— 与 standardrag / factalign / hipporag / … 完全同口径、可比。

     **不**采用官方 `ircot_qa` 的额外 `llmqa` 作答步骤：那会引入一个**只有 IRCoT 才有**的
     生成 prompt，破坏 PARAMS_ALIGNED.md §0 的分层设计（② 生成 prompt 全项目统一自定，
     官方 prompt 只在**检索侧**复刻）。循环的停止条件仍是官方的。

═══ 本项目怎么接（论文里必须写明的适配取舍）═══
  - **语料级**：N 题上下文合成一个语料（与 standardrag/factalign 同语料）；检索单元 =
    **整段 paragraph**（同 title 的句子拼起来）——对应官方的 Wikipedia paragraph，
    不是整段切块（IRCoT 官方检索的是 Wikipedia 整段）。BM25 = **两个字段各算一次再相加**（`title_paragraph_text`），
    打分参数用官方值 **k1=1.2 / b=0.75**（ES 内建默认），`remove_wh_words` 后查询。
  - **prompt 用官方文件**：直接从 `baselines/ircot-main/prompts/{ds}/...` 读，
    按 `IRCOT_DISTRACTORS`（默认 2 = 官方调参最优）选变体；缺文件时回退到
    HippoRAG 内置模板并**打印告警**（那不是逐字官方文件）。
  - **无 spacy 依赖**：用等价正则切句（官方 `spacy.load("en_core_web_sm")` 的 sents）。
  - **10 句上限 + answer 模式**都是官方停止条件；`IRCOT_MAX_SENTENCES` 可调。
  - 每题 LLM 调用 = 实际生成的 thought 句数（≤10，命中 answer 提前停）；检索是本地 BM25。
  - **上下文预算**（会顶到 chat 窗口）：官方 prompt 文件本身很大 —— hotpotqa
    2-distractor 版约 11.2k 词 ≈ **14.9k token**，加 15 篇 ×350 词上下文（≈7k token）
    ≈ **21.9k token/次调用**。30k 窗口够用，24.5k 会偏紧（客户端会在超 max_model_len 时
    按保头保尾截断）。窗口紧就设 `IRCOT_DISTRACTORS=1`（≈11k 词的 prompt → 总 ≈17.8k）。

用法:
  python run.py --method ircot --dataset hotpotqa --sample 300 --workers 16
  旋钮（环境变量）:
    IRCOT_DISTRACTORS=1|2|3   prompt 变体（默认 2，官方调参最优）
    IRCOT_REMOVE_WH=0/1       检索前是否 remove_wh_words（官方=1）
    IRCOT_MAX_SENTENCES=10    官方 max_num_sentences
    IRCOT_RETRIEVAL_COUNT=6   官方 bm25_retrieval_count
    IRCOT_GLOBAL_MAX_PARAS=15 官方 global_max_num_paras
    IRCOT_MAX_PARA_WORDS=350  官方 max_para_num_words（para_to_text 的**截断**）
    IRCOT_SKIP_LONG_PARA_WORDS=600   官方「超 600 词整条丢弃」（ircot.py:474，默认生效）
    IRCOT_DONT_SKIP_LONG_PARAS=0/1   官方 dont_skip_long_paras（=1 则不丢弃）
    IRCOT_PROMPT_FILTER=0/1          官方 qid 过滤（valid_qids，默认 1）
    IRCOT_PROMPT_BUDGET=8000         官方 model_length_limit
    IRCOT_PROMPT_TOKENIZER=embed     数 token 的尺子（embed=本项目尺子；gpt2=官方）
    IRCOT_PROMPT_FILE=<path>  直接指定 prompt 文件
"""

import json
import os
import re
import threading
from pathlib import Path

from methods.base import BaseMethod, generate_answer
from methods.standard_rag import _tokenize, _bm25
from common.llm import chat as llm_chat
from config import BASELINES_DIR

# ── 官方超参（默认 = base_configs/ircot_codex_hotpotqa.jsonnet 的值）──
IRCOT_DISTRACTORS = int(os.environ.get("IRCOT_DISTRACTORS", "2"))
IRCOT_REMOVE_WH = os.environ.get("IRCOT_REMOVE_WH", "1") == "1"
IRCOT_MAX_SENTENCES = int(os.environ.get("IRCOT_MAX_SENTENCES", "10"))
IRCOT_RETRIEVAL_COUNT = int(os.environ.get("IRCOT_RETRIEVAL_COUNT", "6"))
IRCOT_GLOBAL_MAX_PARAS = int(os.environ.get("IRCOT_GLOBAL_MAX_PARAS", "15"))
IRCOT_MAX_PARA_WORDS = int(os.environ.get("IRCOT_MAX_PARA_WORDS", "350"))   # para_to_text 截断
# ★ 官方有**两个**段落长度参数，别混：
#   · 截断 `max_para_num_words=350`（ircot.py:592/:731、participant_qa.py:52）→ 上面的 IRCOT_MAX_PARA_WORDS
#   · 丢弃 `>600 词就整条不要`（ircot.py:403/:474，配 `dont_skip_long_paras=False` 见 :266）
# 我们此前**没有丢弃那个** → 超长段落会占掉 global_max_num_paras 的名额。
# 官方语义（ircot.py:474-476）：候选段落词数 > 600 且 dont_skip_long_paras=False → `continue`
# 直接跳过，**不占 15 篇的名额**，再走 is_para_closely_matching 去重。
IRCOT_SKIP_LONG_PARA_WORDS = int(os.environ.get("IRCOT_SKIP_LONG_PARA_WORDS", "600"))
IRCOT_DONT_SKIP_LONG_PARAS = os.environ.get("IRCOT_DONT_SKIP_LONG_PARAS", "0") == "1"
# ★ BM25 打分参数：**用官方自己的值，不落回项目全局默认**。
#   官方建索引 `retriever_server/build_index.py:244-261` 只写了 `analyzer: "english"`，
#   **没有任何 `similarity` 覆盖** → 走 ES 内建默认 **k1=1.2 / b=0.75**。
#   原来这里不传 → 落回 `standard_rag.BM25_K1=1.5`，比官方饱和得慢、
#   让重复出现的词更"值钱"，段落排序会与官方不同（docstring 却写着"走 ES 内建默认"）。
IRCOT_BM25_K1 = float(os.environ.get("IRCOT_BM25_K1", "1.2"))
IRCOT_BM25_B = float(os.environ.get("IRCOT_BM25_B", "0.75"))
IRCOT_LLM_TIMEOUT = int(os.environ.get("IRCOT_LLM_TIMEOUT", "300"))
# 官方每轮生成上限 `max_tokens=300`（gpt3generator.py:91）。我们过去不传 → vLLM 用
# `max_model_len - prompt_len`，即"能生成多少就生成多少"；而每轮只需一句 thought，
# 不设上限纯属白烧 decode 时间。
IRCOT_MAX_NEW_TOKENS = int(os.environ.get("IRCOT_MAX_NEW_TOKENS", "300"))
# 官方停止串 `stop=["\n"]`（gpt3generator.py:87）：thought 只占一行。配套
# `flatten=False`（保换行）—— 没有 stop 时只能靠 `_first_sentence` 事后正则切。
IRCOT_STOP = os.environ.get("IRCOT_STOP", "1") == "1"

# ── 官方 prompt 两级压缩（此前完全没做）──────────────────────────────────
# 官方是**两级**：`prompt_reader.read_prompt`（读入时，last_first 删尾）+
# `fit_prompt_into_given_limit`（每次生成时，first 删头），参数见
# base_configs/ircot_codex_*.jsonnet:11-19：
#   filter_by_key_values={"qid": valid_qids}（**只留 20 条**）、order_by_key="qid"、
#   estimated_generation_length=300、model_length_limit=8000。
# 我们此前只把 `# METADATA:` 行 strip 掉就**全量 29 条**塞进去（11090 词），
# 比官方多 9 条、且没有 8000 token 预算 —— 这是最容易被审稿人一眼看出的偏离。
IRCOT_PROMPT_FILTER = os.environ.get("IRCOT_PROMPT_FILTER", "1") == "1"
IRCOT_PROMPT_BUDGET = int(os.environ.get("IRCOT_PROMPT_BUDGET", "8000"))
IRCOT_EST_GEN = int(os.environ.get("IRCOT_EST_GEN", "300"))
# token 尺子：默认 **`embed` = 配置的 embedder 尺子（Qwen3-Embedding-0.6B）** ——
# 与全项目切块/截断用**同一把尺子**。官方原本用 gpt2；要严格官方口径设 `gpt2`。
# 两把尺子对同一段英文的 token 数不同 → 同样 8000 预算下能塞进的示例条数会不同，
# 所以实际用的尺子会写进启动日志（**不静默**）。
#   embed（默认）→ 配置的 embedder 尺子（qwen3/bge，随 EMBED_MODEL 变）
#   gpt2         → 官方口径（需要 transformers）
#   auto         → 优先 gpt2，拿不到再退配置尺子
IRCOT_PROMPT_TOKENIZER = os.environ.get("IRCOT_PROMPT_TOKENIZER", "embed")
_METADATA_PREFIX = "# METADATA: "
# 连续多少轮「检索不到新段落」就结束（官方是检索失败计数上限 9；这里是"没有新段落"，
# 语料级下最后一轮通常如此 —— 给得宽一点，让模型仍能把 max_num_sentences 走完）
IRCOT_EMPTY_RETRIEVAL_RETRIES = int(os.environ.get("IRCOT_EMPTY_RETRIEVAL_RETRIES", "10"))

# 官方 prompt 文件所在（相对 baselines/），按数据集选目录
IRCOT_PROMPT_ROOT = BASELINES_DIR / "ircot-main" / "prompts"
_dataset_dir = {"hotpotqa": "hotpotqa", "2wiki": "2wikimultihopqa",
                "musique": "musique", "iirc": "iirc"}

# 官方：answer_extractor_regex=".* answer is:? (.*)\."，去掉末尾句点
_ANSWER_RE = re.compile(r".* answer is:? (.*)", re.DOTALL)
# 官方 remove_wh_words 的词表（ircot.py:58-62）
_WH_WORDS = {"who", "what", "when", "where", "why", "which", "how", "does", "is"}
# 官方 is_reasoning_sentence 的前缀（ircot.py:30-41）：这类句子不参与下一轮检索 query
_REASONING_STARTERS = ("thus ", "thus,", "so ", "so,", "that is,", "therefore", "hence")
_REASONING_REGEX = re.compile(
    r"(.*)(\d[\d,]*\.?\d+|\d+) ([+-]) (\d[\d,]*\.?\d+|\d+) = (\d[\d,]*\.?\d+|\d+)(.*)")
# spacy sents 的等价近似：句末标点后断开（官方 new_sents[0].text）
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

_PROMPT_CACHE: dict = {}
_PROMPT_LOCK = threading.Lock()


def remove_wh_words(text: str) -> str:
    """官方 ircot.py:58-62 原样：去掉疑问词/助动词后再检索。"""
    words = [w for w in text.split(" ") if w.strip().lower() not in _WH_WORDS]
    return " ".join(words)


def is_reasoning_sentence(sentence: str) -> bool:
    """官方 ircot.py:30-41：以 thus/so/therefore/hence 开头，或算式 → 非事实句。"""
    for starter in _REASONING_STARTERS:
        if sentence.lower().startswith(starter):
            return True
    return bool(_REASONING_REGEX.match(sentence))


def remove_reasoning_sentences(sentences: list[str]) -> list[str]:
    return [s for s in sentences if not is_reasoning_sentence(s)]


def para_to_text(title: str, para: str, max_num_words: int = IRCOT_MAX_PARA_WORDS) -> str:
    """官方 ircot.py:133-140：正文截到 max_num_words 词，前缀 `Wikipedia Title: {title}`。"""
    para = " ".join(para.split(" ")[:max_num_words])
    if para.strip().startswith("Wikipedia Title: "):
        return para.strip()
    return "Wikipedia Title: " + title + "\n" + para.strip()


def _token_similarity(a: str, b: str) -> float:
    """`rapidfuzz.fuzz.ratio` 的等价实现，返回 0~100。装了 rapidfuzz 就用它。

    ⚠️ 名字里的 "token" 是历史遗留、**读起来会误导**：`fuzz.ratio` 是**字符级**
    Indel 相似度（归一化 Levenshtein），不是词级或子词级。
    官方 IRCoT 用的也是 rapidfuzz `fuzz.ratio`（阈值 90），所以**行为与官方一致**，
    只是名字容易让人以为是分词后比较。
    """
    try:
        from rapidfuzz import fuzz
        return float(fuzz.ratio(a, b))
    except Exception:
        import difflib
        return 100.0 * difflib.SequenceMatcher(None, a, b).ratio()


def is_para_closely_matching(existing_titles, existing_paras, new_title, new_para,
                             match_threshold: float = 90) -> bool:
    """官方 ircot.py:111-130：title+正文都 ≥90 才算重复。"""
    if new_title in existing_titles and new_para in existing_paras:
        return True
    for t, p in zip(existing_titles, existing_paras):
        if _token_similarity(t, new_title) >= match_threshold and \
                _token_similarity(p, new_para) >= match_threshold:
            return True
    return False


def strip_metadata(text: str) -> str:
    """去掉 `# METADATA: {...}` 行（prompts/README 要求运行时忽略）。"""
    return "\n".join(l for l in text.splitlines() if not l.startswith(_METADATA_PREFIX))


# ── 官方 prompt 处理：解析 metadata → qid 过滤 → 排序 → last_first 压缩 ──
_TT_CACHE: dict = {}


def _prompt_token_counter():
    """返回 `(计数函数, 尺子说明)`。

    官方 `read_prompt(tokenizer_model_name="gpt2")` 用 `len(tokenizer.tokenize(text))`。
    这里优先拿真 gpt2；拿不到就退到配置的尺子；再不行用「词数×1.3」估算。
    **估算会让压缩条数与官方有出入**，所以尺子说明会写进启动日志，绝不静默。
    """
    if "v" in _TT_CACHE:
        return _TT_CACHE["v"]
    pref = (IRCOT_PROMPT_TOKENIZER or "auto").strip().lower()
    res = None
    if pref in ("auto", "gpt2"):
        try:
            from transformers import AutoTokenizer
            _tok = AutoTokenizer.from_pretrained("gpt2")
            res = ((lambda s: len(_tok.tokenize(s))), "gpt2（官方口径）")
        except Exception:
            if pref == "gpt2":
                raise
            res = None
    if res is None and pref in ("auto", "embed"):
        try:
            from common.llm import _get_tokenizer, _is_tiktoken
            _t = _get_tokenizer()
            if _is_tiktoken(_t):
                res = ((lambda s: len(_t.encode(s))), "tiktoken（非官方 gpt2）")
            else:
                res = ((lambda s: len(_t.encode(s, add_special_tokens=False))),
                       "配置的 embedder 尺子（⚠ 非官方 gpt2）")
        except Exception:
            res = None
    if res is None:
        res = ((lambda s: int(len(s.split()) * 1.3)),
               "词数×1.3（⚠ 估算，非官方 gpt2 → 压缩条数与官方会有出入）")
    _TT_CACHE["v"] = res
    return res


def _metadata_examples(text: str) -> list[dict]:
    """照官方 `read_prompt` 把 prompt 文件切成带 metadata 的 example 列表。

    ⚠ `text` 是**文件内容**，不是路径。传路径进来不会报错，会静默返回 1 条
    `{"default": True}` 的空示例（因为整个路径字符串里没有 `# METADATA:` 前缀），
    于是 qid 过滤全落空、few-shot 全丢 —— 外部工具（如
    `tools/check_ircot_demo_leak.py`）最容易被这个坑到，所以这里显式挡一下。
    """
    if "\n" not in text and len(text) < 4096:
        raise ValueError(
            f"_metadata_examples() 收到的是单行短字符串（{text[:80]!r}）——它要的是 prompt "
            f"**文件内容**，不是路径。请先 read_text() 再传进来。")
    lines = [e + "\n" for e in text.strip().split("\n")]
    example: dict = {"default": True, "lines": []}
    examples: list[dict] = []
    for index, line in enumerate(lines):
        if index == len(lines) - 1:
            examples.append(example)
        if line.strip().startswith(_METADATA_PREFIX):
            examples.append(example)
            try:
                md = json.loads(line.strip().replace(_METADATA_PREFIX, "", 1))
            except Exception:
                md = {}
            example = dict(md)
            example["lines"] = []
        else:
            example["lines"].append(line)
    return examples


def _official_valid_qids(dataset: str) -> list[str]:
    """从官方 jsonnet 读 `valid_qids[dataset]`（唯一真源，避免手抄 20 个 id 抄错）。"""
    sub = _dataset_dir.get(dataset, dataset)
    jn = BASELINES_DIR / "ircot-main" / "base_configs" / f"ircot_codex_{sub}.jsonnet"
    if not jn.exists():
        return []
    try:
        src = jn.read_text(encoding="utf-8")
    except Exception:
        return []
    m = re.search(r'"' + re.escape(sub) + r'"\s*:\s*\[(.*?)\]', src, re.S)
    return re.findall(r'"([^"]+)"', m.group(1)) if m else []


def fit_ircot_prompt(prompt: str, test_example: str) -> str:
    """官方 `fit_prompt_into_given_limit`（生成时那一级）：`remove_method="first"`。

    按 `"\\n\\n\\n"` 切示例，从**头部**删，直到 `sum(示例) + test + est_gen` 进预算。
    官方读入时是 last_first（删尾），生成时是 first（删头）—— 两级方向相反，都做了。
    """
    demos = [d.strip() for d in prompt.split("\n\n\n") if d.strip()]
    counter, _ = _prompt_token_counter()
    sizes = [counter(d) for d in demos]
    test_len = counter(test_example)
    while sizes and sum(sizes) + test_len + IRCOT_EST_GEN >= IRCOT_PROMPT_BUDGET:
        demos.pop(0)
        sizes.pop(0)
    # ★ 退化分支：例子删光了还是超预算 = **当前这轮本身**就装不下（累积段落太多）。
    #   官方在这里也会走到空 prompt，但 gemini/gpt3 会直接报错，问题一眼可见；
    #   vLLM 则是 400 "maximum context length"。**必须出声**，否则会被误读成"官方就是
    #   零样本"，而且每题每轮都会静默失败一次。
    #   为什么实际很难发生：单段 >600 词的候选在 `_retrieve` 就被丢了
    #   （`IRCOT_SKIP_LONG_PARA_WORDS`），再经 `max_para_num_words=350` 截断 +
    #   `global_max_num_paras=15` 封顶 → 上文有硬上界，只有把预算调得很小时才会撞。
    if not demos and test_len + IRCOT_EST_GEN >= IRCOT_PROMPT_BUDGET:
        print(f"[ircot] ⚠ 本轮上下文 {test_len} tokens + 预留生成 {IRCOT_EST_GEN} "
              f"已超 prompt 预算 {IRCOT_PROMPT_BUDGET}（已无示例可删）→ 发给模型的 prompt "
              f"将只剩本轮上下文；若 vLLM 报 400，请调大 IRCOT_PROMPT_BUDGET "
              f"或调小 IRCOT_GLOBAL_MAX_PARAS", flush=True)
    return "\n\n\n".join(demos)


def load_official_prompt(dataset: str) -> tuple[str, str]:
    """读官方 prompt 并按**官方口径**处理：qid 过滤 → 按 valid_qids 排序 → last_first 压缩。

    返回 (prompt 文本, 说明)；说明会写进启动日志，含"29 条 → 过滤 N 条 → 压缩后 M 条"。
    找不到官方文件则回退内置模板并告警。
    """
    override = os.environ.get("IRCOT_PROMPT_FILE")
    if override:
        p = Path(override)
    else:
        sub = _dataset_dir.get(dataset, dataset)
        # 运行时可调（模块常量只是默认值）：IRCOT_DISTRACTORS=1/2/3
        n_dist = os.environ.get("IRCOT_DISTRACTORS", str(IRCOT_DISTRACTORS))
        name = f"gold_with_{n_dist}_distractors_context_cot_qa_codex.txt"
        p = IRCOT_PROMPT_ROOT / sub / name
    key = (f"{p}|{IRCOT_PROMPT_FILTER}|{IRCOT_PROMPT_BUDGET}|{IRCOT_EST_GEN}"
           f"|{IRCOT_PROMPT_TOKENIZER}")
    with _PROMPT_LOCK:
        if key in _PROMPT_CACHE:
            return _PROMPT_CACHE[key]

    if not p.exists():
        text = _fallback_prompt()
        note = f"⚠ 缺官方 prompt（{p}）→ 回退内置模板（非逐字官方）"
    else:
        counter, ruler = _prompt_token_counter()
        examples = [e for e in _metadata_examples(p.read_text(encoding="utf-8"))
                    if e.get("lines")]
        total = len(examples)

        if IRCOT_PROMPT_FILTER:
            qids = _official_valid_qids(dataset)
            if qids:
                # 官方 filter_by_key_values={"qid": valid_qids} + order_by_key="qid"
                examples = [e for e in examples if e.get("qid") in qids]
                examples.sort(key=lambda e: qids.index(e["qid"]))
            else:
                ruler += "｜⚠ 未取到 valid_qids，跳过 qid 过滤"
        n_filtered = len(examples)

        texts = ["".join(e["lines"]).strip() for e in examples]
        lens = [counter(t) for t in texts]
        dropped = 0
        # 官方 last_first：从尾部删，直到 sum + max*1 + est_gen <= budget
        while lens and sum(lens) + max(lens) + IRCOT_EST_GEN > IRCOT_PROMPT_BUDGET:
            texts.pop()
            lens.pop()
            dropped += 1
        text = "\n\n\n".join(t.strip() for t in texts)
        note = (f"官方 prompt: {p.relative_to(BASELINES_DIR.parent)}｜"
                f"{total} 条 → qid 过滤 {n_filtered} 条 → 压缩后 {len(texts)} 条"
                f"（预算 {IRCOT_PROMPT_BUDGET}，尺子 {ruler}）")
        if dropped:
            note += f"｜last_first 删尾 {dropped} 条"
        if not texts:
            note += "｜⚠ 全部被删光"
    with _PROMPT_LOCK:
        _PROMPT_CACHE[key] = (text, note)
    return text, note


def _fallback_prompt() -> str:
    """官方 prompt 缺失时的回退（HippoRAG 内置 ircot 模板，非逐字官方文件）。"""
    p = (BASELINES_DIR / "HippoRAG-main" / "src" / "hipporag" / "prompts"
         / "templates" / "ircot_hotpotqa.py")
    if not p.exists():
        return ""
    ns: dict = {}
    try:
        exec(compile(p.read_text(encoding="utf-8"), str(p), "exec"), ns)
    except Exception:
        return ""
    return str(ns.get("ircot_system") or ns.get("one_shot_ircot_demo") or "")


class IRCoT(BaseMethod):
    """官方 IRCoT 复现（语料级：build() 建段落索引，query() 逐题迭代检索+推理）。"""

    name = "ircot"

    def __init__(self, retrieval_count: int = IRCOT_RETRIEVAL_COUNT,
                 global_max_paras: int = IRCOT_GLOBAL_MAX_PARAS,
                 max_num_sentences: int = IRCOT_MAX_SENTENCES):
        self.dataset = ""
        self.retrieval_count = retrieval_count
        self.global_max_paras = global_max_paras
        self.max_num_sentences = max_num_sentences
        self._titles: list[str] = []
        self._paras: list[str] = []
        self._toks: list[list[str]] = []
        self._sents: list[list[int]] = []      # 每篇段落的句号列表（重建 evidence_locs 用）
        self._prompt = ""
        self._prompt_note = ""

    # ── 语料级准备：段落索引（BM25）──
    def build(self, corpus_contexts: list[dict], working_dir: str = "") -> "IRCoT":
        """把 N 题上下文按 title 合并成 Wikipedia 式段落，建 BM25 索引。"""
        by_title: dict[str, list[tuple[int, str]]] = {}
        for c in corpus_contexts:
            title = (c.get("title") or "").strip() or "untitled"
            sents = c.get("sentences") or re.split(r"(?<=[.!?])\s+", c.get("text", ""))
            sents = [s.strip() for s in sents if s and s.strip()]
            if not sents:
                continue
            base = len(by_title.get(title, []))
            by_title.setdefault(title, []).extend(
                (base + i, s) for i, s in enumerate(sents))
        self._titles = list(by_title.keys())
        self._paras = [" ".join(s for _, s in by_title[t]) for t in self._titles]
        self._sents = [[i for i, _ in by_title[t]] for t in self._titles]
        self._toks = [_tokenize(p) for p in self._paras]
        # ★ 官方 document_type="title_paragraph_text" → ES 对 title / paragraph_text
        #   **各算一次 BM25 再把分数相加**（elasticsearch_retriever.py:49-52，bool.should
        #   下的两个 match 子句分数求和；unified_retriever.py:44 由 document_type 决定
        #   query_title_field_too，官方 ircot 配置就是 title_paragraph_text）。
        #   效果：title 命中不会被正文长度稀释，等价于给 title 命中加权。
        #   原来只对正文打分 → **title 完全不计分**（_paras 里没有 title）。
        #   ES 按字段各建一份统计（N / avgdl / df），`_bm25` 分开调用天然一致。
        self._title_toks = [_tokenize(t) for t in self._titles]
        self._prompt, self._prompt_note = load_official_prompt(self.dataset or "hotpotqa")
        print(f"[ircot] 语料段落 {len(self._paras)} 篇（按 title 合并）；"
              f"检索={self.retrieval_count}/轮，全局上限 {self.global_max_paras}，"
              f"max_num_sentences={self.max_num_sentences}；{self._prompt_note}", flush=True)
        return self

    # ── 官方检索（BM25 + 模糊去重 + 全局上限）──
    def _retrieve(self, query: str, sel_titles: list[str], sel_paras: list[str],
                  sel_idx: list[int]) -> int:
        """按官方 retrieve_and_reset_paragraphs 往 selected 里追加。返回本次新增数。"""
        if IRCOT_REMOVE_WH:
            query = remove_wh_words(query)
        if not query.strip():
            return 0
        if not self._paras:
            return 0
        scores = _bm25(query, self._toks, k1=IRCOT_BM25_K1, b=IRCOT_BM25_B)
        # 官方双字段 BM25：正文分 + title 分（理由见 build() 里的注释）
        scores = [a + b for a, b in zip(
            scores, _bm25(query, self._title_toks, k1=IRCOT_BM25_K1, b=IRCOT_BM25_B))]
        # ★ 官方口径是「**先取前 retrieval_count 名，之后才过滤**」：
        #   elasticsearch_retriever.py:99 `retrieval = retrieval[:max_hits_count]`
        #   （max_hits_count = retrieval_count = 6），紧接着 ircot.py:474-487 才丢弃
        #   >600 词 / 模糊重复的条目 → **最终可能不足 6 篇**。
        #   原来是「一直往排名深处扫，直到新增满 6」→ 每轮必定补足 6 篇，比官方拿得多。
        order = sorted(range(len(self._paras)), key=lambda i: -scores[i])[:self.retrieval_count]
        added = 0
        for i in order:
            if len(sel_paras) >= self.global_max_paras:
                break
            title, para = self._titles[i], self._paras[i]
            # 官方 ircot.py:474-476 —— 候选段落超过 600 词直接丢弃，**不占 15 篇的名额**。
            # 判断方式逐字照抄官方：`len(para.split(" "))`（不是 split()）。
            if (not IRCOT_DONT_SKIP_LONG_PARAS
                    and len(para.split(" ")) > IRCOT_SKIP_LONG_PARA_WORDS):
                continue
            if is_para_closely_matching(sel_titles, sel_paras, title, para):
                continue
            sel_titles.append(title)
            sel_paras.append(para)
            sel_idx.append(i)
            added += 1
        return added

    @staticmethod
    def _first_sentence(text: str) -> str:
        """官方用 spacy 切句取第一句；这里用等价正则。"""
        parts = _SENT_SPLIT_RE.split(text.strip(), maxsplit=1)
        return parts[0].strip() if parts and parts[0].strip() else ""

    @staticmethod
    def _extract_answer(thought: str) -> str | None:
        """官方 answer_extractor_regex=".* answer is:? (.*)"（去掉末尾句点）。"""
        m = _ANSWER_RE.match(thought)
        if not m:
            return None
        ans = m.group(1).strip()
        return ans[:-1] if ans.endswith(".") else ans

    def query(self, question: str) -> dict:
        if not self._paras:
            raise RuntimeError("请先 build()")
        sel_titles: list[str] = []
        sel_paras: list[str] = []
        sel_idx: list[int] = []
        generated: list[str] = []
        hit_answer = False
        cur_query = question
        empty_retrievals = 0
        rounds = 0
        # ★ 每轮单独一条（`common/trace.py:334` 里 `trace_rounds` 优先级最高）。
        #   官方 IRCoT = 「检索 → 生成一句 thought → 拿最后一句事实性 thought 再检索」，
        #   每轮**真正新拿到的**是末尾 `added` 篇（`_retrieve` 往 sel_* 尾部追加）。
        #   以前只返回 `thoughts`，而 `rounds_from_result` 给每轮挂的都是**最终**证据
        #   → "每轮各检索到什么"其实看不出来。
        trace_rounds: list[dict] = []

        while len(generated) < self.max_num_sentences:
            # 1) 检索（官方：query = 原问题 / 最后一句事实性 thought）
            added = self._retrieve(cur_query, sel_titles, sel_paras, sel_idx)
            if added == 0:
                empty_retrievals += 1
                if empty_retrievals > IRCOT_EMPTY_RETRIEVAL_RETRIES:
                    break
            else:
                empty_retrievals = 0
            rounds += 1

            # 每轮记录：这轮的 query + **本轮新**检索到的段落（+ 累计数）
            _new = []
            if added:
                for _t, _p in list(zip(sel_titles, sel_paras))[-added:]:
                    _new.append({"text": para_to_text(_t, _p), "title": _t})
            trace_rounds.append({"round": rounds, "query_used": cur_query,
                                 "chunks": _new, "entities": [], "paths": [],
                                 "n_new": added, "n_cum": len(sel_paras)})

            # 2) 生成一句 thought（官方 prompt：上下文 + Q + A:已生成）
            context = "\n\n".join(para_to_text(t, p) for t, p in zip(sel_titles, sel_paras))
            test_example = (context + "\n\n" if context else "") + \
                f"Q: {question}" + "\n" + f"A: {' '.join(generated)}"
            prompt = "\n\n\n".join([fit_ircot_prompt(self._prompt, test_example),
                                    test_example]).strip()
            out = llm_chat("", prompt, json_mode=False, timeout=IRCOT_LLM_TIMEOUT,
                           flatten=False, max_tokens=IRCOT_MAX_NEW_TOKENS,
                           stop=["\n"] if IRCOT_STOP else None)
            sentence = self._first_sentence(out or "")
            if not sentence:
                break
            generated.append(sentence)
            if trace_rounds:
                trace_rounds[-1]["thought"] = sentence

            # 3) 官方停止条件：命中 answer 模式即结束**检索**（`step_by_step_exit_controller`）
            #    —— 这只是循环的停止信号，**不再拿它当最终答案**（见下方收尾）
            ans = self._extract_answer(sentence)
            if ans is not None:
                hit_answer = True
                break

            # 4) 下一轮检索 query = 最后一句「非推理」thought（官方过滤 thus/so/算式）
            facts = remove_reasoning_sentences(generated)
            cur_query = facts[-1] if facts else ""

        # ── 最终答案：走**项目统一生成口径**（`generate_answer` → `prompts/rag_prompt.txt`）──
        # 原做法是拿自己 CoT prompt 的输出正则抽答案 —— 那是**唯一**不走统一生成口径的
        # 方法，与 PARAMS_ALIGNED.md §0 的分层设计冲突（② 生成 prompt 全项目统一自定，
        # 官方 prompt 只在**检索侧**复刻）。改统一后：
        #   · 与 standardrag / factalign / hipporag / proprag / … 的答案口径一致、可比
        #   · 不必再复刻官方的 `llmqa` 作答步骤，也就没有"每题少 1 次 LLM 调用"的问题
        # 循环本身的停止条件仍是官方的（命中 `answer is:` 或 10 句）—— 它决定**检索到
        # 什么程度**，属检索侧口径（`step_by_step_exit_controller`）。
        evidence = [{"text": para_to_text(t, p)} for t, p in zip(sel_titles, sel_paras)]
        answer = generate_answer(question, evidence, dataset=getattr(self, "dataset", "")) if evidence else "unknown"
        # evidence_locs：把选中的段落映射回语料句子 (title, sent_idx)，与其它方法同字段
        locs = [(self._titles[i], si) for i in sel_idx for si in self._sents[i]]
        return {"answer": answer,
                "evidence": evidence,
                "evidence_locs": locs,
                "thoughts": generated,
                "hit_answer_in_loop": hit_answer,
                "trace_rounds": trace_rounds,
                "retrieval_rounds": rounds}

    def run(self, question, contexts):
        """per-question 路径不用（每题的语料不同），统一走语料级 build/query。"""
        raise NotImplementedError("IRCoT 走语料级：先 build(全部上下文) 再逐题 query")
