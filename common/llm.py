"""统一 LLM 封装：chat + embed，支持 ollama / vllm 切换。

附带「分阶段、线程安全」的 token/时间统计：
- build（建图：抽取 + embedding）
- query（检索 + 生成 + judge）
通过 `with phase('build'): ...` 标记当前线程的统计归属，支持并发跑题时各阶段统计仍准确。
"""

import asyncio
import contextlib
import contextvars
import json
import os
import re
import threading
import time
from pathlib import Path

import numpy as np
import requests

# ★★ HuggingFace 下载镜像：**必须在本模块 body 里 export**，不能只在 `_load_tokenizer`
#    里面 setdefault。原因：`huggingface_hub` 在 **import 时**就把 `HF_ENDPOINT` 读成模块常量，
#    之后再改 os.environ **毫无作用** —— 于是"设了镜像却还在连 huggingface.co 超时"这种
#    查不出原因的故障。放在模块 body、且此处**早于**任何 `from transformers import ...`，
#    就能保证镜像一定先于它生效。
#    优先级：真实环境变量 / `configs/*.json` 的 `hf_endpoint` > 这里的默认镜像（setdefault 不覆盖）。
#    无外网的机器请显式关掉联网：`export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from config import (
    BACKEND, OLLAMA_BASE_URL, CHAT_MODEL, EMBED_MODEL, VLLM_BASE_URL,
    VLLM_EMBED_BASE_URL, EMBED_MODEL_SERVED, EMBED_QUERY_INSTRUCTION,
    EMBED_BATCH_SIZE, EMBED_REQUEST_TOKENS,
    EMBED_DIM, EMBED_MAX_TOKENS, EMBED_TOKENIZER, EMBED_TIKTOKEN_MODEL,
    CHUNK_TOKENIZER,
    EMBED_DIM_CHECK, EMBED_QUERY_FORMAT, EMBED_QUERY_TASK, EMBED_QUERY_TEMPLATE,
    EMBED_DOC_PREFIX, EMBED_VERBOSE,
)

# ── 分阶段统计（build / query 两桶）──
# 每个线程独立两桶（thread-local），支持并发跑题时按题取差值；
# 全局视图 = 所有线程桶的合并。
_phase_tls = threading.local()
_bucket_registry: list[dict] = []
_registry_lock = threading.Lock()
# 统计"世代"：reset_stats() 每调用一次 +1。每个线程记下自己上次登记时的世代号，
# 世代不一致 → 重新登记。**只有世代号能解决这个问题**：注册表是全局的、线程是各自的，
# 主线程 reset 时不可能去改别的线程的 thread-local 标记（实测：worker 的 `registered`
# 标志会一直是 True，于是它 reset 之后的调用全丢账）。
_stats_generation = 0


def _thinking_kwarg_enabled() -> bool:
    """是否在请求里带 `chat_template_kwargs={"enable_thinking": false}`。

    这是 **Qwen3** 的模板变量。Qwen2.5/Llama 的模板会忽略未定义变量，但个别模型/版本
    下 vLLM 会直接回 400 → 换非 Qwen3 模型时可 `export CHAT_TEMPLATE_THINKING=0` 全局去掉。
    默认 1：保证既有 Qwen 路径的请求体一字不变。
    """
    return os.environ.get("CHAT_TEMPLATE_THINKING", "1") == "1"

# 失败原因去重：并发建图时同一条 400/超时会刷成几百行，只报首次
_ERR_SEEN: set[str] = set()
_ERR_SEEN_LOCK = threading.Lock()
# 全局默认阶段：外部库（graphrag/lightrag）内部起线程池时无法继承调用线程的
# phase 上下文，适配器在 build 期间用 set_default_phase("build") 覆盖默认归属。
_default_phase: str | None = None
# 统计桶覆盖（跨线程/跨 event loop 派发用）：适配器在调用线程取桶后 set，
# contextvars 会随 run_coroutine_threadsafe / 线程派发复制到执行线程，
# chat_messages 优先用它归账，保证单题 token 差值记到发起查询的线程。
bucket_ctx = contextvars.ContextVar("llm_bucket_override", default=None)

_EMPTY = {"chat_calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
          "chat_seconds": 0.0, "embed_calls": 0, "embed_tokens": 0, "embed_seconds": 0.0}


def _cur_phase() -> str:
    p = getattr(_phase_tls, "name", None)
    if p is not None:
        return p
    return _default_phase or "query"


# ★ trace 用的「角色」标签：judge / reader / selector / extract / rewrite / linking…
# 适配器用 `with llm.role("judge"):` 给期间的调用打标（contextvars：线程/异步安全）。
role_ctx = contextvars.ContextVar("llm_role", default="")


def _cur_role() -> str:
    return role_ctx.get() or ""


@contextlib.contextmanager
def role(name: str):
    """给期间的所有 LLM 调用打角色标签（只影响 trace，不影响行为）。

    用法：`with llm.role("judge"): out = llm.chat(...)` —— trace 里那次调用就是
    role="judge"，事后能一眼看出「这一条输出是哪个环节发的」。
    """
    tok = role_ctx.set(name or "")
    try:
        yield
    finally:
        role_ctx.reset(tok)


def cur_phase() -> str:
    """当前线程当前统计阶段名（"build" / "query"）。适配器可按阶段改变行为
    （如响应缓存只在 build 阶段生效）。"""
    return _cur_phase()


def set_default_phase(name: str | None):
    """覆盖「无 phase 上下文线程」的统计归属（供外部方法适配器使用），**返回旧值**。

    graphrag 建索引时内部起线程池调 LLM/embedding，那些线程没有 phase 上下文，
    默认会落到 query 桶；适配器在 build() 前后用 set_default_phase("build" / None)
    把期间的调用归到 build 桶。

    ⚠️ 为什么这是**必须的**而不是可选的：`phase('build')` 设的是 `_phase_tls.name`
    （**线程局部**，common/llm.py:35/153-159），对 `ThreadPoolExecutor` 里的 worker
    **无效** → worker 里 `_cur_phase()` 落到 `_default_phase or "query"`（:68）→
    建图期在线程池里做的 LLM 调用全被记进 **query 桶**：`get_stats()["build"]`
    显示 chat prompt=0，而 query 桶被撑大。实测 proprag @2wiki：
    3356 块 × 2 次抽取，build 段记成 `chat prompt=0 completion=0`。
    原先靠各 adapter 自己 set/reset（hipporag :1126-1131、hypergraphrag :782-834），
    是**易漏的逐方法约定**（proprag 就漏了）→ 现在 run.py 在建图段统一兜住。
    返回值用于 try/finally 安全还原（不要硬写 None，会清掉外层设置）。
    """
    global _default_phase
    prev, _default_phase = _default_phase, name
    return prev


def get_phase_bucket() -> dict:
    """当前线程当前阶段的统计桶（适配器跨线程调用时把统计归到调用方）。

    例：graphrag 的 completion_async 用 asyncio.to_thread 执行 HTTP 调用，统计
    默认会记到线程池线程上；先在调用线程取桶、作为 bucket 参数传给 chat_messages，
    单题差值统计才能正确归到发起查询的线程。
    """
    return _get_buckets()[_cur_phase()]


def _get_buckets() -> dict:
    """当前线程的两桶统计（首次访问/世代过期时创建并登记，供全局合并）。

    ★ 登记条件**不能**只看"桶是不是刚建的"：`reset_stats()` 会清空注册表，而各线程的
      `_phase_tls.buckets` 还留着。只用 `b is None` 判据时，**已经调用过统计的线程**
      在 reset 之后不会再登记 → 它后续的 token 对 `get_stats()` **永久不可见**
      （实测：worker 先调一次得 100，主线程 reset 后再调一次，全局仍是 0）。
      光加一个 per-thread `registered` 布尔标记**也不够** —— 那个标记也在 thread-local
      里，主线程改不到别的线程 → worker 会一直以为自己是登记着的。必须用全局**世代号**：
      reset 时 +1，线程发现自己的世代号落后就补登记。
    """
    global _stats_generation
    b = getattr(_phase_tls, "buckets", None)
    if b is None:
        b = {"build": dict(_EMPTY), "query": dict(_EMPTY)}
        _phase_tls.buckets = b
        _phase_tls.generation = -1
    if getattr(_phase_tls, "generation", -1) != _stats_generation:
        with _registry_lock:
            # ★ 必须按**身份**判重，不能 `b not in _bucket_registry` —— list 的 `in`
            #   走 `==`，而桶是 dict：两个都还没花过 token 的空桶**值相等** →
            #   判断成"已在表里" → 新线程的桶**永远登记不进去** → 它的 token 全部
            #   对 get_stats() 不可见。实测踩过（worker 桶 id 与表里 id 不同，
            #   但 `b in 表` 返回 True）。用 id 集合按身份判。
            if not any(x is b for x in _bucket_registry):
                _bucket_registry.append(b)
            _phase_tls.generation = _stats_generation
    return b


def snapshot() -> dict:
    """当前线程统计快照（深拷贝，用于算单题 token/时间差值）。"""
    b = _get_buckets()
    return {"build": dict(b["build"]), "query": dict(b["query"])}


def diff_snapshot(before: dict, after: dict) -> dict:
    """两次快照差值：{phase: {k: after[k]-before[k]}}。"""
    out = {}
    for p in ("build", "query"):
        out[p] = {k: after[p][k] - before[p][k] for k in _EMPTY}
    return out


class phase:
    """`with phase('build'): ...` 把期间的 LLM 调用计入对应阶段。"""

    def __init__(self, name: str):
        self.name = name
        self.prev = None

    def __enter__(self):
        self.prev = getattr(_phase_tls, "name", "query")
        _phase_tls.name = self.name
        return self

    def __exit__(self, *args):
        _phase_tls.name = self.prev


def reset_stats():
    """清空所有线程的统计（在启动 worker 之前调用）。

    ★ 做三件事，缺一不可：
      ① **原地清零**每个已登记桶 —— 只 `clear()` 注册表的话，那些 dict 仍被各线程的
         `_phase_tls.buckets` 引用着，值还留在里面（旧账没清掉）；
      ② 清注册表；
      ③ **世代号 +1** —— 让其它线程下次调用时发现"我登记的那一代已经作废"并补登记。
         不加这一步，reset 之后**已用过统计的线程**的 token 全部丢账
         （不报错、只让成本表偏小，最难发现的那类）。
    """
    global _stats_generation
    with _registry_lock:
        for b in _bucket_registry:
            for ph in b.values():                    # b = {"build": {...}, "query": {...}}
                for k in ph:
                    ph[k] = 0
        _bucket_registry.clear()
        _stats_generation += 1
    # 主线程（调用方自己）：清零并按新世代重新登记，保证后续 get_stats 一定能看到它
    b = _get_buckets()
    for ph in b.values():
        for k in ph:
            ph[k] = 0


def _summarize(b: dict) -> dict:
    b = dict(b)
    b["total_tokens"] = b["prompt_tokens"] + b["completion_tokens"] + b["embed_tokens"]
    b["total_seconds"] = b["chat_seconds"] + b["embed_seconds"]
    return b


def get_stats() -> dict:
    """返回 {"build": {...}, "query": {...}, "total": {...}}（所有线程桶合并）。"""
    with _registry_lock:
        buckets = list(_bucket_registry)
    build = dict(_EMPTY)
    query = dict(_EMPTY)
    for b in buckets:
        for k in _EMPTY:
            build[k] += b["build"][k]
            query[k] += b["query"][k]
    build = _summarize(build)
    query = _summarize(query)
    total = {
        "total_tokens": build["total_tokens"] + query["total_tokens"],
        "total_seconds": build["total_seconds"] + query["total_seconds"],
        "chat_calls": build["chat_calls"] + query["chat_calls"],
        "embed_calls": build["embed_calls"] + query["embed_calls"],
    }
    return {"build": build, "query": query, "total": total}


def _strip_fences(content: str) -> str:
    content = re.sub(r"^```(?:json)?\s*", "", content.strip())
    content = re.sub(r"\s*```$", "", content)
    return content


def strip_thinking(content: str, flatten: bool = True) -> str:
    """去掉 Qwen3 / DeepSeek-R1 的思考输出，覆盖四种形态：

    - <think>...</think>（DeepSeek-R1 / 部分模型）
    - Qwen3 的 /think ... /no_think 标记（文本会夹在二者之间）
    - 未闭合的 <think> 或 /think（生成在思考中途被截断，如触顶 max-model-len）

    清理后若只剩空白则返回空串。

    flatten=True（默认，**与旧版逐字一致**）：把全部空白（含换行）压成单空格 ——
    单行答案/短输出用。**需要保留多行结构的调用方（如按 `output.split("\\n")`
    逐行解析子查询的 adapter）必须传 flatten=False**，
    否则换行会被压平、多行输出塌成一行。
    """
    content = re.sub(r"<\s*think\s*>.*?<\s*/\s*think\s*>", "", content, flags=re.DOTALL | re.IGNORECASE)
    content = re.sub(r"/think\b.*?/no_think\b", "", content, flags=re.DOTALL | re.IGNORECASE)
    content = re.sub(r"<\s*think\s*>.*$", "", content, flags=re.DOTALL | re.IGNORECASE)
    content = re.sub(r"/think\b.*$", "", content, flags=re.DOTALL | re.IGNORECASE)
    if flatten:
        content = re.sub(r"\s+", " ", content)
    else:
        # 只归一化水平空白与空行，**保留换行**（多行解析方的依赖）
        content = re.sub(r"[ \t\r\f\v]+", " ", content)
        content = re.sub(r"\n{2,}", "\n", content)
    return content.strip()


def _clean_output(content: str, json_mode: bool, flatten: bool = True) -> str | None:
    """统一清理链：strip fences + 去 think 后解析 JSON（或返回纯文本）。

    flatten 透传给 `strip_thinking`（False = 保留换行，见其 docstring）。

    注意：不做「残留 /think 标记就返回 None」的兜底 —— 一旦内容被 strip_thinking
    清空，json.loads 会自然失败走 None 分支，不影响调用方的兜底逻辑。
    """
    content = strip_thinking(content, flatten=flatten)
    if not json_mode:
        return content
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return None


def chat(system: str, user: str, json_mode: bool = True, timeout: int = 120,
         thinking: bool = False, max_tokens: int | None = None,
         flatten: bool = True, stop=None):
    """返回 dict(json_mode=True) 或 str。失败返回 None。

    thinking=False 时请求级关掉 Qwen3 思考。默认关（抽取/planner/judge/生成全部
    关思考：省 token/时间、避免思考触顶截断）。如需开思考，显式传 thinking=True。

    max_tokens：显式限制本次生成长度。默认 None = 不传（vLLM 会用
    ``max_model_len - prompt_len``，即「能生成多少就生成多少」）——对「只输出一个词/
    一段筛选结果」的调用（route/decompose/extract 这类）不设上限会让 decode 白烧时间。

    flatten：是否把输出空白（含换行）压平，默认 True（老行为）。**多行输出必须
    传 False**（否则解析方按 "\\n" 切分会拿到一行）。

    stop：停止串（`str` 或 `list[str]`），命中即截断生成；默认 None = 不传该字段。
    **官方 baseline 正是靠它切分输出的**（Self-Ask 的 ``"\\nIntermediate answer:"``、
    IRCoT 的 ``"\\n"``）——不传就只能事后用正则猜边界，会多生成一段且容易猜错。
    两条后端路径都已支持：vLLM 放 payload 顶层 `stop`，ollama 放 `options.stop`。
    """
    p = _cur_phase()
    t0 = time.time()
    pt = ct = 0
    result = None
    raw = None
    err = None
    _conc_start()                      # ★ 全局在飞计数（所有方法都经过这里）
    try:
        if BACKEND == "ollama":
            result, pt, ct, raw = _chat_ollama(system, user, json_mode, timeout,
                                               flatten=flatten, stop=stop)
        else:
            result, pt, ct, raw = _chat_vllm(system, user, json_mode, timeout, thinking,
                                             max_tokens, flatten=flatten, stop=stop)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    finally:
        _conc_end()
        b = _get_buckets()[p]
        b["chat_calls"] += 1
        b["prompt_tokens"] += pt
        b["completion_tokens"] += ct
        b["chat_seconds"] += time.time() - t0
        # ★ 逐题 trace（未开时立即返回）：留「发出去的 prompt + 模型原始输出 + 用量」
        _trace_out([{"role": "system", "content": system},
                    {"role": "user", "content": user}], raw,
                   pt=pt, ct=ct, seconds=time.time() - t0, error=err)
        # ★ 逐条调用审计（默认开）：建图阶段只有这里能留下单次成本 + 原始输入输出
        _call_log_out([{"role": "system", "content": system},
                       {"role": "user", "content": user}], raw,
                      pt=pt, ct=ct, seconds=time.time() - t0, error=err, kind="chat",
                      req={"json_mode": bool(json_mode), "max_tokens": max_tokens,
                           "timeout": timeout, "stop": stop, "flatten": bool(flatten),
                           "thinking": bool(thinking)})
    return result


# ── 输入超长保护（客户端截断）──
# PRoH/HGR 这类方法的 query prompt 会把检索上下文全塞进去，长到超过 vLLM 的
# `max_model_len`（实测 26369 / 36411 vs 上限 24576）→ 服务端直接 400、整题失败。
# 这里在发送前把总输入压到 `max_model_len - max_tokens - 余量` 以内（见 fit_messages）。
LLM_INPUT_MARGIN = int(os.environ.get("LLM_INPUT_MARGIN", "512"))       # 给 chat 模板留的余量
LLM_TRUNC_KEEP_HEAD = float(os.environ.get("LLM_TRUNC_KEEP_HEAD", "0.35"))  # 保头部比例
LLM_TRUNC_PROBE_RATIO = float(os.environ.get("LLM_TRUNC_PROBE_RATIO", "0.85"))  # 超过预算这个比例才去精确计数
_MAXLEN = {"v": None}           # 缓存的 max_model_len（None=没查过，0=查不到）
_TRUNC_LOGGED = 0


def _max_model_len() -> int | None:
    """从 vLLM `/v1/models` 读 `max_model_len`（缓存在进程内；拿不到返回 None）。

    用它当"单请求输入上限"的判据，而不是写死常量——换模型/换 --max-model-len 时自动跟随。
    """
    if _MAXLEN["v"] is not None:
        return _MAXLEN["v"] or None
    try:
        r = requests.get(f"{VLLM_BASE_URL}/models", timeout=5)
        d = r.json()["data"][0]
        _MAXLEN["v"] = int(d.get("max_model_len") or 0)
        print(f"[llm] 单请求输入上限取服务端 max_model_len={_MAXLEN['v']}"
              f"（来自 vllm serve --max-model-len；超长会按它截断）", flush=True)
    except Exception as e:
        _MAXLEN["v"] = 0                      # 查不到 → 记 0，本次进程不再试
        print(f"[llm] ⚠️ 拿不到 max_model_len（{type(e).__name__}）→ "
              f"本次进程**不做超长截断**（超长会被 vLLM 直接 400）", flush=True)
    return _MAXLEN["v"] or None


def _est_tokens(text: str) -> int:
    """**便宜的**上界估算（字符/4，英文偏保守）。

    注意：chat prompt 的 token 数**不能用 bge tokenizer 数**——bge 的词表/合并规则和
    Llama/Qwen 不同（英文下多 ~25~35%），拿它当预算会多截一大截。bge tokenizer 只用于
    embedding 侧的截断（`fit_embed`，按 `EMBED_MAX_TOKENS`，现默认 4096）。真正接近上限时会用 `_chat_tokens()`
    走 vLLM `/tokenize` 精确计数。
    """
    return len(text) // 4 + 1


def _chat_tokens(text: str) -> int | None:
    """用**被服务的 chat 模型**的 tokenizer 精确计数（vLLM `/tokenize`）；拿不到返回 None。"""
    try:
        r = requests.post(f"{VLLM_BASE_URL}/tokenize",
                          json={"model": CHAT_MODEL, "prompt": text,
                                "add_special_tokens": False},
                          timeout=10)
        if r.status_code == 200:
            n = r.json().get("count")
            return int(n) if n is not None else None
    except Exception:
        pass
    return None


def _cut_text(text: str, tokens_used: int, tokens_keep: int, limit: int) -> str:
    """按 token 比例砍中段：保头 `LLM_TRUNC_KEEP_HEAD`，尾部留到 `tokens_keep`。

    比例用该条消息自己的 chars/token（由精确计数得出），保证按真实 token 尺度砍。
    """
    cpt = max(1.0, len(text) / max(tokens_used, 1))
    head_n = int(len(text) * LLM_TRUNC_KEEP_HEAD)
    keep_chars = int(tokens_keep * cpt)
    if keep_chars >= len(text) - head_n - 100:
        return text
    cut = len(text) - head_n - keep_chars
    return (text[:head_n]
            + f"\n...[truncated ~{int(cut / cpt)} tokens to fit context]...\n"
            + text[len(text) - keep_chars:])


def fit_messages(messages: list[dict], max_new: int | None,
                 limit: int | None = None) -> tuple[list[dict], int]:
    """把整段对话的输入压到 `limit - max_new - 余量` 以内：**保头保尾、截最长那条的中段**。

    为什么要它：PRoH / HGR 这类方法的 query prompt 会把检索到的上下文全塞进去，长到
    超过 vLLM 的 `max_model_len` → 服务端直接 400（实测见过 26369 / 36411 token vs 上限
    24576）。在客户端按文本身长截断，比让整题失败或写死方法参数都干净。

    截中段而不是截尾：RAG 类 prompt 的结构是「规则/示例 … 证据 … 问题/Answer:」，
    头尾都要保；中间的证据是可牺牲的那部分。
    返回 (新 messages, 被丢掉的估算 token 数)；不需要截断时原样返回、丢 0。
    """
    if limit is None:
        limit = _max_model_len()
    if not limit:
        return messages, 0                    # 查不到上限 → 不动（保持旧行为）
    margin = LLM_INPUT_MARGIN
    budget = limit - (max_new or 0) - margin
    if budget <= 0:
        budget = max(1024, limit // 2)        # max_new 就快把窗口占满的极端情况

    counts = [_est_tokens(m.get("content") or "") for m in messages]
    total = sum(counts)
    # 两级判定：便宜的字符估算离上限还远（<85% 预算）就直接发，不做任何 HTTP；
    # 只有接近/超过上限时才用 vLLM /tokenize 精确计数（chat 模型自己的 tokenizer）。
    if total <= budget * LLM_TRUNC_PROBE_RATIO:
        return messages, 0
    exact = [_chat_tokens(m.get("content") or "") for m in messages]
    if all(e is not None for e in exact):
        counts = [int(e) for e in exact]
        total = sum(counts)
        if total <= budget:
            return messages, 0

    need = total - budget                     # 要丢的 token 数
    out = [dict(m) for m in messages]
    dropped = 0
    # 从最长的那条开始丢它的中段；不够再丢下一条
    for idx in sorted(range(len(out)), key=lambda i: -counts[i]):
        if need <= 0:
            break
        text = out[idx].get("content") or ""
        if len(text) < 400:
            continue                          # 太短，丢了也没用（多半是 system/问题）
        keep = max(64, counts[idx] - need)
        new_text = _cut_text(text, counts[idx], keep, budget)
        if new_text == text:
            continue
        after = _chat_tokens(new_text)
        new_n = after if after is not None else _est_tokens(new_text)
        dropped += counts[idx] - new_n
        out[idx]["content"] = new_text
        need -= counts[idx] - new_n
    if dropped:
        _note_truncation(total, budget, dropped, limit, max_new)
    else:
        return messages, 0
    return out, dropped


def _note_truncation(total, budget, dropped, limit, max_new):
    global _TRUNC_LOGGED
    _TRUNC_LOGGED += 1
    if _TRUNC_LOGGED <= 50:                   # 只报前 50 次，别刷屏
        print(f"[llm] prompt 超长 → 客户端截断中段：约 {dropped} tokens "
              f"（原 {total} > 预算 {budget}；max_model_len={limit}, max_tokens={max_new}）"
              f"｜phase={_cur_phase()}", flush=True)



def chat_messages(messages: list[dict], json_mode: bool = True, timeout: int = 120,
                  bucket: dict | None = None, max_tokens: int | None = 4096,
                  err_out: list | None = None):
    """OpenAI messages 格式调用 vLLM，统一计入统计（供 graphrag 等外部适配器复用）。

    与 chat() 的区别：messages 是完整对话（可多轮），返回 (原始文本, prompt_tokens,
    completion_tokens)——不做 JSON 解析（外部库自己解析，如 graphrag 用 json-repair）。
    bucket：统计归属桶（默认当前线程当前阶段）；调用方把 HTTP 放到线程池执行时，
    可先取 get_phase_bucket() 传入，统计仍归调用线程。失败返回 (None, 0, 0)。

    max_tokens：默认 4096 兜底——正常抽取输出 500~2500 token 不受影响；个别 chunk
    若陷入重复循环，最多跑到 4096 就截断（8192 时约 5 分钟、且这 5 分钟一直占着
    vLLM 的 KV 块不让，是"KV 一直涨 + 排队"的元凶）。传 None 可完全关闭。

    err_out：可选列表；失败时把 `异常类名: 消息` 追加进去，供调用方区分
    「排队/连接类超时（该退让并发）」和「请求本身被拒（400，退让没用）」。
    """
    p = _cur_phase()
    t0 = time.time()
    pt = ct = 0
    result = None
    raw = None
    err = None
    sent_msgs = messages
    _dropped = None                    # fit_messages() 截掉的 token 数（审计要记这一条）
    _conc_start()                      # ★ 全局在飞计数（外部库走 chat_messages 的也算）
    try:
        messages, _dropped = fit_messages(messages, max_tokens)
        sent_msgs = messages
        payload = {"model": CHAT_MODEL, "messages": messages, "temperature": 0.0}
        if max_tokens:
            payload["max_tokens"] = int(max_tokens)
        # 统一关 Qwen3 思考（与 chat() 默认一致；外部适配器抽取/生成也走这里）
        # `enable_thinking` 是 Qwen3 的模板变量；换 Llama 等模型若被 vLLM 判非法参数（400），
        # 设 CHAT_TEMPLATE_THINKING=0 即可全局去掉它（默认 1，Qwen 路径行为不变）。
        if _thinking_kwarg_enabled():
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        resp = requests.post(f"{VLLM_BASE_URL}/chat/completions", json=payload, timeout=timeout)
        if resp.status_code >= 400:
            # 400/404 的**响应体**才有用（vLLM 会说清「max_model_len 不够 / 参数不支持 /
            # 模型名不对」），raise_for_status 只给状态码，等于没信息
            raise RuntimeError(f"HTTP {resp.status_code} {resp.reason}: {resp.text[:400]}")
        resp.raise_for_status()
        data = resp.json()
        usage = data.get("usage", {})
        pt = usage.get("prompt_tokens", 0)
        ct = usage.get("completion_tokens", 0)
        raw = data["choices"][0]["message"]["content"]     # 原始输出，供 trace 留档
        result = strip_thinking(_strip_fences(raw))
    except Exception as e:
        # 失败原因打印（失败才打，不刷屏）：LightRAG/GraphRAG 建图的「文档抽取失败」
        # 只记录到「vLLM 调用 N 次均失败」，看不到根因；这里把超时/400/断连等
        # 细节暴露出来，方便区分「排队超时」和「请求本身被拒」。
        # 同一原因只打一次：建图并发调用失败时同一条 400 会刷几百行，反而淹没别的信息。
        sig = f"{type(e).__name__}: {e}"
        err = sig
        with _ERR_SEEN_LOCK:
            first = sig not in _ERR_SEEN
            _ERR_SEEN.add(sig)
        if first:
            print(f"[llm] chat_messages 失败（同因只报一次，另有 chat_calls 计数）: {sig}",
                  flush=True)
        if err_out is not None:
            err_out.append(sig)
        result = None
    finally:
        _conc_end()
        b = bucket or bucket_ctx.get() or _get_buckets()[p]
        b["chat_calls"] += 1
        b["prompt_tokens"] += pt
        b["completion_tokens"] += ct
        b["chat_seconds"] += time.time() - t0
        _trace_out(sent_msgs, raw, pt=pt, ct=ct, seconds=time.time() - t0, error=err)
        _call_log_out(sent_msgs, raw, pt=pt, ct=ct, seconds=time.time() - t0, error=err,
                      kind="chat_messages", dropped=_dropped,
                      req={"json_mode": bool(json_mode), "max_tokens": max_tokens,
                           "timeout": timeout, "n_msgs": len(sent_msgs or [])})
    return result, pt, ct


def _trace_out(messages, raw, pt=0, ct=0, seconds=0.0, error=None) -> None:
    """逐题 trace 出口（未开 trace 时**立即返回**，热路径零开销）。

    为什么 lazy import + 提前返回：common/trace.py → common/paths.py → config，
    顶层 import 会绕一圈；而这条路径每次 LLM 调用都会走，必须最便宜。
    角色/阶段各取当前线程/上下文的 `role_ctx` / `phase`。
    """
    try:
        from common import trace as _trace
    except Exception:                       # noqa: BLE001
        return
    if _trace.current_trace() is None:
        return
    _trace.record_llm_call(messages, raw, pt=pt, ct=ct, seconds=seconds, error=error,
                           role=_cur_role(), phase=_cur_phase())


# ── 逐条 LLM 调用审计（默认关：`LLM_CALL_LOG=1`）────────────────────────────────
# 为什么要：现有成本粒度只有「逐题合计」（查询段，results/*.jsonl）和「整段合计」
# （建图段，results/summary.jsonl / build/stats.json）。**建图阶段连一次调用的 token 都
# 没落盘** —— trace 走的是「按题绑定」的线程本地上下文（common/trace.py），建图发生在
# answer_one 之外，`_trace_out` 直接返回。这里补上「单条」这一级：
#
#   LLM_CALL_LOG=1                 **默认开**；=0 关（跑大批量想省空间时用）
#   LLM_CALL_LOG=<路径>            写到指定文件（默认 logs/llm_calls.jsonl；run.py 会给
#                                  本次 run 的 logs/ 目录，见 set_call_log_path）
#   LLM_CALL_LOG_CONTENT=1         **默认开**：存 messages 原文 + 模型原始输出；=0 只留计数
#   LLM_CALL_LOG_EMBED=1           **默认开**：embedding 批次也记（含喂进去的文本）；=0 不记
#   LLM_CALL_LOG_MAX_CHARS=0       **默认 0 = 不截断**；要限长才设（只对 CONTENT 生效）
#
# ★ 一次 run 按**阶段分文件**（run.py 在 `switch_log("query")` 处切路径）：
#     logs/{method}/{dataset}_{sample}_{v}r{m}_build_{stamp}.llm_calls.jsonl
#     logs/{method}/{dataset}_{sample}_{v}r{m}_query_{stamp}.llm_calls.jsonl
#   为什么拆：建图跑完、作答阶段挂掉是常态（单题异常、进程被杀、显存打满）。混在一个文件里
#   就答不了"这批调用里哪些是建图花的、建图到底建成没有"。两段的 `seq` 各自从 1 开始。
#   本文件里的 `phase` 字段仍然保留（build/query），与文件名互为冗余校验。
#
# 关闭时一次字符串判断即返回，热路径开销 ≈ 0；写入失败只报一次、绝不影响主流程。
# 每行 = 一次**尝试**（重试 3 次就有 3 行，含失败那几次的原始输出）；`messages` 是
# **真正发出去的那份**（fit_messages() 截断后的 sent_msgs），`dropped_tokens` 记截掉多少。
# 并发下写入顺序不保证（每行有 seq/ts，可事后排序）。
LLM_CALL_LOG = os.environ.get("LLM_CALL_LOG", "1").strip()
LLM_CALL_LOG_ON = LLM_CALL_LOG.lower() not in ("", "0", "false", "off", "no")
LLM_CALL_LOG_CONTENT = os.environ.get("LLM_CALL_LOG_CONTENT", "1") == "1"
LLM_CALL_LOG_EMBED = os.environ.get("LLM_CALL_LOG_EMBED", "1") == "1"
LLM_CALL_LOG_MAX_CHARS = int(os.environ.get("LLM_CALL_LOG_MAX_CHARS", "0"))
_CALL_LOG = {"path": None, "seq": 0, "fh": None, "warned": False}
_CALL_LOG_LOCK = threading.Lock()


def set_call_log_path(path) -> None:
    """把逐条调用日志指到本次 run 的日志目录（run.py 在 start_log() 之后调用）。

    路径由**知道 run 上下文**的调用方给（method/dataset/时间戳），llm.py 只负责 append
    —— 与 `common.trace.set_trace_dir()` 同一个套路。`LLM_CALL_LOG=0` 时是 no-op。
    """
    if not LLM_CALL_LOG_ON:
        return
    with _CALL_LOG_LOCK:
        if _CALL_LOG["fh"] is not None:
            try:
                _CALL_LOG["fh"].close()
            except Exception:                              # noqa: BLE001
                pass
        _CALL_LOG["fh"] = None
        _CALL_LOG["path"] = Path(path)
        # ★ seq 归零：一次 run 会**切文件**（build 段 → query 段，见 run.py `_use_call_log`），
        #   每份文件都是独立可排序的审计（从 1 开始），不留"前半截在别的文件里"的悬空序号。
        _CALL_LOG["seq"] = 0


def _call_log_default_path() -> Path:
    """没有 setter 时的兜底：`LLM_CALL_LOG` 是路径就用它，否则 `logs/llm_calls.jsonl`。"""
    v = LLM_CALL_LOG
    if v.lower() in ("1", "0", "true", "false", "yes", "no", "on", "off"):
        from common.paths import ROOT                      # 延迟 import：避免顶层绕圈
        return Path(ROOT) / "logs" / "llm_calls.jsonl"
    return Path(v)


def _clip(s, limit: int) -> str:
    s = "" if s is None else str(s)
    if limit and len(s) > limit:
        return s[:limit] + f"...[+{len(s) - limit} chars]"
    return s


def _msg_view(m) -> dict:
    """消息 → {role, content}；非 dict（异常输入）也不让审计路径抛异常。"""
    if isinstance(m, dict):
        return {"role": m.get("role", ""), "content": m.get("content")}
    return {"role": "", "content": m}


def _call_log_out(messages=None, raw=None, pt=0, ct=0, seconds=0.0, error=None,
                  kind="chat", req=None, dropped=None, inputs=None) -> None:
    """逐条调用出口（关掉时**立即返回**，热路径零开销）。

    `messages`/`response` = chat 的输入输出；`inputs` = embedding 批次真正喂进去的文本。
    """
    if not LLM_CALL_LOG_ON or (kind == "embed" and not LLM_CALL_LOG_EMBED):
        return
    try:
        with _CALL_LOG_LOCK:
            path = _CALL_LOG["path"] or _call_log_default_path()
            _CALL_LOG["path"] = path
            _CALL_LOG["seq"] += 1
            rec = {"seq": _CALL_LOG["seq"], "ts": round(time.time(), 3),
                   "phase": _cur_phase(), "role": _cur_role(), "kind": kind,
                   "model": CHAT_MODEL, "pt": int(pt or 0), "ct": int(ct or 0),
                   "seconds": round(seconds, 3)}
            if error:
                rec["error"] = _clip(error, 400)
            if dropped:
                rec["dropped_tokens"] = int(dropped)
            if req:
                rec["req"] = req
            if LLM_CALL_LOG_CONTENT:
                lim = LLM_CALL_LOG_MAX_CHARS
                if messages:
                    rec["messages"] = [{**_msg_view(m),
                                        "content": _clip(_msg_view(m)["content"], lim)}
                                       for m in messages]
                if raw is not None:
                    rec["response"] = _clip(raw, lim)
                if inputs:
                    rec["inputs"] = [_clip(t, lim) for t in inputs]
            try:
                if _CALL_LOG["fh"] is None:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    _CALL_LOG["fh"] = path.open("a", encoding="utf-8")
                _CALL_LOG["fh"].write(json.dumps(rec, ensure_ascii=False) + "\n")
                _CALL_LOG["fh"].flush()
            except Exception as e:                          # noqa: BLE001
                if not _CALL_LOG["warned"]:
                    _CALL_LOG["warned"] = True
                    print(f"[llm] 逐条调用日志写入失败（已忽略，不影响实验）："
                          f"{type(e).__name__}: {e}", flush=True)
    except Exception:                                       # noqa: BLE001
        pass


def _norm_stop(stop) -> list[str] | None:
    """把 `stop` 归一化成 `list[str]`；None / 空 / 全空串 → None（= 不传该字段）。

    官方 baseline 依赖停止串来切分输出，没有它就只能事后用正则猜边界
    （我们过去正是如此，会多生成一整段、还容易猜错）：
      · Self-Ask：`stop=["\\nIntermediate answer:"]`（notebook:220）；
        无结果分支 `stop=['\\nFollow up:', finalans]`（:232,237）；兜底 `stop='\\n'`（:244）
      · IRCoT：`stop=["\\n"]`（gpt3generator.py:87）
    """
    if not stop:
        return None
    if isinstance(stop, str):
        stop = [stop]
    seqs = [s for s in stop if s]
    return seqs or None


def _chat_ollama(system, user, json_mode, timeout, flatten=True, stop=None):
    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
    }
    if json_mode:
        payload["format"] = "json"
    _stop = _norm_stop(stop)
    if _stop:
        payload.setdefault("options", {})["stop"] = _stop   # ollama 把采样参数放 options
    resp = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    pt = data.get("prompt_eval_count", 0)
    ct = data.get("eval_count", 0)
    raw = data["message"]["content"]            # 原始输出（未 strip/未解析），供 trace 留档
    out = _clean_output(_strip_fences(raw), json_mode, flatten=flatten)
    return out, pt, ct, raw


def _chat_vllm(system, user, json_mode, timeout, thinking=True, max_tokens=None,
               flatten=True, stop=None):
    msgs, _dropped = fit_messages(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens)
    payload = {
        "model": CHAT_MODEL,
        "messages": msgs,
        "temperature": 0.0,  # 对齐 hotpotqa：固定温度保证抽取/生成可复现
    }
    if max_tokens:
        payload["max_tokens"] = int(max_tokens)
    if not thinking and _thinking_kwarg_enabled():
        # Qwen3 请求级关思考（vLLM 会把 chat_template_kwargs 传给 chat template）
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    _stop = _norm_stop(stop)
    if _stop:
        payload["stop"] = _stop      # OpenAI 兼容接口：stop 在顶层（ollama 则在 options 里）
    resp = requests.post(f"{VLLM_BASE_URL}/chat/completions", json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    usage = data.get("usage", {})
    pt = usage.get("prompt_tokens", 0)
    ct = usage.get("completion_tokens", 0)
    raw = data["choices"][0]["message"]["content"]   # 原始输出，供 trace 留档
    # ★ flatten 必须透传（曾经漏了 → 调用方传 flatten=False 也被压平换行）。
    # 影响面是实质的：selfask 的 extract_followup 按行扫 —— 被压平后只能得到 1 条，
    # 静默降低方法能力。
    content = _clean_output(_strip_fences(raw), json_mode, flatten=flatten)
    # 注意：这里**不**调 _trace_out —— chat() 的 finally 已经记过一次了（同一调用记两遍
    # 会让 trace 里每条 prompt 重复出现，曾是一个真 bug）。
    return content, pt, ct, raw


# 公开的向量维度：= `config.EMBED_DIM`（由 `EMBED_DIM` 环境变量 / 配置文件给定）。
# 它只是**配置值**；服务端真实维度由 `_embed_dim_check()` 探一次并比对（不一致告警）。
# 适配器（hipporag 的 `embedding_dim=`、各 `np.zeros((0, EMBED_DIM))`）都读这个值。
# 保留旧名 EMBED_DIM 是为了不动 `from common.llm import EMBED_DIM` 的调用点。
EMBED_DIM = EMBED_DIM
_EMBED_ACTUAL_DIM: int | None = None      # 服务端真实维度（探到就缓存）
_EMBED_DIM_WARNED = False                 # 已给出结论（一致 / 不一致 / 探测失败）→ 不再重复

# 两把尺子各自缓存：**不能只用一个全局变量**。`fit_embed`（embedding 侧、
# EMBED_TOKENIZER）与切块（CHUNK_TOKENIZER）会在同一进程里交替调用；共用一个槽位时
# 每次换尺子都要重新 from_pretrained，而且 `_tokenizer_kind` 会被对方顶掉——
# 计数与截断就会用错尺子（口径静默混用）。
_TOKENIZERS: dict[str, object] = {}
# 尺子加载失败的原因（**必须留证**）：失败会让切块静默退化成字符级，而"语料单元"是
# 会写进 manifest / corpus meta 的口径 —— 不记下来，事后没人知道那份图是按字符切的。
_TOKENIZER_FAILURES: dict[str, str] = {}
# 已经打印过"退化为字符级"告警的尺子（每种只打一次，避免逐块刷屏）
_TOKENIZER_FALLBACK_WARNED: dict[str, bool] = {}


def _offline_env_on() -> bool:
    """用户是否显式要求离线（`HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE`）。"""
    return any(os.environ.get(k, "").strip() in ("1", "true", "True", "yes")
               for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"))


def _load_tiktoken():
    """官方口径的 tiktoken 尺子（`*_TOKENIZER=tiktoken` 时才用）。"""
    import tiktoken                                            # type: ignore
    try:
        return tiktoken.encoding_for_model(EMBED_TIKTOKEN_MODEL)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def _load_tokenizer(kind: str):
    """按尺子名加载并缓存：`tiktoken` = 官方原版；其余 = `EMBED_MODEL` 的 HF tokenizer。

    `EMBED_MODEL` 指向本地目录时离线直读；指向 HF 名时走 hf-mirror（国内直连 HF 不通）。
    """
    kind = (kind or "embed").strip().lower()
    if kind in _TOKENIZERS:
        return _TOKENIZERS[kind]
    try:
        tok = _load_tokenizer_once(kind)
    except Exception as e:                                         # noqa: BLE001
        # 失败**留证并缓存**：`None` 会被上面 `in _TOKENIZERS` 命中 → 不再重试、不再刷屏。
        # 于是 `tokenizer_failure()` 在任何时刻都能回答"这台机器上尺子到底行不行"，
        # 不必等真的切过一次块（manifest / 日志头都要靠它，它们可能早于第一次切块）。
        _TOKENIZERS[kind] = None
        _TOKENIZER_FAILURES[kind] = f"{type(e).__name__}: {e}"
        raise
    _TOKENIZERS[kind] = tok
    return tok


def _load_tokenizer_once(kind: str):
    """真正去加载尺子（不做缓存判断）。`tiktoken` = 官方原版；其余 = `EMBED_MODEL` 的 HF tokenizer。"""
    if kind == "tiktoken":
        return _load_tiktoken()
    _is_dir = os.path.isdir(EMBED_MODEL)
    if _is_dir:
        # ★ 先检查目录内容（**在 import transformers 之前**，这样连"没装 transformers"
        #   的机器也能拿到这条更具体的报错）：本地目录 + 缺文件 = 必然失败，而且
        #   `local_files_only=True` 下**没有任何兜底**（以前还能走一次慢速转换）。
        _need = [f for f in ("tokenizer_config.json", "tokenizer.json")
                 if not (Path(EMBED_MODEL) / f).exists()]
        if _need:
            raise FileNotFoundError(
                f"EMBED_MODEL={EMBED_MODEL} 是本地目录，但缺少 {_need} —— "
                f"AutoTokenizer(local_files_only=True) 无法加载。"
                f"请确认目录内容（ls {EMBED_MODEL}），或把 EMBED_MODEL 指向真正含 "
                f"tokenizer_config.json + tokenizer.json 的目录")
    from transformers import AutoTokenizer
    kwargs = {}
    if _is_dir:
        kwargs["local_files_only"] = True      # 本地目录：禁止任何联网尝试
    elif _offline_env_on():
        # 用户已声明离线 + `EMBED_MODEL` 是 HF 名字 → 明确要求只读本地缓存。
        # 不这么做的话 transformers 会去 hf-mirror **重试 5 次、每次 10s 超时**
        # （实测踩过：`Connection to hf-mirror.com timed out`），在无外网的服务器上
        # 白等 30+ 秒，最后仍然失败 —— 而且失败被吞掉、只表现为"切块悄悄变成字符级"。
        # 另外：路径写错（目录不存在）时会落到这里之后的分支，HF 把绝对路径当**仓库 id**
        # 校验，报 `HFValidationError: Repo id must be in the form ...` —— 完全误导。
        kwargs["local_files_only"] = True
    tok = AutoTokenizer.from_pretrained(EMBED_MODEL, **kwargs)
    # 只用它做「按 token 切块」的 offset 计算（split_by_tokens 自己控制上限），
    # 不依赖模型自带的 max length。把 model_max_length 调大，避免编码长段落时
    # 反复打印「879 > 512」的无害警告。
    tok.model_max_length = 10_000_000
    return tok


def _get_tokenizer(kind: str | None = None):
    """取「token 尺子」。两把尺子、两个轴，别混：

    - `embed`（= `EMBED_MODEL` 的 HF tokenizer，默认 bge）：**embedding 侧**单条截断
      （`fit_embed`）必须用它——embedder 自己的 token 数才是它自己的上限。
    - `tiktoken`（`EMBED_TIKTOKEN_MODEL`，默认 gpt-4o-mini）：官方 HyperGraphRAG /
      LightRAG / PRoH 的**原版内部口径**，用于跟官方数字对账（`CHUNK_TOKENIZER`）。

    `kind` 省略时用 `CHUNK_TOKENIZER`（切块尺子）。两把尺子各加载一次并各自缓存。
    """
    return _load_tokenizer(kind or CHUNK_TOKENIZER)


def tokenizer_kind(kind: str | None = None) -> str:
    """尺子的可读名字（`embed:<模型>` / `tiktoken:<编码>`），供日志与 verify 记录。

    ⚠️ `_get_tokenizer` 在加载失败时**返回 None**（失败原因缓存在 `_TOKENIZER_FAILURES`，
    以免每次调用都重试、重刷屏）。所以这里必须先判 `None` 再走鸭子类型：
    `_is_tiktoken()` 的判据是 `not hasattr(tok, "encode_plus")` —— `None` 恰好也没有这个
    属性，会被**误判成 tiktoken**（真踩过：日志里写着"切块尺子 = tiktoken:gpt-4o-mini"，
    而实际是 HF tokenizer 加载失败，把人往错误方向带）。
    """
    try:
        tok = _get_tokenizer(kind)
    except Exception:                                            # noqa: BLE001
        return "unavailable"
    if tok is None:
        return "unavailable"
    if _is_tiktoken(tok):
        return f"tiktoken:{EMBED_TIKTOKEN_MODEL}"
    return f"embed:{EMBED_MODEL}"


def embed_dim() -> int:
    """服务端真实向量维度（探过就缓存；探不到返回配置值 `EMBED_DIM`）。

    为什么值得多一次 HTTP：`EMBED_DIM` 写错（换模型忘改、或 vLLM 那侧
    `--served-model-name` 指到了别的实例）时，向量会以错误的维度流进 numpy/faiss，
    **要么崩在某个莫名其妙的地方，要么静默算出垃圾相似度**。探一次、比一次，最省。
    """
    _embed_dim_check()
    return _EMBED_ACTUAL_DIM or EMBED_DIM


def _embed_dim_check() -> None:
    """首次嵌入时探测 vLLM 返回的真实维度并与 `EMBED_DIM` 比对（只探一次）。"""
    global _EMBED_ACTUAL_DIM, _EMBED_DIM_WARNED
    if _EMBED_ACTUAL_DIM is not None or not EMBED_DIM_CHECK or _EMBED_DIM_WARNED:
        return                      # 探过 / 关了 / 已失败过一次：都不再重复发请求
    try:
        v = _embed_request_one("dim probe")
        _EMBED_ACTUAL_DIM = int(v.shape[0])
    except Exception as e:                                     # noqa: BLE001
        if EMBED_VERBOSE and not _EMBED_DIM_WARNED:
            _EMBED_DIM_WARNED = True
            print(f"[embed] 维度探测失败（{type(e).__name__}: {e}）→ "
                  f"按配置值 EMBED_DIM={EMBED_DIM} 继续（无法校验服务端一致性）",
                  flush=True)
        return
    if _EMBED_ACTUAL_DIM != EMBED_DIM and not _EMBED_DIM_WARNED:
        _EMBED_DIM_WARNED = True
        print(f"[embed] ⚠️ 维度不一致：服务端返回 {_EMBED_ACTUAL_DIM} 维，但 "
              f"EMBED_DIM={EMBED_DIM}。向量会以错误的维度参与相似度计算（结果不可信）。\n"
              f"        修法（改一处即可，然后**重建所有向量** + 开新 --graph-version）：\n"
              f"          · 配置文件里设 \"embed_dim\": {_EMBED_ACTUAL_DIM}\n"
              f"          · 或 export EMBED_DIM={_EMBED_ACTUAL_DIM}\n"
              f"          · 或核对 vLLM 侧 --served-model-name / --task embed 是否指对实例",
              flush=True)
    if EMBED_VERBOSE and not _EMBED_DIM_WARNED:
        print(f"[embed] 服务端维度 {_EMBED_ACTUAL_DIM} 与配置一致；尺子 "
              f"{tokenizer_kind()}；单条截断 {EMBED_MAX_TOKENS} token", flush=True)
    return


def _is_tiktoken(tok) -> bool:
    """尺子是不是 tiktoken：它没有 HF 的 `encode_plus(return_offsets_mapping=)`。

    用鸭子类型判，不靠额外的"当前 kind"全局状态——两把尺子会在同一进程交替使用。
    """
    return not hasattr(tok, "encode_plus")


def _count_tokens_each(texts: list[str], kind: str | None = None) -> list[int]:
    """每条文本的 token 数（尺子 = `_get_tokenizer(kind)`；不可用时退化到词数）。

    `embed_tokens` 统计必须用 **embedder 自己的 tokenizer**（`kind=EMBED_TOKENIZER`），
    否则计费口径与真实消耗不一致：.split() 数的是单词，WordPiece 下 1 词 ≈ 1.3~1.5
    token，会系统性低估。
    """
    try:
        tok = _get_tokenizer(kind)
    except Exception:
        return [len(t.split()) for t in texts]
    if _is_tiktoken(tok):
        return [len(tok.encode(t)) for t in texts]
    return [len(tok.encode_plus(t, add_special_tokens=False)["input_ids"])
            for t in texts]


def _count_tokens(texts: list[str], kind: str | None = None) -> int:
    return sum(_count_tokens_each(texts, kind))


def tokenizer_failure(kind: str | None = None) -> str:
    """尺子加载失败的原因（空串 = 没失败过）。给 manifest / verify 记录用。

    为什么必须能查：tokenizer 加载失败 → `split_by_tokens` 静默退化成**字符级**切块
    （1200 token 会变成 4800 字符），而"语料单元"是要写进 manifest 与 corpus meta 的口径。
    没有这个字段时，图是按字符切的、账上却写着 token，事后完全对不上。

    ★ 会**主动尝试加载一次**（失败原因在模块级缓存）：manifest 往往早于第一次切块生成，
      不主动试的话这个字段永远是空串 —— 而"空串"恰好和"一切正常"长得一样。
    """
    k = (kind or CHUNK_TOKENIZER).strip().lower()
    if k not in _TOKENIZER_FAILURES:
        try:
            _get_tokenizer(k)
        except Exception:                                          # noqa: BLE001
            pass
    return _TOKENIZER_FAILURES.get(k, "")


def _encode_with_offsets(text: str, kind: str | None = None):
    """(ids, offsets) —— 尺子可能是 HF tokenizer 也可能是 tiktoken，这里统一接口。

    tiktoken 没有 `encode_plus(return_offsets_mapping=)`，得自己按 token 的字符串
    累加长度造 offsets（tiktoken 是 lossless 解码，累加长度与原串对齐）。
    """
    tok = _get_tokenizer(kind)
    if _is_tiktoken(tok):
        ids = tok.encode(text)
        out, pos = [], 0
        for t in ids:
            s = tok.decode([t])
            out.append((pos, pos + len(s)))
            pos += len(s)
        return ids, out
    enc = tok.encode_plus(text, add_special_tokens=False, return_offsets_mapping=True)
    return enc["input_ids"], enc["offset_mapping"]


def _decode_ids(ids, kind: str | None = None) -> str:
    """ids → 文本（HF tokenizer 与 tiktoken 的参数名不同）。"""
    tok = _get_tokenizer(kind)
    if _is_tiktoken(tok):
        return tok.decode(list(ids))
    return tok.decode(list(ids), skip_special_tokens=True)


def split_by_tokens(text: str, max_tokens: int = 400, overlap_tokens: int = 40,
                    kind: str | None = None):
    """按 token 尺子切块，返回 [(chunk_text, start_char, end_char)]。

    `kind` 默认 = `CHUNK_TOKENIZER`（切块尺子）；`fit_embed` 会显式传
    `EMBED_TOKENIZER`（embedder 自己的 tokenizer）。
    字符区间来自原文本的 token offsets（供句子定位）。
    tokenizer 不可用时退化到字符级切块（英文 1 token ≈ 4 字符）。

    ★ **`max_tokens <= 0` = 不切块**：整段原样返回一个块（`EXTRACT_CHUNK_TOKENS=0`
    就是这个口径）。此时 `overlap_tokens` 无意义、tokenizer 也不需要用。
    """
    text = text.strip()
    if not text:
        return []
    if max_tokens <= 0:
        # 不切块：一个 context（标题 + 正文）= 一个块，原样送去抽取/检索/embed。
        return [(text, 0, len(text))]
    try:
        ids, offsets = _encode_with_offsets(text, kind)
    except Exception as e:                                         # noqa: BLE001
        # ⚠️ 这一步是**静默改口径**的地方，所以必须响一次、且留证：
        #    tokenizer 不可用 → 下面按「英文 1 token ≈ 4 字符」切，于是 `1200 token` 实际
        #    变成 **4800 字符**、块数与"按 token 切"完全不同。日志头那句"切块可能退化为
        #    字符级"是唯一的提示，而它离现场太远。这里直接打一次（每种尺子只打一次）。
        _k = (kind or CHUNK_TOKENIZER).strip().lower()
        _TOKENIZER_FAILURES.setdefault(_k, f"{type(e).__name__}: {e}")
        if not _TOKENIZER_FALLBACK_WARNED.get(_k):
            _TOKENIZER_FALLBACK_WARNED[_k] = True
            print(f"[tokenizer] ⚠️ 尺子 '{_k}' 不可用（{type(e).__name__}: {e}）"
                  f"\n[tokenizer] → 切块**退化为字符级**：{max_tokens} token 会按 "
                  f"{max_tokens * 4} 字符切（英文 1 token ≈ 4 字符），块数与按 token 切不同。"
                  f"\n[tokenizer] → 语料单元口径已偏离；修法：把 EMBED_MODEL 指向含 "
                  f"tokenizer_config.json 的**本地目录**（并 export HF_HUB_OFFLINE=1）。"
                  f"\n[tokenizer] → 当前 EMBED_MODEL={EMBED_MODEL}", flush=True)
        ids, offsets = None, None

    if ids is None or not ids:
        # 退化：字符级切块
        size = max_tokens * 4
        step = size - overlap_tokens * 4
        out = []
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            out.append((text[start:end], start, end))
            if end >= len(text):
                break
            start += step
        return out

    if len(ids) <= max_tokens:
        return [(text, 0, len(text))]
    step = max_tokens - overlap_tokens
    starts = list(range(0, len(ids) - max_tokens + 1, step)) or [0]
    # 收尾：`range()` 的步长通常除不尽，**最后一个窗往往没贴到文末**（2400 token、1200/100 →
    # range 只给 [0, 1100]，`1100+1200=2300 < 2400`，尾巴没人覆盖）。
    # 口径（用户确认）：**按固定步长 `step` 再走一格**当最后一块，它切到文末即止、不必满 max。
    #   2400、1200/100 → 起点 [0, 1100, 2200] → 块 [0..1199] [1100..2299] [2200..2399]
    #   （最后一块只有 200 token，这是预期的，不是 bug）
    # 为什么不是「再 append 一个贴右端的窗（`len-max`）」：那会得到 [1100..2299] 与 [1200..2399]
    #   两个几乎全等的块 —— 白烧一次嵌入，检索侧还会出现两份近似文本互相挤占名次。
    # 为什么不是「把上一个窗的起点挪到 `len-max`」：会让倒数第二个块起点偏移、丢失「等距滑窗」
    #   这个可复现性质；等距推进 + 末块截断更容易解释，块数也更少。
    if starts[-1] + max_tokens < len(ids):
        starts.append(starts[-1] + step)
    out = []
    for s in starts:
        e = min(s + max_tokens, len(ids))
        cs = offsets[s][0]
        ce = offsets[e - 1][1]
        out.append((_decode_ids(ids[s:e], kind), cs, ce))
    return out


def _embed_request(inputs: list[str], timeout: int):
    """调 vLLM embedding 端点（OpenAI 兼容），返回 (N, dim) float32 未归一化。

    失败时把 vLLM 的响应体带进异常（400 的原因都在里面：
    常见的是没加 --task embed 或 served-model-name 不匹配）。
    """
    _embed_dim_check()          # 首次调用时探一次服务端维度并与 EMBED_DIM 比对
    resp = requests.post(
        f"{VLLM_EMBED_BASE_URL}/embeddings",
        json={"model": EMBED_MODEL_SERVED, "input": inputs},
        timeout=timeout,
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"embedding 请求失败 {resp.status_code}: {resp.text[:400]}"
        )
    data = sorted(resp.json()["data"], key=lambda d: d["index"])
    return np.array([d["embedding"] for d in data], dtype=np.float32)


def _embed_request_one(text: str, timeout: int = 30) -> np.ndarray:
    """单条 embedding 原始请求（**不加查询前缀、不归一化、不计费**）。供维度探测用。"""
    resp = requests.post(
        f"{VLLM_EMBED_BASE_URL}/embeddings",
        json={"model": EMBED_MODEL_SERVED, "input": [text]},
        timeout=timeout,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"embedding 探测失败 {resp.status_code}: {resp.text[:200]}")
    return np.array(resp.json()["data"][0]["embedding"], dtype=np.float32)


def _normalize_rows(vecs: np.ndarray) -> np.ndarray:
    """vLLM 返回未归一化向量，客户端归一化（对齐原 sentence-transformers 行为）。"""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def query_prefix(text: str, instruction: str | None = None) -> str:
    """按 `EMBED_QUERY_FORMAT` 给查询文本加前缀（bge 指令句 / Qwen3 Instruct 格式）。

    集中的一个原因：换 embedding 模型时**查询侧前缀必须跟着换**，而它散在
    `embed()` 和 `test_layer.py` 两处（曾经就是两处各写一遍）。
    """
    text = str(text)
    if EMBED_QUERY_TEMPLATE:
        return EMBED_QUERY_TEMPLATE.format(task=EMBED_QUERY_TASK, text=text)
    fmt = (EMBED_QUERY_FORMAT or "bge").lower()
    if fmt == "none":
        return text
    if fmt == "qwen3":
        # 模板逐字取自**模型自己带的** `config_sentence_transformers.json`：
        #   "prompts": {"query": "Instruct: {task}\nQuery:", "document": ""}
        # sentence-transformers 把 prompt 与正文**直接拼接**（不插分隔符）→ 实际输入是
        # `Instruct: {task}\nQuery:{text}` —— `Query:` 后**无空格**。
        # ⚠️ 出处更正：README 的 `get_detailed_instruct()` 写的是 `Query: {query}`（**有**空格），
        #   与模型随附的 ST 配置**不一致**。本项目采用后者 —— 那是 ST 实际会施加的模板。
        #   （本地可查：<模型目录>/config_sentence_transformers.json）
        # 曾经这里写成 `Query: {text}`（多一个空格）→ 多切一个 token、与 ST 口径不符。
        return f"Instruct: {EMBED_QUERY_TASK}\nQuery:{text}"
    return (EMBED_QUERY_INSTRUCTION if instruction is None else instruction) + text


def fit_embed(text: str, max_tokens: int | None = None) -> str:
    """按 token 尺子截断到 `max_tokens`（默认 = 配置 `EMBED_MAX_TOKENS`，480 起）。

    LLM 生成的实体/关系描述可能超过 embedder 的单条上限，vLLM embed 端会直接 400
    拒绝；外部适配器（lightrag/graphrag/hypergraphrag）嵌入这类文本前先截断，
    避免整个实体/关系的向量被丢弃。
    """
    n = EMBED_MAX_TOKENS if max_tokens is None else int(max_tokens)
    for chunk, _cs, _ce in split_by_tokens(text, n, 0, kind=EMBED_TOKENIZER):
        return chunk
    return ""


def embed(text: str, timeout: int = 30) -> np.ndarray:
    """单条 embedding（query 侧：按 `EMBED_QUERY_FORMAT` 加前缀 + 归一化）。"""
    p = _cur_phase()
    t0 = time.time()
    err = None
    full = text
    _n = 0
    try:
        full = query_prefix(text)
        _n = _count_tokens([full], EMBED_TOKENIZER)
        _note_embed_len([full], [_n])
        vec = _normalize_rows(_embed_request([full], timeout))[0]
        b = _get_buckets()[p]
        b["embed_calls"] += 1
        b["embed_tokens"] += _n
        return vec
    except Exception as e:                       # noqa: BLE001 —— 审计要记失败原因，原样抛出
        err = f"{type(e).__name__}: {e}"
        raise
    finally:
        _get_buckets()[p]["embed_seconds"] += time.time() - t0
        _call_log_out(pt=_n, seconds=time.time() - t0, error=err, kind="embed",
                      req={"n": 1, "model": EMBED_MODEL_SERVED, "timeout": timeout,
                           "query_side": True},
                      inputs=[full])


def vllm_token_count() -> dict:
    """从 vLLM /metrics 读累计 token（chat + embed 两个端点）。

    返回 {"chat": {"prompt": int, "completion": int}, "embed": {...}}；
    端点不可达或指标缺失时为 None。建图前后各取一次做差值 = 本次建图
    的真实 LLM/embedding token——不依赖项目进程内存统计，崩溃也不丢。
    """
    out = {}
    for name, base in (("chat", VLLM_BASE_URL), ("embed", VLLM_EMBED_BASE_URL)):
        try:
            # http://host:port/v1 -> http://host:port（metrics 挂在根路径）
            root = base.rsplit("/v1", 1)[0]
            resp = requests.get(f"{root}/metrics", timeout=10)
            resp.raise_for_status()
            p = c = None
            for line in resp.text.splitlines():
                if line.startswith("#") or "=" not in line:
                    continue
                if "vllm:prompt_tokens_total" in line:
                    p = int(line.split()[-1])
                elif "vllm:generation_tokens_total" in line:
                    c = int(line.split()[-1])
            out[name] = {"prompt": p, "completion": c}
        except Exception:
            out[name] = None
    return out


# ── embedder 单条长度**实测**（用来定服务端 `--max-model-len`）──
# 为什么要有：`EMBED_MAX_TOKENS` 必须 ≤ 服务端 `--max-model-len`，而"各方法实际会嵌多长"
# 是个**可测**的量，不该靠推导（切块 400 / 摘要 500 / 命题 summary 都是上界，不是实测）。
# `embed_batch` 本来就算了每条 token 数（用于切请求预算），顺手记下最大值即可；
# run 结束时打印 → 直接拿去设服务端参数，而不是猜。
_EMBED_LEN = {"n": 0, "max": 0, "head": "", "over_2048": 0, "over_4096": 0}


def _note_embed_len(texts, counts) -> None:
    for t, n in zip(texts, counts):
        n = int(n)
        _EMBED_LEN["n"] += 1
        if n > _EMBED_LEN["max"]:
            _EMBED_LEN["max"] = n
            _EMBED_LEN["head"] = str(t)[:70].replace("\n", " ")
        if n > 2048:
            _EMBED_LEN["over_2048"] += 1
        if n > 4096:
            _EMBED_LEN["over_4096"] += 1


def embed_len_stats() -> dict:
    """本次进程喂给 embedder 的单条 token 长度（最大值 + 超 2048/4096 的条数）。"""
    return dict(_EMBED_LEN)


def _split_by_token_budget(texts: list[str], counts: list[int],
                           budget: int = EMBED_REQUEST_TOKENS):
    """把文本按 token 预算（+ 条数上限）切成若干个 HTTP 请求体。

    vLLM 对 pooling/embedding 模型的请求：token 总量超过引擎的
    --max-num-batched-tokens 时既不报错也不执行，永久静默挂住
    （vllm-project/vllm#41969）→ 客户端必须自己保证单请求装得下。
    单条就超预算的单独成组（embedding 前都 fit_embed 过，正常不会出现）。
    """
    parts: list[list[str]] = []
    cur: list[str] = []
    cur_tok = 0
    for t, n in zip(texts, counts):
        n = max(int(n), 1)
        if cur and (cur_tok + n > budget or len(cur) >= EMBED_BATCH_SIZE):
            parts.append(cur)
            cur, cur_tok = [], 0
        cur.append(t)
        cur_tok += n
    if cur:
        parts.append(cur)
    return parts


# ── 自适应并发闸（AIMD + 引擎指标）──
# 本地 vLLM 的并发上限其实由 KV 池决定：并发超过「KV 装得下的请求数」时，多出来的
# 请求排队，延迟被顶到超时线，然后整批重跑，时间全浪费。与其拍一个静态并发数、
# 也不等 300s 超时，不如**直接读引擎自己的排队指标**：
#   - 超时 / 引擎 waiting>0 / KV≥95% → 并发减半（下限 1）
#   - 重试 N 次仍失败 → 退到串行（1），抽完一块再下一块
#   - 连续成功满一批且引擎不拥堵 → 并发翻倍（封顶 = 配置的并发数）
GATE_METRICS = os.environ.get("LLM_GATE_METRICS", "1") == "1"      # 0=只靠超时信号
GATE_METRICS_SEC = float(os.environ.get("LLM_GATE_METRICS_SEC", "5"))
GATE_KV_HIGH = float(os.environ.get("LLM_GATE_KV_HIGH", "0.95"))
# 引擎空闲水位：waiting=0 且 KV 低于它 → 主动把并发加回去（不用等成功攒满一批）
GATE_KV_LOW = float(os.environ.get("LLM_GATE_KV_LOW", "0.5"))
# 空闲时的加性增比例（AIMD：降是乘性、快；升是加性、慢，才不会冲过头再被打回来）
GATE_GROW_STEP = float(os.environ.get("LLM_GATE_GROW_STEP", "0.25"))
# 防抖动：waiting>0 要连续这么多次采样才减半；减半后先观察这么多次采样再决定是否再减
GATE_STREAK = int(os.environ.get("LLM_GATE_STREAK", "2"))
# 任何一次减半之后的冷却期：期间其它减半请求（含一批同时超时）一律忽略，
# 防止 10 个请求同时超时把并发一口气砸到 1
GATE_COOLDOWN_SEC = float(os.environ.get("LLM_GATE_COOLDOWN_SEC",
                                         str(max(GATE_METRICS_SEC * GATE_STREAK, 5.0))))
# 单次调用（含它自己的重试）的总时间预算：超了就放弃，别让一块拖 5×300s
LLM_RETRY_BUDGET_SEC = float(os.environ.get("LLM_RETRY_BUDGET_SEC", "600"))
# 等并发槽超过这么多秒就打一次告警（vLLM 空闲却推进不动 → 客户端卡住的现场证据）
GATE_WAIT_WARN_SEC = float(os.environ.get("LLM_GATE_WAIT_WARN_SEC", "30"))
# 失败后重试几次（2 = 首次 + 重试 1 次）。重试不改并发，只减半输出上限
LLM_RETRY_ATTEMPTS = int(os.environ.get("LLM_RETRY_ATTEMPTS", "2"))

_BACKPRESSURE_HINTS = ("timeout", "timed out", "connectionerror", "connection reset",
                       "remote end closed", "remotedisconnected", "brokenpipe",
                       "connection aborted", "eof occurred")


def is_backpressure(sig: str) -> bool:
    """这次失败是不是「排队/连接」类（退让并发有用），而不是请求本身被拒（400）。"""
    s = (sig or "").lower()
    return any(h in s for h in _BACKPRESSURE_HINTS)


def vllm_engine_stats(timeout: int = 5):
    """读 vLLM `/metrics`：返回 (KV 使用率 0~1, running, waiting)；拿不到返回 None。

    并发闸靠它**在超时之前**就发现引擎在排队（官方日志里那行
    `Running: 15 reqs, Waiting: 17 reqs, GPU KV cache usage: 99.6%`）。
    """
    try:
        root = VLLM_BASE_URL.rsplit("/v1", 1)[0]
        resp = requests.get(f"{root}/metrics", timeout=timeout)
        resp.raise_for_status()
    except Exception:
        return None
    kv, running, waiting = 0.0, 0, 0
    got = False
    for line in resp.text.splitlines():
        if line.startswith("#") or " " not in line:
            continue
        if "num_requests_waiting" in line:
            waiting, got = int(float(line.split()[-1])), True
        elif "num_requests_running" in line:
            running, got = int(float(line.split()[-1])), True
        elif "gpu_cache_usage_perc" in line or "kv_cache_usage_perc" in line:
            kv, got = float(line.split()[-1]), True
    return (kv, running, waiting) if got else None


class AdaptiveLimiter:
    """AIMD 并发闸：`async with gate:` 占一个槽，成功/超时回报给它。

    规则：
    - **vLLM 引擎自己在排队就减半**：每 `GATE_METRICS_SEC` 秒读一次 `/metrics`，
      `waiting` 连续 `GATE_STREAK` 次 >0，或一次越过硬阈值
      （`waiting ≥ max(2, 并发/4)` 或 KV ≥ `GATE_KV_HIGH`）→ 减半（下限 1）；
      每次减半后先观察 `GATE_STREAK` 次采样再决定要不要继续减；
    - 连续成功满 max(当前并发, recover_window) 次且**引擎不拥堵** → 并发**翻倍**，
      封顶 = 构造时的 limit。

    只认引擎指标——**请求重试不改并发**（重试策略见 `chat_retry`）。
    只在**真正发 HTTP 的路径**上加闸，缓存命中不占槽。
    """

    def __init__(self, limit: int, tag: str = "llm", recover_window: int = 4):
        self.max_limit = max(1, int(limit))
        self.limit = self.max_limit
        self.recover_window = max(1, int(recover_window))
        self.tag = tag
        self._inflight = 0
        self._ok_run = 0
        self._cond = asyncio.Condition()
        self._print_lock = threading.Lock()
        self._monitor = None            # 指标轮询 task（懒建，绑当前 loop）
        self._warned_wait = False       # 等槽告警只报一次（拿到槽后复位）
        self._congested = False         # 引擎最近一次采样是否在排队
        self._streak = 0                # waiting>0 连续采样次数
        self._last_halve = 0.0          # 上次减半的时刻（冷却期用）
        # ★ 实际并发轨迹（动态并发下"并发是多少"是个分布，不是 workers 那个数）
        #   在持锁期间采样 inflight，最后按时间加权出均值/峰值，写进 summary。
        self._conc_bins: dict[int, float] = {}   # inflight → 持续秒数
        self._conc_peak = 0
        self._conc_last_t: float | None = None
        self._conc_wait_s = 0.0         # 累计"等并发槽"的时间（客户端侧开销，不是 GPU）

    def _sample_conc(self):
        """记一段「上一档 inflight 持续了多久」。调用方必须已持 `self._cond` 的锁。

        ⚠️ 顺序很重要：**先用当前 inflight 结算上一段时间，再改 inflight**。
        反过来写会把每段时长都记到「改完之后」的那一档（并且丢掉第一档）——踩过。
        inflight=0 也记一档：那是引擎空闲（对 GPU 利用率有解释力）。
        """
        now = time.monotonic()
        if self._conc_last_t is not None:
            dt = now - self._conc_last_t
            if 0 < dt < 60:                      # 窗口异常（进程挂起）→ 丢弃
                self._conc_bins[self._inflight] = self._conc_bins.get(self._inflight, 0.0) + dt
        self._conc_last_t = now
        self._conc_peak = max(self._conc_peak, self._inflight)

    def concurrency_stats(self) -> dict:
        """实际并发分布：时间加权均值 / 峰值 / 各档占比 / 等槽耗时 / 当前 limit。"""
        bins = dict(self._conc_bins)
        total = sum(bins.values())
        peak = self._conc_peak
        if not bins:
            return {"available": False, "limit": self.limit, "max_limit": self.max_limit,
                    "peak": peak, "wait_for_slot_seconds": round(self._conc_wait_s, 2)}
        avg = sum(k * v for k, v in bins.items()) / total
        return {
            "available": True,
            "avg": round(avg, 2),                 # ← 论文里该报这个，而不是 --workers
            "peak": peak,
            "limit_now": self.limit, "max_limit": self.max_limit,
            "gate_used": True,
            "hist": {str(k): round(v, 1) for k, v in sorted(bins.items())},
            "time_weighted_seconds": round(total, 1),
            "wait_for_slot_seconds": round(self._conc_wait_s, 2),
            # 客户端时长 / GPU busy ≈ 排队放大倍数（≈1 = 几乎没排队）
            "note": "avg 是时间加权实际在飞请求数；以它而不是 --workers 报并发",
        }

    def _log(self, msg: str):
        with self._print_lock:
            print(f"[{self.tag}] 自适应并发：{msg}（上限 {self.max_limit}）", flush=True)

    async def __aenter__(self):
        self._ensure_monitor()
        t0 = time.monotonic()
        async with self._cond:
            while self._inflight >= self.limit:
                await self._cond.wait()
                waited = time.monotonic() - t0
                if waited > GATE_WAIT_WARN_SEC and not self._warned_wait:
                    self._warned_wait = True
                    print(f"[{self.tag}] ⚠️ 等并发槽已 {waited:.0f}s"
                          f"（inflight={self._inflight} limit={self.limit}）→ "
                          f"若 vLLM 显示 Running=0 就是客户端卡住："
                          f"要么闸被占满，要么常驻 loop 线程被同步操作堵住"
                          f"（用 py-spy dump 或看 logs/ 里的线程栈）", flush=True)
            self._sample_conc()                            # 先按旧 inflight 结算上一段
            self._inflight += 1
            self._conc_wait_s += time.monotonic() - t0     # 等槽耗时（客户端开销）
            self._warned_wait = False
        return self

    async def __aexit__(self, *exc):
        async with self._cond:
            self._sample_conc()                            # 先结算当前档
            self._inflight -= 1
            self._cond.notify()
            # 没有任何在途请求、且最近一次采样没报拥堵 → 并发**直接恢复到上限**。
            # 否则被减半后只能靠监控每 5s +25% 慢慢爬（1→2→3… 要几分钟），
            # 而这段时间引擎全空闲，纯属白等。
            if (self._inflight == 0 and self.limit < self.max_limit
                    and not self._congested):
                old = self.limit
                self.limit = self.max_limit
                self._ok_run = 0
                self._streak = 0
                self._cond.notify_all()
                self._log(f"无在途请求 → 恢复到上限（{old} → {self.limit}）")

    # ── 拥堵信号：直接问 vLLM 引擎（waiting / KV），不用等超时 ──

    def _ensure_monitor(self):
        if not GATE_METRICS or self._monitor is not None:
            return
        try:
            self._monitor = asyncio.get_running_loop().create_task(self._monitor_loop())
        except RuntimeError:            # 没有运行中的 loop（理论上到不了这里）
            self._monitor = None

    async def _monitor_loop(self):
        while True:
            await asyncio.sleep(GATE_METRICS_SEC)
            await self._monitor_step()

    async def _monitor_step(self):
        """采一次引擎指标并决定是否减半（拆出来是为了能确定性地单测）。

        A 方案（防抖动）：waiting 连续 GATE_STREAK 次 >0，或一次性越过硬阈值
        （waiting ≥ max(2, 并发/4)，或 KV ≥ GATE_KV_HIGH）才减半；每次减半后先观察
        GATE_STREAK 次采样（cooldown），再决定要不要继续减。返回是否减半。
        """
        try:
            st = await asyncio.to_thread(vllm_engine_stats)
        except Exception:
            st = None
        if not st:
            return False
        kv, running, waiting = st
        self._streak = self._streak + 1 if waiting > 0 else 0
        hard = kv >= GATE_KV_HIGH or waiting >= max(2, self.limit // 4)
        congested = hard or self._streak >= GATE_STREAK
        async with self._cond:
            self._congested = congested
        if not congested:
            self._streak = 0
            # 引擎明显空闲（waiting=0 且 KV 低于水位）→ 主动加回去。否则并发被压到 1
            # 之后只能等"连续成功满一批"才翻倍，而慢块本身就要跑很久 → 引擎长期空转。
            if self._inflight == 0 and self.limit < self.max_limit:
                await self._grow(f"无在途请求（waiting=0）", to_max=True)
            elif waiting == 0 and kv < GATE_KV_LOW:
                await self._grow(f"引擎空闲（waiting=0 KV={kv:.0%}）")
            return False
        await self._halve(f"引擎排队（waiting={waiting} running={running} "
                          f"KV={kv:.0%}{'，硬阈值' if hard else f'，连续{self._streak}次'}）")
        return True

    async def _halve(self, why: str):
        """减半；冷却期内（GATE_COOLDOWN_SEC）忽略后续减半请求，返回是否真的减了。"""
        async with self._cond:
            now = time.monotonic()
            if now - self._last_halve < GATE_COOLDOWN_SEC or self.limit <= 1:
                return False
            old, self.limit = self.limit, max(1, self.limit // 2)
            self._ok_run = 0
            self._last_halve = now
            self._cond.notify_all()
            self._log(f"{why} → {old} 减半到 {self.limit}")
            return True

    async def on_timeout(self):
        """一次排队/连接类失败（超时、断连）：也让并发退让一步。

        指标是"提前发现"（每 GATE_METRICS_SEC 采样），这条是"事后兜底"——
        指标拿不到时它是唯一的降并发信号。带冷却期，所以一批请求同时超时只会减一次。
        """
        await self._halve("请求超时")

    async def _grow(self, why: str, to_max: bool = False):
        """加性增（每次约 +GATE_GROW_STEP，至少 +1）：升得慢，避免冲过头又被减半。

        `to_max=True`：目前没有任何在途请求时用——直接回到上限（引擎必然空闲）。
        """
        async with self._cond:
            if self.limit < self.max_limit:
                old = self.limit
                self.limit = self.max_limit if to_max else min(
                    self.max_limit, self.limit + max(1, int(self.limit * GATE_GROW_STEP)))
                self._ok_run = 0
                self._cond.notify_all()
                self._log(f"{why} → {old} 加到 {self.limit}")

    async def on_success(self):
        """一次成功：攒够一整批（= 当前并发数，至少 recover_window）且引擎不拥堵就翻倍。"""
        async with self._cond:
            if self._congested:         # 引擎还在排队，别急着加回去
                self._ok_run = 0
                return
            self._ok_run += 1
            need = max(self.limit, self.recover_window)
            if self.limit < self.max_limit and self._ok_run >= need:
                old, self.limit = self.limit, min(self.max_limit, self.limit * 2)
                self._ok_run = 0
                self._cond.notify_all()
                self._log(f"连续 {old} 次无超时 → 加倍到 {self.limit}")

    async def force_serial(self, why: str = "手动"):
        """退到串行：并发降到 1，一次只跑一个请求，跑完再放下一个。"""
        async with self._cond:
            if self.limit != 1 or self._ok_run:
                old = self.limit
                self.limit = 1
                self._ok_run = 0
                self._cond.notify_all()
                self._log(f"{why} → 退出并发（{old} → 1），一次只抽一块")


_GATE: AdaptiveLimiter | None = None


def gate_for(limit: int, tag: str = "llm", recover_window: int = 4) -> AdaptiveLimiter:
    """进程内共用的并发闸：**首次调用决定** 上限/tag，之后一律返回同一个实例
    （一个进程只跑一个方法，所以不需要多实例；重复传参不会重建，别指望改它）。"""
    global _GATE
    if _GATE is None:
        _GATE = AdaptiveLimiter(limit, tag=tag, recover_window=recover_window)
    return _GATE


def concurrency_stats() -> dict:
    """本进程实际并发分布（动态并发下「并发是多少」是分布，不是 --workers）。

    **优先用全局 LLM 调用计数器**（`_LLM_CONC`）：所有方法都经过 `chat()` /
    `chat_messages()`，而自适应闸只有部分 adapter 用（standardrag 直连 chat，不经闸）。
    闸那份统计作为兜底（它还能给出「等槽耗时」）。
    """
    best = _LLM_CONC.stats()
    if best.get("available"):
        return best
    if _GATE is None:
        return {"available": False, "note": "本次 run 没有任何 LLM 调用？（也未走自适应闸）"}
    return _GATE.concurrency_stats()


# ── 全局 LLM 在飞计数（覆盖所有调用路径，和并发闸无关）──
class _InflightTracker:
    """按「调用完成时刻的在飞数」做时间加权：样本 = 每次 LLM 调用结束时记一档。

    与闸那套的区别：闸只有用它的 adapter 才统计（standardrag 直连 chat 不经闸），
    这里的计数在 `chat()` / `chat_messages()` 里，**所有方法都覆盖**。
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        with self._lock:
            self.inflight = 0
            self.peak = 0
            self.bins: dict[int, float] = {}
            self._last_t = None
            self._t0 = time.monotonic()

    def start(self):
        with self._lock:
            # ★ 顺序：**先结算「上一段」（此时 inflight 还是旧值）再自增**。
            #   反过来（先改后结算）会把每段时长都记到「改完之后」那一档，于是
            #   一次不重叠的调用会把**整段**记进 inflight=0 → avg=avg_while_active=0，
            #   症状就是「并发显示 0 但回答一直在出」（peak 仍是对的，因为它在自增后取 max）。
            #   同一个坑 AdaptiveLimiter._sample_conc（:1249）写着「踩过」，当时只修了闸，
            #   覆盖全部方法的这个计数器漏了 —— 详见 _t_conc_bug 判别性测试。
            self._bump()
            self.inflight += 1
            self.peak = max(self.peak, self.inflight)

    def end(self):
        with self._lock:
            self._bump()                 # ★ 同上：先结算，再自减
            self.inflight = max(0, self.inflight - 1)

    def _bump(self):
        """按当前 inflight 结算上一段时间（调用方必须持锁）。"""
        now = time.monotonic()
        if self._last_t is not None:
            dt = now - self._last_t
            if 0 < dt < 60:
                self.bins[self.inflight] = self.bins.get(self.inflight, 0.0) + dt
        self._last_t = now

    def stats(self) -> dict:
        with self._lock:
            bins = dict(self.bins)
            # 收尾：把最后一段（到调用时刻为止）也算上
            now = time.monotonic()
            if self._last_t is not None:
                dt = now - self._last_t
                if 0 < dt < 60:
                    bins[self.inflight] = bins.get(self.inflight, 0.0) + dt
            peak = self.peak
            elapsed = now - self._t0
            wait = _GATE._conc_wait_s if _GATE is not None else 0.0
        total = sum(bins.values())
        if not bins or total <= 0:
            return {"available": False, "note": "没有采样到 LLM 在飞窗口"}
        busy = {k: v for k, v in bins.items() if k > 0}
        avg = sum(k * v for k, v in bins.items()) / total
        avg_busy = (sum(k * v for k, v in busy.items()) / sum(busy.values())
                    if busy else 0.0)
        return {
            "available": True,
            "source": "llm_calls",              # 统计来源：LLM 调用级（覆盖全部方法）
            "avg": round(avg, 2),               # 含空闲段
            "avg_while_active": round(avg_busy, 2),   # ← 论文更该报这个（工作时段的并发）
            "peak": peak,
            "hist": {str(k): round(v, 1) for k, v in sorted(bins.items())},
            "sampled_seconds": round(total, 1),
            "elapsed_seconds": round(elapsed, 1),
            "wait_for_slot_seconds": round(wait, 2),
            # 并发闸的两档（只有走闸的 adapter 才有；直连 chat 时为 None）
            "max_limit": _GATE.max_limit if _GATE is not None else None,
            "limit_now": _GATE.limit if _GATE is not None else None,
            "gate_used": _GATE is not None,
            "note": "avg_while_active = 有在飞请求时段的加权并发；以它而不是 --workers 报并发",
        }


_LLM_CONC = _InflightTracker()


def reset_concurrency():
    """清零并发统计（run.py 在开跑前调；默认自动）。"""
    _LLM_CONC.reset()


def _conc_start():
    _LLM_CONC.start()


def _conc_end():
    _LLM_CONC.end()


async def chat_retry(messages: list[dict], *, gate: AdaptiveLimiter,
                     json_mode: bool = False, timeout: int = 300,
                     max_tokens: int | None = 4096, attempts: int | None = None,
                     budget_sec: float | None = None):
    """`chat_messages` + 重试 + 自适应并发的**完整策略**——适配器只调这一个。

    - **并发只由引擎指标决定**（`AdaptiveLimiter` 的监控：waiting/KV 减半、连续成功
      且不堵翻倍）。重试**不改变并发**——重试就是重试。
    - **失败只重试 `LLM_RETRY_ATTEMPTS`（默认 2 = 首次 + 1 次重试）**，重试**原样重发**
      （不改输出上限、不改 prompt——要的就是同一次请求再试一遍）。
    - **总时间预算** `budget_sec`（默认 `LLM_RETRY_BUDGET_SEC`，600s）：这一次调用的
      所有尝试加起来超过它就停止重试；每次尝试的超时值也会按剩余预算收紧。
    - **输出上限兜底 4096**（`chat_messages` 默认），防个别块陷入重复循环一直占 KV。

    返回 (content, prompt_tokens, completion_tokens)；全部失败时 content 为 None。
    """
    budget = LLM_RETRY_BUDGET_SEC if budget_sec is None else budget_sec
    attempts = LLM_RETRY_ATTEMPTS if attempts is None else attempts
    t0 = time.monotonic()
    for attempt in range(attempts):
        left = None if not budget else budget - (time.monotonic() - t0)
        if left is not None and left <= 1:
            print(f"[{gate.tag}] 重试预算 {budget:.0f}s 用尽（已试 {attempt} 次），放弃这次调用",
                  flush=True)
            break
        err: list = []
        async with gate:
            content, pt, ct = await asyncio.to_thread(
                chat_messages, messages, json_mode=json_mode,
                timeout=timeout if left is None else max(1, min(timeout, int(left))),
                max_tokens=max_tokens, err_out=err)
            if content is not None:
                await gate.on_success()
                return content, pt, ct
            if err:
                print(f"[{gate.tag}] 第 {attempt + 1}/{attempts} 次失败：{err[0][:80]}",
                      flush=True)
                if is_backpressure(err[0]):      # 超时/断连 → 让并发退让一步（带冷却期）
                    await gate.on_timeout()
            if attempt < attempts - 1:          # 后面没有尝试了就不用再等
                back = 2 ** attempt
                await asyncio.sleep(back if left is None else max(0, min(back, left - 1)))
    return None, 0, 0


def embed_batch(texts: list[str], timeout: int = 120) -> np.ndarray:
    """批量 embedding（passage 侧：不加指令 + 归一化）。

    按 token 预算 + EMBED_BATCH_SIZE 条数切请求（见 _split_by_token_budget）。
    """
    if not texts:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)
    p = _cur_phase()
    t0 = time.time()
    err = None
    sent: list[str] = []
    ntok = 0
    try:
        texts = list(texts)
        if EMBED_DOC_PREFIX:                    # 文档侧前缀（bge/Qwen 为空；e5 之类才需要）
            texts = [EMBED_DOC_PREFIX + t for t in texts]
        sent = texts
        counts = _count_tokens_each(texts, EMBED_TOKENIZER)
        ntok = sum(counts)
        _note_embed_len(texts, counts)          # ★ 记录"实际喂给 embedder 的最长单条"
        parts = [_embed_request(batch, timeout)
                 for batch in _split_by_token_budget(texts, counts)]
        vecs = _normalize_rows(np.concatenate(parts, axis=0))
        b = _get_buckets()[p]
        b["embed_calls"] += len(parts)
        b["embed_tokens"] += sum(counts)
        return vecs
    except Exception as e:                       # noqa: BLE001 —— 审计要记失败原因，原样抛出
        err = f"{type(e).__name__}: {e}"
        raise
    finally:
        _get_buckets()[p]["embed_seconds"] += time.time() - t0
        _call_log_out(pt=ntok, seconds=time.time() - t0, error=err, kind="embed",
                      req={"n": len(sent), "model": EMBED_MODEL_SERVED,
                           "timeout": timeout, "doc_prefix": bool(EMBED_DOC_PREFIX)},
                      inputs=sent)
