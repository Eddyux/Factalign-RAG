"""FactAlign-RAG 全局配置。服务器 5090 + vLLM。"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SAMPLES_DIR = DATA_DIR / "samples"
PROMPTS_DIR = ROOT / "prompts"
RESULTS_DIR = ROOT / "results"
BASELINES_DIR = ROOT / "baselines"
CACHE_DIR = DATA_DIR / "cache"
MERGED_DIR = DATA_DIR / "merged"
CORPUS_DIR = DATA_DIR / "corpus"

# ============================================================
# LLM backend: "vllm" | "ollama"
# 服务器 5090 用 vLLM 服务 chat + embedding（embedding 用 --task embed 单独起实例）
# ★ 必须读环境变量：`configs/*.json` 的 `"backend"` 一直在提供这个键，
#   而这里以前写死 `BACKEND = "vllm"` → 改配置文件**毫无作用**（属于"看着能调、
#   其实代码没接"的死键；`tools/audit_params.py --dead` 会把它揪出来）。
# ============================================================
BACKEND = os.environ.get("BACKEND", "vllm")

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")    # chat
# CHAT_MODEL 是 vLLM 服务的「模型名」（serve 时用 --served-model-name 指定），不是磁盘路径
CHAT_MODEL = os.environ.get("CHAT_MODEL", "Llama-3.1-8B-Instruct")
# embedding 也走 vLLM（--task embed），可与 chat 同实例不同端口，如:
#   vllm serve Qwen/Qwen3-Embedding-0.6B --task embed --port 8001 \
#       --served-model-name Qwen3-Embedding-0.6B --max-num-batched-tokens 32768
VLLM_EMBED_BASE_URL = os.environ.get("VLLM_EMBED_BASE_URL", "http://127.0.0.1:8001/v1")
# ★ 默认 = **服务器上的本地目录**：`os.path.isdir()` 为真 → `local_files_only=True`，
#   离线直读、零联网、可复现（`EMBED_MODEL` 同时是"切块尺子"的 tokenizer 来源，见 common/llm.py）。
#   指 HF 名（`Qwen/Qwen3-Embedding-0.6B`）会走 hf-mirror 联网一次再缓存 —— 只在有外网时可行。
EMBED_MODEL = os.environ.get("EMBED_MODEL", "/root/rivermind-data/models/Qwen3-Embedding-0.6B")          # 索引指纹用（本地目录）
EMBED_MODEL_SERVED = os.environ.get("EMBED_MODEL_SERVED", "Qwen3-Embedding-0.6B")  # vLLM 的 served-model-name
# bge 系列非对称指令：query 侧加、passage 侧不加（common/llm.py 的 embed/embed_batch 已处理）
# 现已改用 Qwen3 格式（见下面 EMBED_QUERY_FORMAT=qwen3），本行只在切回 bge 时生效。
EMBED_QUERY_INSTRUCTION = os.environ.get(
    "EMBED_QUERY_INSTRUCTION", "Represent this sentence for searching relevant passages: ")

# ollama（本地调试备用，需同时把 BACKEND 改为 "ollama"）
OLLAMA_BASE_URL = "http://127.0.0.1:11434"

# ============================================================
# Evaluation
# ============================================================
SAMPLE_SIZE = 300
RANDOM_SEED = 42
# ★ `SEARCH_TOP_K`(20) / `SEARCH_TOP_K_CHUNKS`(5) 这两个旋钮**已删除**（2025 审查）。
#   原因：名字像"喂给模型的条数"，实际生产路径**没有任何消费方** ——
#     · `SEARCH_TOP_K`   只被 search.py 的函数默认参数（factalign 不走 search_graph）、
#       独立实验工具 test_layer.py 读到；它跟 factalign 浅层召回的 `recall_chunks`(=20)
#       只是数字撞了，不是同一个东西。
#     · `SEARCH_TOP_K_CHUNKS` 只剩 factalign 的消融分支（`--decompose-recall chunk`）。
#   真正决定"发几条证据"的是 `FACTALIGN_EVIDENCE_PROPS`/`_WINDOWS` 与 `EVIDENCE_WINDOWS`。
#   留着只会让人把日志里的 `topk=20` 误读成本次 run 的生成预算（真踩过这个坑）。
# 切块按 token（尺子 = `CHUNK_TOKENIZER`：默认 embedding 模型的 tokenizer）：
# ★ 默认**不切块**（一个 context = 一个整段块）：配置里写 `0`，读进来归一成 `CHUNK_NO_SPLIT`=2000、
#   overlap 0（原因见下方 ★★）。旧口径为 400 token/块、40 重叠（bge 上限 512，400 留余量）。
# ⚠️ 这两条**必须走环境变量**：`configs/*.json` 正是通过注入 `EXTRACT_CHUNK_*` 来控制
# 分块口径的（官方对齐要 1200/100）。写死的话配置文件里写了也不生效——曾是真 bug。
# 改动 = 语料单元口径变更 → 必须开新 --graph-version 重建。
#
# ★★ 口径变更：默认**不切块**（配置里写 `0` = 一个 context = 一个整段块）。
#    旧的 400/40 口径设 `EXTRACT_CHUNK_TOKENS=400 EXTRACT_CHUNK_OVERLAP=40` 即可回退。
#    实测：2wiki/hotpotqa/musique/popqa 的原始 passage 有 98% 本来就短于 400 token
#    （根本没被切过），所以这个变更对它们近似 no-op；**但 triviaqa 的 passage 是整篇
#    维基文章**（中位 3635 词 ≈ 4900 token）→ 会被 embed 截断，需单独处理。
#
# ⚠️⚠️ **"不切块"在代码里不能是 0，统一归一成 `CHUNK_NO_SPLIT`（= 2000）。**
#    配置/CLI 上写 `0`（可读），**读进来立刻归一成 2000** 再往下传。
#    为什么不能是 0：`0` 是个脚枪——所有对块尺寸做算术的下游都会炸，已实测三处：
#      · HGR 官方 `operate.chunking_by_token_size`：`range(0, N, max_token_size - overlap)`，
#        0-0 = 步长 0 → `ValueError: range() arg 3 must not be zero`（建索引直接崩）
#      · LightRAG `_window_step()`：`if overlap >= size: raise ValueError`（0>=0 成立）
#      · 我们自己的 `measure_rechunk_risk()`：`if n > HGR_CHUNK_TOKENS` 恒真 → 误报"复核异常"
#    归一成 2000 后，**所有下游一行都不用改**（含官方库）：
#      · 块 ≤ 2000 token → 原样 1 个块（项目关心的 passage 几乎都在这以内）
#      · 块 > 2000 token → 按 2000 切（triviaqa 那种整篇维基文章走这条，相当于安全阀，
#        也顺带说明"块上限 = 2000"这个口径本身）
#    官方 HGR 只在这一个调用点用该字段、LightRAG 的 chunker 无上界（均已核对）→ 安全。
CHUNK_NO_SPLIT = 2000


def normalize_chunk_tokens(v: int) -> int:
    """块尺寸归一：`<= 0`（不切块）→ `CHUNK_NO_SPLIT`（2000），否则原值。

    所有 chunk 尺寸旋钮（`EXTRACT_CHUNK_TOKENS` / `HGR_CHUNK_TOKENS` /
    `LIGHTRAG_CHUNK_TOKENS`）都必须过这一道，否则 0 会漏到下游炸掉。见上面说明。
    """
    return CHUNK_NO_SPLIT if v <= 0 else v


EXTRACT_CHUNK_TOKENS = int(os.environ.get("EXTRACT_CHUNK_TOKENS", "0"))
EXTRACT_CHUNK_OVERLAP = int(os.environ.get("EXTRACT_CHUNK_OVERLAP", "0"))
# 归一：0 → `CHUNK_NO_SPLIT`(2000)。**下游只应看到归一后的值。**
EXTRACT_CHUNK_TOKENS = normalize_chunk_tokens(EXTRACT_CHUNK_TOKENS)
# 建图管线版本：合并/关联逻辑或窗口参数变更后 bump，旧缓存（per-question 与语料图）自动失效
EXTRACT_VERSION = "v7"
EMBED_BATCH_SIZE = 64
# 单个 embedding HTTP 请求的 token 预算。vLLM 对 pooling/embedding 模型的请求：
# token 总量超过引擎的 --max-num-batched-tokens 时**既不报错也不执行**，永久静默
# 挂住（vllm-project/vllm#41969）。只按条数（64 条）切挡不住长文本——chunk 是
# 480 token/条，64 条 ≈ 3 万 token/请求，直接装不下。4096 在引擎默认档和显式
# --max-num-batched-tokens 32768 下都安全。
EMBED_REQUEST_TOKENS = int(os.environ.get("EMBED_REQUEST_TOKENS", "4096"))
# 服务端 embedding 实例的 `--max-num-batched-tokens`（**声明值，vLLM API 不上报**）。
# 填了 → `server_probe` 会拿它和 `EMBED_REQUEST_TOKENS` 精确比对并在超了时报警；
# 留 0（默认）→ 只在日志里提示"无法探测、请自行核对启动参数"，不误报。
# 例：起服务时写了 `--max-num-batched-tokens 8192`，这里就填 8192。
# ⚠️ 注意别和 `--max-model-len` 混：前者管**一批**的 token 总量（→ EMBED_REQUEST_TOKENS），
#    后者管**单条**序列上限（→ EMBED_MAX_TOKENS）。两者常常差很多（如 8192 vs 2048）。
EMBED_SERVER_BATCH_TOKENS = int(os.environ.get("EMBED_SERVER_BATCH_TOKENS", "0"))

# ============================================================
# Embedding 模型（尺子 + 向量空间）——**换 embedder 只改这一节**
# ============================================================
# 为什么集中在这里：embedding 模型决定了三件事，散着改一定会漏一处：
#   ① 向量空间（维度必须与 vLLM 实际返回一致，不一致时向量全错但**不报错**）；
#   ② 截断尺子（chunk 切块、fit_embed、HGR/LightRAG 内部计数都按它的 tokenizer）；
#   ③ 查询侧前缀（bge 是英文指令句，Qwen3-Embedding 是 `Instruct: …\nQuery: …`）。
# 换模型 = 换 `EMBED_MODEL` / `EMBED_MODEL_SERVED` / `EMBED_DIM` / `EMBED_MAX_TOKENS`
# / `EMBED_TOKENIZER` / `EMBED_QUERY_FORMAT` **一起**，并**重建所有向量**
# （开新 `--graph-version`）：旧向量和新向量不在同一空间，混用只会静默给出垃圾结果。
# 现成配方见 `configs/embedding-qwen3-8b.json`。

# 向量维度：必须与 vLLM 该实例真实输出一致（Qwen3-Embedding-0.6B=1024 原生；
# Qwen3-Embedding-8B=4096 或 MRL 截断后的 1024/2048/…；旧 bge-large-en-v1.5=1024）。
# `common.llm._embed_dim_check()` 首次嵌入时探一次服务端真实维度，不一致会**告警**
# （不 raise：告警足够定位，硬崩会把一次几百题的批整个打断）。
EMBED_DIM = int(os.environ.get("EMBED_DIM", "1024"))
# 单条文本喂给 embedding 前的截断上限。Qwen3-Embedding 上下文 32768，不必再像 bge 那样
# 卡在 480；不切块后本项目实际最长的单条是**整段 passage**（triviaqa 的整篇维基文章会被本值
# 截断到 4096），2048 只是常规 passage 的余量。
# ★ 必须 ≤ `EMBED_REQUEST_TOKENS`（否则单条超预算会被拆成一条一组，能跑但很慢），
#   且 `EMBED_REQUEST_TOKENS` 必须 ≤ vLLM 的 `--max-num-batched-tokens`（超了会**静默挂住**）。
#   要把长实体描述也完整嵌入 → 两个一起提到 8192 / 16384，并把引擎的
#   `--max-num-batched-tokens` 提到 ≥16384（默认 4096 不需要改任何服务端参数）。
EMBED_MAX_TOKENS = int(os.environ.get("EMBED_MAX_TOKENS", "4096"))
# **embedding 侧**用哪把 tokenizer 尺子：`embed`（默认，= `EMBED_MODEL` 自己的 HF
# tokenizer，当前模型是 Qwen3-Embedding-0.6B，即它的 tokenizer）或 `tiktoken`。
# embedder 的单条上限必须按它自己的 tokenizer 数，
# 所以这根轴只服务于 `fit_embed` / `embed`（历史默认就是 embedder 自己的 tokenizer）。
EMBED_TOKENIZER = os.environ.get("EMBED_TOKENIZER", "embed").strip().lower()
# **切块/内部计数**用哪把尺子：`embed`（默认，= 与 `EMBED_TOKENIZER` 同一个
# tokenizer）或 `tiktoken`（`EMBED_TIKTOKEN_MODEL`，官方 HGR/LightRAG 原版口径）。
# 两根轴分开的原因：官方那些库是**按 tiktoken 调参**的（4000 截断、1200 分块…），
# 想跟官方数字对账就得把内部计数切回 tiktoken，而这不该连 embedder 的截断一起改。
# 注意：改它 = 改语料口径 → 旧图/旧向量作废，**必须开新 --graph-version 重建**。
CHUNK_TOKENIZER = os.environ.get("CHUNK_TOKENIZER", EMBED_TOKENIZER).strip().lower()
EMBED_TIKTOKEN_MODEL = os.environ.get("EMBED_TIKTOKEN_MODEL", "gpt-4o-mini")
# 首次嵌入探测服务端真实维度（1=探，0=不探，省一次 HTTP）。
EMBED_DIM_CHECK = os.environ.get("EMBED_DIM_CHECK", "1") == "1"


# ── 按数据集覆盖语料单元（triviaqa 专属）────────────────────────────────────────
# 理由：**triviaqa 的 passage 是整篇维基文章**（实测中位 3483 词、均值 4645、最长 22305 词，
# 569/840 行超过 2000 token），而其余 4 个数据集的 passage 中位只有 46–103 词、一条就一块。
# 用同一把尺子会得到两种完全不同的检索单元（triviaqa 一块 = 一篇长文，其他一块 = 一个短段），
# 跨数据集比"块召回"没有意义；给 triviaqa 显式切成 1200/100 后，它才是"正常大小的块"。
# ★ 只覆盖**被显式配置**的数据集；其余一律走全局 `EXTRACT_CHUNK_TOKENS/OVERLAP`（默认 0/0
#   → 归一成 2000/0 = 不切块）→ **其他数据集的行为一个字节都不变**。
# ★ 改动 = 语料单元变更 → 该数据集的旧图/旧向量作废，**必须开新 --graph-version 重建**。
#   （建图缓存 key 里带了切块参数，见 `extract.py:_cache_key`，所以不会静默复用旧图。）
# 优先级：`{DATASET}_CHUNK_TOKENS` / `{DATASET}_CHUNK_OVERLAP` 环境变量 > 本默认表 > 全局值。
#   环境变量由 `configs/*.json` 注入（键：`{dataset}_chunk_tokens`，见 configs/README.md）。
#   ⚠️ 读取**必须走下面的 `DATASET_CHUNK_ENV` 映射**，不要在这里写死
#      `os.environ.get("TRIVIAQA_CHUNK_TOKENS", "1200")` —— 那样 `configs/*.json` 就改不动它了
#      （这正是历史上真出现过的 bug：`config.py` 写死 400，配置文件一直提供 400，改了没用）。
#      键名映射放在数据里而不是 f-string 拼，也让
#      `smoke_new_baselines.check_config_keys_are_read`（扫源码里的环境变量字面量）能看见。
DATASET_CHUNK_DEFAULTS: dict[str, tuple[int, int]] = {
    "triviaqa": (1200, 100),
}
# 每行 = (数据集名, 读 token 的环境变量名, 读 overlap 的环境变量名)。
# 加新数据集：这里加一行 + 上面 `DATASET_CHUNK_DEFAULTS` 加一条（不给默认值就不必加）。
DATASET_CHUNK_ENV: tuple[tuple[str, str, str], ...] = (
    ("triviaqa", "TRIVIAQA_CHUNK_TOKENS", "TRIVIAQA_CHUNK_OVERLAP"),
)


def dataset_chunking(dataset: str) -> tuple[int, int]:
    """返回该数据集的 `(chunk_tokens, overlap)` —— **切块前必须走这一道**。

    显式传 `chunk_tokens` 给 `split_by_tokens` 的地方，都要用本函数的结果，否则 triviaqa
    会静默退回全局的"不切块"口径（= 2000/0 的安全阀），与已建好的图对不上。

    `chunk_tokens` 已经过 `normalize_chunk_tokens()`：`0` 不在这里出现，下游拿到的一定 > 0。
    """
    ds = (dataset or "").strip().lower()
    env_t = env_o = None
    for name, kt, ko in DATASET_CHUNK_ENV:
        if name == ds:
            env_t, env_o = os.environ.get(kt), os.environ.get(ko)
            break
    dt, do = DATASET_CHUNK_DEFAULTS.get(ds, (None, None))  # type: ignore[assignment]
    if env_t is not None and str(env_t).strip() != "":
        t = int(env_t)
    elif dt is not None:
        t = int(dt)
    else:
        return EXTRACT_CHUNK_TOKENS, EXTRACT_CHUNK_OVERLAP
    if env_o is not None and str(env_o).strip() != "":
        o = int(env_o)
    elif do is not None:
        o = int(do)
    else:
        o = EXTRACT_CHUNK_OVERLAP
    t = normalize_chunk_tokens(t)
    # overlap 必须 < 块尺寸，否则 `step = max - overlap <= 0` → 滑窗死循环/起点不前进。
    o = max(0, min(int(o), t - 1))
    return t, o
# 查询侧前缀怎么拼：
#   bge   —— `EMBED_QUERY_INSTRUCTION + text`（bge 系列的英文指令句）
#   qwen3 —— `Instruct: {EMBED_QUERY_TASK}\nQuery: {text}`（Qwen3-Embedding 官方格式）
#   none  —— 不加任何前缀
# `EMBED_QUERY_TEMPLATE` 非空时优先用它（支持 {task} / {text} 两个占位符）。
# ★ 默认 `qwen3`（配合上面的全局 Qwen3-Embedding-0.6B）；切回 bge 时设 `EMBED_QUERY_FORMAT=bge`。
EMBED_QUERY_FORMAT = os.environ.get("EMBED_QUERY_FORMAT", "qwen3").strip().lower()
EMBED_QUERY_TASK = os.environ.get(
    "EMBED_QUERY_TASK",
    "Given a web search query, retrieve relevant passages that answer the query")
EMBED_QUERY_TEMPLATE = os.environ.get("EMBED_QUERY_TEMPLATE", "")
# 文档侧前缀：bge/Qwen 都不加（留口子给需要加的模型，如 e5 的 "passage: "）。
EMBED_DOC_PREFIX = os.environ.get("EMBED_DOC_PREFIX", "")
# 首次嵌入时打印一次 resolved 配置（0=安静）。
EMBED_VERBOSE = os.environ.get("EMBED_VERBOSE", "1") == "1"

# 建图（窗口抽取）并发：extract.py 的 ThreadPoolExecutor 用它。
# 注意：这条路径**没过自适应并发闸**（闸只在 methods/*_adapter 的 chat_retry 里），
# 并发给太高时 vLLM 的 KV 会被打满 → 排队 → 请求超时（8B/Llama 每 token KV 128KB）。
# 想临时调小：export EXTRACT_WORKERS=16
EXTRACT_WORKERS = int(os.environ.get("EXTRACT_WORKERS", "64"))

# ★ 以下 6 个常量**已删除**（2025 审查）：`DEFAULT_WEIGHTS`(4 维权重) /
#   `ANSWER_TYPE_WEIGHT` / `ENABLE_HYBRID` / `HYBRID_BM25_WEIGHT` /
#   `CONFIDENCE_THRESHOLD` / `MAX_ROUNDS` / `ENABLE_MULTI_ROUND`。
#   它们只喂 **旧检索设计**（`planner` 打 4 维分 + BM25 混合 + 置信度多轮），
#   而那条链的消费方 `search.search_graph()` / `search_graph_ppr()` / `plan()` / `replan()`
#   在生产里**没有任何调用点**（生产只用 `search.build_faiss_index`，它不依赖这些常量）。
#   验证方式：`python tools/audit_params.py --unconfigured` 看不到它们（它们连
#   `os.environ` 都不读、写死在代码里，所以配置文件夹也改不了）；调用图见 git 历史。
#   ⚠️ `common/trace.py` 里另有一个**同名但无关**的 `MAX_ROUNDS=64`（trace 每题的轮数上限）。

for d in (DATA_DIR, SAMPLES_DIR, RESULTS_DIR, CACHE_DIR, MERGED_DIR, CORPUS_DIR):
    d.mkdir(parents=True, exist_ok=True)


def make_tag(method: str, dataset: str, sample_size: int) -> str:
    """结果文件名 tag：{时间}_{method}_{dataset}_n{N}_{模型名}。

    以前 tag 里带 `_k{SEARCH_TOP_K}` —— 那个旋钮已删除（生产路径无人消费，见上面注释），
    留一个"看起来像这次检索深度、其实无关"的数字在文件名里只会误导。
    """
    from datetime import datetime
    date = datetime.now().strftime('%Y%m%d_%H%M%S')
    model = Path(CHAT_MODEL).name.replace("/", "_").replace(":", "_")
    return f"{date}_{method}_{dataset}_n{sample_size}_{model}"
