"""PropRAG 适配器（EMNLP 2025）—— 命题图 + 两阶段 PPR + 命题路径 beam search。

论文: "PropRAG: Guiding Retrieval with Beam Search over Proposition Paths" (EMNLP 2025)
官方代码: https://github.com/OSU-NLP-Group/PropRAG  →  `baselines/PropRAG-main/`
（官方是 HippoRAG 的改编版，`src/proprag/` 下 ~2800 行；行号见下方引用。）

═══ 官方检索链（逐函数核对）═══
  **图**（`index()` PropRAG.py:162-268）——注意：**命题不是图顶点**，只有 entity/chunk 两类顶点：
    · E1 命题内实体两两全连接（:944-995）：权重 = 共现**整数计数**（同一对实体在多个命题里
      出现就累加），两个方向各写一遍 → 无向图里是**两条平行边**（权重近似翻倍，官方如此，不修）。
    · E2 passage→entity（:569-611）：权重恒 **1.0**。
    · E3 同义实体（:613-674）：实体向量 KNN（topk=2047、阈值 0.8、实体需 >2 个字母数字、
      每点最多 101 个邻居），权重 = **cosine 相似度**（float）。
    · 顶点 id = `"entity-"+文本` / `"chunk-"+文本`（`compute_mdhash_id` = **字符串拼接**，非哈希）。
   **检索**（`graph_search_with_proposition_entities` :1478-1726）：
    1. `dense_passage_retrieval(query, no_sort=True)`（:1129-1177）：query 与**全部** passage
       向量点积（向量 L2 归一化 ⇒ 余弦）→ **全语料 min-max 归一化**（不截 top-k）。
    2. **第一阶段 linking = beam search**（:1516-1529）：beam_width=200、max_path_length=1、
       "average" ⇒ 只有**单命题路径**；取前 20 条路径 → 前 40 个实体，`phrase_weights[顶点]=1.0`
       （**二值**，不是分数），再 min-max 归一化。
    3. 重启向量（:1531-1540）= phrase_weights + `dpr_score * 0.05`（passage_node_weight）。
    4. **第一阶段 PPR，damping=0.75（硬编码）**（:1552）。
    5. **聚焦子图**（:1557-1598）= 第一阶段 top-50 passage 顶点 ∪ 它们的直接 entity 邻居。
    6. **第二阶段 beam search**（:1611-1633）：beam_width=4、max_path_length=3、
       second_stage_filter_k=40，`prop_set` = 这 50 篇 passage 的全部命题；
       focused = 前 5 条路径的实体，initial = top-3 单命题路径的实体。
    7. 子图重启向量（:1635-1682）：focused/initial 各自 `max(path_scores)` → 各自 min-max 归一化
       → 逐元素 max；子图内 passage 再加 `dpr * 0.05`。
    8. **子图 PPR，damping=0.45（硬编码）**（:1689-1696）→ 合并（:1697-1714）：全局分数先整体乘
       `min(subgraph_ppr) * 0.5`，子图 passage 分数**覆盖**之。
   **作答**（`qa()` :463-567）：`qa_top_k=5` 篇 passage + 官方 `rag_qa_musique` 模板
    （system + 1-shot + `${prompt_user}`；user 里每篇前缀 `Wikipedia Title: {title}\\ntext`）
    → `response.split('Answer:', 1)[1].strip()`。
   官方 config：passage_node_weight=0.05、retrieval_top_k=200、qa_top_k=5、
    synonymy_edge_sim_threshold=0.8、beam_width=4、max_path_length=3、second_stage_filter_k=40。
    **死参数**（传了不读，本文件也不实现）：linking_top_k / use_beam_search / damping /
    max_qa_steps / graph_type。
  **beam search**（`graph_beam_search.py`；本文件按行重写，官方那版顶层 `import torch`）：
    · 种子 = `np.dot(命题向量, query)` 取 top-200（`np.argpartition` + 降序）→ 单命题路径，
      稳定排序（官方 `list.sort` 稳定 ⇒ 并列按插入序）。
    · 扩展（depth=2..max_path_length）：沿「共享实体 / 同义实体」连接命题；路径内去重 +
      `frozenset(命题集)` 全局去重；另把 **top-3 初始命题**当「并行推理」候选接上。
    · 路径打分（`batch_score_paths` :387-582）= `cos(query, 归一化(路径命题向量平均))`；
      `"concatenate"` = 命题文本拼起来重新编码；`second_stage_filter_k>0` 时对前 40 条用
      concatenate 重打分并**覆盖**原 score。
    · `get_entities_from_paths`（:924-961）：路径中每个命题的实体、以及同义连接的**目标实体**
      （`entity2`）各累加一次该路径分数，再按 `sum(scores)` 排序（并列按首次出现顺序）。

═══ 本项目怎么接（论文里必须写明的适配取舍）═══
  - **语料单元 = 整段 passage（不切块）**（`title\ntext` 原样，与 hipporag 适配器同口径），
    不是官方「一篇文章一个 chunk」。切块更细 ⇒ 命题更碎、图更大；这是「所有方法同一语料
    单元」的硬要求，必须写进论文。
  - **命题抽取用本项目 vLLM + 官方 prompt 逐字**（`ner_expanded` + `proposition_extraction`，
    每块 **2 次** LLM 调用，先 NER 再命题，与官方 `EnhancedOpenIE` 同序）。官方论文用
    Llama-3.3-70B 抽命题，这里用本项目 chat 模型 —— **prompt-level 复现**。
  - **embedding 换成项目 bge-large-en-v1.5**（官方 NV-Embed-v2）：query 侧沿用官方
    `query_to_passage` 指令原文（`prompts/linking.py`），文档侧不加指令，均 L2 归一化。
    换 embedding 必然改变同义边与路径分数 —— 论文写明。
  - **官方 `outputs/<ds>/openie_results_*.json` 不可复用**：那是**他们自己的语料**
    （本地 500 题样本与 `hotpotqa_corpus.json` 标题交集仅 ~13%、问题交集仅 ~10%），
    不是同一批文章 ⇒ 命题必须在本地重跑（建图成本按这个报）。
  - **beam search 重写而非 import 官方 `graph_beam_search.py`**（该文件顶层 `import torch`，
    评估机不一定有）；参数、公式、排序稳定性逐条对齐。`igraph` 只用于 PPR
    （`requirements.txt` 已有 `python-igraph`）。
  - **PPR = igraph `personalized_pagerank(directed=False, weights="weight",
    implementation="prpack")`**，damping 0.75 / 0.45 与官方一致；**不传 niter/eps**（prpack 直接解）。
  - **作答走统一 `generate_answer`**（与其余 baseline 同生成口径）。★ 官方
    `rag_qa_musique` 模板那条对照路径**已删除**：全项目的最终作答只有两条生成器
    （`prompts/rag_prompt.txt` / `rag_prompt_triviaqa.txt`）。
  - 检索 **LLM-free**（在飞调用为 0，只有本地矩阵运算 + PPR）；每题 LLM 调用 = 作答 1 次。
    建图期成本 = `2 × passage 数`（+失败重试），落盘后重跑**不重建**。

用法:
  python run.py --method proprag --dataset hotpotqa --sample 300 --workers 8
  旋钮（环境变量，无需改代码；默认值全部 = 官方值）:
    PROPRAG_TOP_DOCS=5            喂作答的 passage 数（= 官方 qa_top_k=5）
    PROPRAG_PASSAGE_WEIGHT=0.05   passage_node_weight
    PROPRAG_SYN_THRESHOLD=0.8     synonymy_edge_sim_threshold
    PROPRAG_BEAM_WIDTH=4          PROPRAG_MAX_PATH_LENGTH=3   PROPRAG_SECOND_STAGE_K=40
    PROPRAG_SIM_THRESHOLD=0.75    beam 里判定同义边的阈值
    PROPRAG_FOCUS_TOP_K=50        聚焦子图取第一阶段 top 多少篇
    PROPRAG_EXTRACT_WORKERS=64    建图期命题抽取并发（本项目统一 64）
    其余官方值也都可调（默认 = 官方）: PROPRAG_PPR_DAMPING_1=0.75
      PROPRAG_PPR_DAMPING_2=0.45  PROPRAG_STAGE1_PATH_TOP=20
      PROPRAG_STAGE1_ENTITY_TOP=40  PROPRAG_BEAM_SEED_TOP=200
      PROPRAG_KEEP_INITIAL_PROPS=3  PROPRAG_FOCUS_PATH_TOP=5
"""

import json
import os
import re
import threading

import numpy as np

from methods.base import BaseMethod, generate_answer
from methods.standard_rag import build_chunks
from methods.scaffold import SourceNotReady, require_repo
from common.llm import (chat_messages, embed_batch, _embed_request, _normalize_rows, phase,
                        fit_embed, role)
from config import BASELINES_DIR, EXTRACT_CHUNK_TOKENS, EXTRACT_CHUNK_OVERLAP

# ══════════════════════════════════════════════════════════════════════════════
# ★ 官方口径参数（默认 = 官方，勿随意改）。行号已逐条对官方源码核过：
#   `PropRAG.py` / `graph_beam_search.py` / `utils/config_utils.py`（同仓）
# ⚠️ 一个坑：官方 `config_utils.py:181` 的 `damping = 0.5` 是**死配置** ——
#   `retrieve()` 的生产路径在 `PropRAG.py:1552` 硬编码 `damping=0.75` 覆盖了它，
#   第二阶段在 `:1691` 硬编码 `0.45`。只看 config 会以为该用 0.5，那是错的。
# ══════════════════════════════════════════════════════════════════════════════
PROPRAG_PASSAGE_WEIGHT = float(os.environ.get("PROPRAG_PASSAGE_WEIGHT", "0.05"))   # config_utils.py:68 默认 / PropRAG.py:1536 使用处
PROPRAG_SYN_THRESHOLD = float(os.environ.get("PROPRAG_SYN_THRESHOLD", "0.8"))      # config_utils.py:149 / PropRAG.py:646
PROPRAG_SYN_TOPK = int(os.environ.get("PROPRAG_SYN_TOPK", "2047"))                 # config_utils.py:137 / PropRAG.py:643
PROPRAG_SYN_MAX_NN = int(os.environ.get("PROPRAG_SYN_MAX_NN", "101"))              # PropRAG.py:661 (`num_nns > 100`)
PROPRAG_BEAM_WIDTH = int(os.environ.get("PROPRAG_BEAM_WIDTH", "4"))                # config_utils.py:159 / PropRAG.py:1611
PROPRAG_MAX_PATH_LENGTH = int(os.environ.get("PROPRAG_MAX_PATH_LENGTH", "3"))      # config_utils.py:163 / PropRAG.py:1612
PROPRAG_SECOND_STAGE_K = int(os.environ.get("PROPRAG_SECOND_STAGE_K", "40"))       # config_utils.py:167 / PropRAG.py:1613
PROPRAG_SIM_THRESHOLD = float(os.environ.get("PROPRAG_SIM_THRESHOLD", "0.75"))     # PropRAG.py:278 (retrieve 的 sim_threshold 形参)
PROPRAG_FOCUS_TOP_K = int(os.environ.get("PROPRAG_FOCUS_TOP_K", "50"))             # PropRAG.py:335 调用处硬编码 / :1482 签名默认
PROPRAG_PPR_DAMPING_1 = float(os.environ.get("PROPRAG_PPR_DAMPING_1", "0.75"))   # PropRAG.py:1552 硬编码（覆盖 config 的 0.5）
PROPRAG_PPR_DAMPING_2 = float(os.environ.get("PROPRAG_PPR_DAMPING_2", "0.45"))   # PropRAG.py:1691 硬编码
PROPRAG_STAGE1_PATH_TOP = int(os.environ.get("PROPRAG_STAGE1_PATH_TOP", "20"))   # PropRAG.py:1517 paths[:20]
PROPRAG_STAGE1_ENTITY_TOP = int(os.environ.get("PROPRAG_STAGE1_ENTITY_TOP", "40"))  # PropRAG.py:1519 [:40]
PROPRAG_BEAM_SEED_TOP = int(os.environ.get("PROPRAG_BEAM_SEED_TOP", "200"))      # PropRAG.py:1047 (beam_width=200, max_path_length=1)
PROPRAG_KEEP_INITIAL_PROPS = int(os.environ.get("PROPRAG_KEEP_INITIAL_PROPS", "3"))  # graph_beam_search.py:650 (keep_initial_props = 3)
# 聚焦子图重启向量取的路径/实体数。官方 PropRAG.py:1623/:1631/:1633 三处硬编码 `[:5]`
PROPRAG_FOCUS_PATH_TOP = int(os.environ.get("PROPRAG_FOCUS_PATH_TOP", "5"))

# ★ 本项目口径（与官方不同处一律列这里）
PROPRAG_TOP_DOCS = int(os.environ.get("PROPRAG_TOP_DOCS", "5"))
PROPRAG_LLM_TIMEOUT = int(os.environ.get("PROPRAG_LLM_TIMEOUT", "300"))
# 建图期命题抽取并发：本项目口径 64，但**并非全方法一致**（原文写"本项目统一 64 …
# 与 hypergraphrag HGR_MAX_ASYNC 同口径"，是错的 —— HGR 实为 32）。实际：
#   factalign `EXTRACT_WORKERS`=64（extract.py:175）、hipporag `HIPPORAG_MAX_ASYNC`=64、
#   本项 =64 ｜ hypergraphrag `HGR_MAX_ASYNC`=32
# 限流机制还分三种：本项与 factalign 是**裸线程池**（只限线程数，没有在途闸）；
# hipporag `_SLOTS`(:207) 另加信号量；只有 hypergraphrag
# 接了 `gate_for()` 的 AIMD 自适应闸（全仓唯一调用点 hypergraphrag_adapter.py:137；
# 其余方法走 `chat_messages`，只计数不限流，见 common/llm.py:475/1527）。
PROPRAG_EXTRACT_WORKERS = int(os.environ.get("PROPRAG_EXTRACT_WORKERS", "64"))
# ★ 生成上限 = **8192，逐字对齐官方**。`max_tokens` 是「单次响应最多生成多少 token」，
#   与输入（prompt）上限无关；它是**上限不是配额**（模型自己吐到 EOS 就停）。
#   官方出处（本次已从调用链往上核过，不再是抄某个文件）：
#     · `utils/config_utils.py:30-33`  `max_new_tokens` 字段 **默认 8192**
#     · `main.py:146`  `# max_new_tokens=None` **被注释掉** → 用上面的默认
#     · `llm/openai_gpt.py:255`  `kwargs.get("max_new_tokens", 8192)`
#     · `:278-280`  模型名不含 "gpt"（Llama）→ `max_completion_tokens` 改名
#       `max_tokens` 发给 OpenAI 兼容端点（就是我们这种 vLLM 端点）
#     · `information_extraction/enhanced_openie.py:86/:118/:225` 三处 `infer()` **都不传
#       该参数** → 全程 8192
#   ⚠️ `openie_vllm_offline.py:38/:45` 的 512/2048 **不是这条路径的值** —— 那是
#      HippoRAG 2 血统的姊妹实现，全仓只有类定义 + 一行 import（`PropRAG.py:24`），
#      **从未被实例化**。曾误引它把这里改成 512/2048，已回退，别再犯。
#   为什么要跟到 8192：官方**预期会撞上限**并写了补救（`enhanced_openie.py:95-98`
#      告警、`:230-231` 调 `fix_broken_generated_json`）—— 上限给紧会截断输出，
#      截断的 JSON 解析失败会触发 `PROPRAG_RETRY` 重试，成倍放大成本。
#      （实测本项目 `ct` 中位数仅 144、贴顶 1.2%，所以这个值平时基本不生效。）
PROPRAG_NER_MAX_NEW_TOKENS = int(os.environ.get("PROPRAG_NER_MAX_NEW_TOKENS", "8192"))
PROPRAG_PROP_MAX_NEW_TOKENS = int(os.environ.get("PROPRAG_PROP_MAX_NEW_TOKENS", "8192"))
# 旧旋钮兜底：跑过老配置的环境可能还设着 `PROPRAG_MAX_NEW_TOKENS`（单一上限）。
# 尊重它（否则同样的环境变量会静默换口径），但只作为兜底、新代码不再写它。
_LEGACY_MAX_NEW = os.environ.get("PROPRAG_MAX_NEW_TOKENS", "").strip()
if _LEGACY_MAX_NEW:
    PROPRAG_NER_MAX_NEW_TOKENS = PROPRAG_PROP_MAX_NEW_TOKENS = int(_LEGACY_MAX_NEW)
PROPRAG_RETRY = int(os.environ.get("PROPRAG_RETRY", "2"))   # 重试次数 → 共 3 次尝试
# ★ 这是**项目自造的旋钮，不是官方参数**（官方没有对应字段）：
#   官方 PropRAG 的重试是两层，且**都不封顶**——
#   ① 条目层：`RetryExecutor.process()` 是 `while self.futures:` + 把失败项重新 submit
#      （`baselines/PropRAG-main/src/proprag/utils/misc_utils.py:130-157`）→ **无上限、
#      永不丢弃失败块**。
#   ② 传输层：`llm_utils.py:112` 用 tenacity `stop_after_attempt(
#      experiment_config.async_max_retry_attempts)` —— 但 `async_max_retry_attempts`
#      **在整个官方仓库里没有任何定义**（`config_utils.py` 只有 `max_retry_attempts=5`，
#      且无任何 `async_*` 字段）→ 这条路径一跑就 AttributeError，八成是字段改名后漏同步。
#   所以「每个 passage 的每一步最多 3 次、失败丢弃」是**我们主动的成本控制偏离**
#   （相对官方的"无上限重试"），论文里要单独声明。
PROPRAG_KNN_BLOCK = int(os.environ.get("PROPRAG_KNN_BLOCK", "2000"))

PROPRAG_REPO_CANDIDATES = [BASELINES_DIR / "PropRAG", BASELINES_DIR / "PropRAG-main"]
PROPRAG_ARTIFACT_DIRNAME = "proprag_graph"      # data/{method}/{dataset}/v{n}/proprag_graph/
PROPRAG_ARTIFACT_VERSION = 1

# 官方 prompts/linking.py 的 query_to_passage 指令（逐字；只有这一条被用到）
_QUERY_INSTRUCTION = (
    "Given a question, retrieve relevant sentences that best answer the question. "
    "Please make sure each sentence is connected to the next through at least a shared entity.")

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)
_ALNUM_RE = re.compile(r"[^A-Za-z0-9]")
_ENTITY_PREFIX = "entity-"
_CHUNK_PREFIX = "chunk-"
_PROP_PREFIX = "proposition-"


# ══════════════════════════════════════════════════════════════════════════════
# 官方数值/命名工具（逐字对齐 misc_utils.py）
# ══════════════════════════════════════════════════════════════════════════════
def min_max_normalize(x) -> np.ndarray:
    """`misc_utils.py:253-254`：全相等时返回 NaN（官方如此，调用方必须挡 NaN）。"""
    x = np.asarray(x, dtype=np.float64)
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi == lo:
        out = np.full(x.shape, np.nan)
        return out
    return (x - lo) / (hi - lo)


def chunk_key(text: str) -> str:
    """官方 `compute_mdhash_id(text, "chunk-")` = **字符串拼接**（不是哈希）。"""
    return _CHUNK_PREFIX + text


def entity_key(text: str) -> str:
    return _ENTITY_PREFIX + text


def prop_key(text: str) -> str:
    return _PROP_PREFIX + text


def _json_from_text(text: str):
    """等价官方 `safe_json_load` / `_extract_*_from_response`：直接解析 → 抓第一个 {...}。"""
    if not text:
        return None
    if isinstance(text, dict):
        return text
    try:
        return json.loads(text)
    except Exception:                       # noqa: BLE001
        pass
    m = _JSON_OBJ_RE.search(text)
    if not m:
        return None
    blob = m.group(0)
    try:
        return json.loads(blob)
    except Exception:                       # noqa: BLE001
        pass
    try:                                    # 官方 NER 那条用 eval（单引号也认）
        import ast
        return ast.literal_eval(blob)
    except Exception:                       # noqa: BLE001
        return None


# ══════════════════════════════════════════════════════════════════════════════
# 官方 prompt（按路径 exec，不 import 其包：`src.proprag` 依赖 torch/transformers）
# ══════════════════════════════════════════════════════════════════════════════
def _official_convert_format_to_template(original_string: str, placeholder_mapping=None,
                                         static_values=None) -> str:
    """官方 `utils/llm_utils.py:32-66` 的 `convert_format_to_template`（逐字等价）。

    语义：`{name}` → `${name}`（可经 placeholder_mapping 改名）；static_values 里的
    占位符直接填值。官方 prompt 文件调用时两个参数都是 None，这里为接口兼容全实现。
    """
    placeholder_mapping = placeholder_mapping or {}
    static_values = static_values or {}
    pattern = re.compile(r"\{(\w+)\}")

    def _sub(m):
        name = m.group(1)
        if name in static_values:
            return str(static_values[name])
        return f"${{{placeholder_mapping.get(name, name)}}}"

    return pattern.sub(_sub, original_string)


def _load_official_prompts(repo) -> dict:
    """按路径 exec 官方 prompt 文件。

    `proposition_extraction.py` 首行是 `from ...utils.llm_utils import
    convert_format_to_template`（相对导入）。exec 没有包上下文、且官方 `utils` 包依赖
    很重，所以给 exec 的命名空间预置**官方同名同义**的实现，并在 exec 失败时
    （ImportError/KeyError）把首行相对导入替换成预置实现再试一次——prompt 文本本身
    始终逐字来自官方文件。
    """
    from pathlib import Path
    tdir = Path(repo) / "src" / "proprag" / "prompts" / "templates"
    files = {
        "ner_expanded": tdir / "ner_expanded.py",
        "proposition_extraction": tdir / "proposition_extraction.py",
        "rag_qa_musique": tdir / "rag_qa_musique.py",
        "linking": Path(repo) / "src" / "proprag" / "prompts" / "linking.py",
    }
    out: dict = {}
    for name, p in files.items():
        if not p.exists():
            raise SourceNotReady(
                f"[proprag] 官方 prompt 缺失：{p}\n"
                f"  → 把官方仓库整个目录放到 baselines/PropRAG-main/"
                f"（需 src/proprag/prompts/templates/*.py）")
        src = p.read_text(encoding="utf-8")
        ns: dict = {"__name__": f"proprag_official_{name}", "__file__": str(p),
                    "convert_format_to_template": _official_convert_format_to_template}
        try:
            exec(compile(src, str(p), "exec"), ns)                          # noqa: S102
        except Exception:                                                   # noqa: BLE001
            # 去掉那行相对导入后重试（其余内容逐字不变）
            src2 = "\n".join(l for l in src.splitlines()
                             if "from ...utils" not in l and "from ..utils" not in l)
            ns = {"__name__": f"proprag_official_{name}", "__file__": str(p),
                  "convert_format_to_template": _official_convert_format_to_template}
            exec(compile(src2, str(p), "exec"), ns)                         # noqa: S102
        out[name] = ns
    return out


def _render(template: list, **kwargs) -> list[dict]:
    """官方 `prompt_template` 是 [{role, content}]，content 里带 `${passage}` 之类占位。"""
    subs = {f"${{{k}}}": str(v) for k, v in kwargs.items()}
    out = []
    for m in template or []:
        content = str(m.get("content", ""))
        for k, v in subs.items():
            content = content.replace(k, v)
        out.append({"role": m.get("role", "user"), "content": content})
    return out


def _chat(messages: list[dict], timeout: int, max_tokens: int) -> str:
    """多轮消息版 chat：官方 prompt 是 system/user/assistant 多轮，不能塌成 system+user。

    ⚠️ **这里曾有一个把整份抽取打死的 bug（2wiki 3356 块实测 0 实体 0 命题）**：
    `chat_messages()`（`common/llm.py:450`）**没有 `flatten` 形参**，且返回的是
    **(text, prompt_tokens, completion_tokens) 三元组**（`:529`）。旧写法
    `out = chat_messages(..., flatten=False)` 直接 `TypeError` → 被 `_extract_one` 的
    `except Exception` 吞掉 → 每块的 NER/命题都"解析失败" → 全库零实体零命题，
    **且一行报错都没有**（症状只有"抽取 N/N 块，命题 0 条"）。
    正确用法对齐 `hipporag_adapter.py:793` 的 `content, pt, ct = chat_messages(...)`。
    """
    out = chat_messages(messages, json_mode=False, timeout=timeout, max_tokens=max_tokens)
    text = out[0] if isinstance(out, tuple) else out   # 元组 = (text, pt, ct)
    if isinstance(text, dict):                        # 防御：个别路径可能回 dict
        text = text.get("text") or text.get("content") or ""
    return text or ""


# ══════════════════════════════════════════════════════════════════════════════
# 命题路径 beam search（官方 `graph_beam_search.py` 的 numpy 重写）
# ══════════════════════════════════════════════════════════════════════════════
class BeamSearchPathFinder:
    """官方 `BeamSearchPathFinder` 逐行对齐（去掉 torch 依赖）。

    依赖 duck-typed 的 `rag`：graph / node_name_to_vertex_idx / prop_ids / prop_vectors() /
    prop_text() / prop_entities / entity_to_props / query_vector() / batch_encode()。
    """

    def __init__(self, rag, beam_width=4, max_path_length=3,
                 embedding_combination="average", sim_threshold=0.75,
                 second_stage_filter_k=0):
        self.rag = rag
        # ★ 这些都是**默认值**，不是每题状态：真正的值由 `find_paths()` 每次显式传入。
        #   为什么必须搬出去：作答期 32 个 worker 共用一个 finder 实例，原先
        #   `find_paths()` 会在自己身上改 `active_graph` / `node_name_to_vertex_idx` /
        #   `beam_width` / `max_path_length` / `second_stage_filter_k` 再 `finally` 还原，
        #   而 `score_paths()` 还会临时改 `embedding_combination`。多线程互踩 →
        #   同义边按**别人的图**算、beam 参数取到别人的值，结果随机。
        self.beam_width = beam_width
        self.max_path_length = max_path_length
        self.embedding_combination = embedding_combination
        self.sim_threshold = sim_threshold
        self.second_stage_filter_k = second_stage_filter_k
        self.entity_to_propositions_map = rag.entity_to_props
        self.clear_caches()

    def clear_caches(self):
        """清空两个查询缓存。

        ⚠️ **只在「图变了」之后调用**（`build()` 末、`_load_artifacts()` 末），
        **不要**在 `retrieve()` 热路径里调 —— 那两个缓存按 `(图, key[, prop_set])`
        分槽，作答期图是只读的，跨题长期有效；每题清一次等于把 memo 全废掉
        （原先 `retrieve()` 里每题清两次，`find_connected_propositions` 在 3 层
        beam 里被反复调，这是作答期 CPU 的大头）。
        """
        self.synonymous_entities_cache: dict = {}
        self.connected_propositions_cache: dict = {}

    # ── 官方 :215-266：同义边 = **非整数 float 且 ≥ sim_threshold**（E1 是整数、E2 是 1.0）──
    def find_synonymous_entities(self, ekey: str, graph, node_index: dict):
        """返回 [(同义实体 key, similarity)]。

        `graph` / `node_index` 必须**显式传入**（原先读 `self.active_graph`，被多线程
        轮流改成全图/子图，互相看到对方的图）。缓存 key 含图身份，全图与子图的结果
        天然分槽，不会串。
        """
        ck = (id(graph), ekey)
        if ck in self.synonymous_entities_cache:
            return self.synonymous_entities_cache[ck]
        out: list[tuple[str, float]] = []
        idx = node_index.get(ekey)
        if idx is not None and graph is not None:
            seen = set()
            for nb in graph.neighbors(idx, mode="all"):
                nm = graph.vs[nb]["name"]
                if nm in seen or not nm.startswith(_ENTITY_PREFIX):
                    continue
                seen.add(nm)
                eid = graph.get_eid(idx, nb, error=False)
                if eid == -1:
                    continue
                w = graph.es[eid]["weight"]
                if isinstance(w, float) and (not float(w).is_integer()) and w >= self.sim_threshold:
                    out.append((nm, float(w)))
        self.synonymous_entities_cache[ck] = out
        return out

    # ── 官方 :268-359：共享实体 / 同义实体连接的命题 ──
    def find_connected_propositions(self, pkey: str, graph, node_index: dict,
                                    prop_set=None):
        """返回 (相连命题, 已处理集合)。

        ★ 缓存 key **必须含 `prop_set` 的身份**：本函数内有
          `if prop_set is not None and other not in prop_set: continue` 这一层过滤，
          换一个 `prop_set` 结果就不同。原先 key 只有 `pkey` → 第一阶段
          （`prop_set=None`，全图自由连）算出的结果被第二阶段（子图那 50 篇的命题集）
          直接复用 → **子图 beam 会拿到不在子图里的命题**，还会带 `KeyError` 风险。
          `prop_set` 是每题的临时 set，用 `id()` 当槽位即可（生命周期内不释放，
          id 不会复用；图/题跑完由 `clear_caches()` 统一清）。
        """
        ck = (id(graph), id(prop_set), pkey)
        if ck in self.connected_propositions_cache:
            return self.connected_propositions_cache[ck]
        connected, processed = [], set()
        entities = self.rag.prop_entities.get(pkey, [])
        if not entities:
            self.connected_propositions_cache[ck] = (connected, processed)
            return connected, processed
        for etext in entities:
            ekey = entity_key(etext)
            cand = [(other, [{"type": "exact", "entity1": ekey, "entity2": ekey,
                              "similarity": 1.0}])
                    for other in self.entity_to_propositions_map.get(ekey, [])]
            for syn_key, sim in self.find_synonymous_entities(ekey, graph, node_index):
                cand.extend((other, [{"type": "synonym", "entity1": ekey,
                                      "entity2": syn_key, "similarity": sim}])
                            for other in self.entity_to_propositions_map.get(syn_key, []))
            for other, conn in cand:
                if other == pkey or other in processed:
                    continue
                if prop_set is not None and other not in prop_set:
                    continue
                connected.append((other, conn))
                processed.add(other)
        self.connected_propositions_cache[ck] = (connected, processed)
        return connected, processed

    # ── 官方 batch_score_paths :387-582 ──
    def score_paths(self, paths: list[dict], qvec: np.ndarray,
                    embedding_combination: str | None = None) -> list[float]:
        """给一组路径打分。

        `embedding_combination` 显式传入（原先是临时改 `self.embedding_combination`
        再还原 → 多线程互踩）。
        """
        if not paths:
            return []
        comb_mode = embedding_combination or self.embedding_combination
        keys = [p["propositions"] for p in paths]
        if len({len(k) for k in keys}) > 1:
            raise ValueError("All paths must have the same length for batch processing")
        if comb_mode == "concatenate":
            texts = [" ".join(self.rag.prop_text(k) for k in ks) for ks in keys]
            comb = np.asarray(self.rag.batch_encode(texts), dtype=np.float64)
        else:                                       # 官方默认分支 = average
            mat = np.array([self.rag.prop_vectors(ks) for ks in keys], dtype=np.float64)
            comb = np.mean(mat, axis=1)
        norms = np.linalg.norm(comb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return [float(s) for s in np.matmul(comb / norms, np.asarray(qvec).squeeze())]

    # ── 官方 find_paths :584-897 ──
    def find_paths(self, query: str, prop_set=None, query_vector=None, *,
                   graph=None, node_index: dict | None = None,
                   beam_width: int | None = None,
                   max_path_length: int | None = None,
                   second_stage_filter_k: int | None = None,
                   embedding_combination: str | None = None):
        """两阶段 beam。

        ★ `graph` / `node_index` / `beam_width` / `max_path_length` /
          `second_stage_filter_k` **全部显式传入**（默认 = finder 构造时的值）。
          原先这些都是在 `self` 上改来改去再 `finally` 还原 —— 作答期 32 线程共用
          一个 finder 时，A 线程还原成子图、B 线程正拿它当全图用，结果随机。
          本函数现在**不修改任何 `self` 状态**。
        """
        graph = self.rag.graph if graph is None else graph
        node_index = self.rag.node_name_to_vertex_idx if node_index is None else node_index
        bw = self.beam_width if beam_width is None else beam_width
        mpl = self.max_path_length if max_path_length is None else max_path_length
        ssfk = self.second_stage_filter_k if second_stage_filter_k is None else second_stage_filter_k
        comb_mode = embedding_combination or self.embedding_combination

        qvec = query_vector if query_vector is not None else self.rag.query_vector(query)
        ids = list(prop_set) if prop_set is not None else self.rag.prop_ids
        if not ids:
            return [], [], []
        scores = np.asarray(np.dot(self.rag.prop_vectors(ids), np.asarray(qvec).squeeze()))
        scores = scores.reshape(-1)
        top_k = min(PROPRAG_BEAM_SEED_TOP, len(scores))
        top_idx = np.argpartition(scores, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

        beam = [{"propositions": [ids[int(i)]], "connections": [],
                 "score": float(scores[int(i)])} for i in top_idx]
        beam.sort(key=lambda x: x["score"], reverse=True)          # 稳定（官方同）
        initial_props = [p["propositions"][0] for p in beam[:PROPRAG_KEEP_INITIAL_PROPS]]
        beam = beam[:bw]
        all_paths = list(beam)
        seen = {frozenset(p["propositions"]) for p in all_paths}
        other_candidates: list = []

        for depth in range(2, mpl + 1):
            new_candidates: list = []
            seen_here: set = set()
            for path in beam:
                last = path["propositions"][-1]
                connected, _ = self.find_connected_propositions(
                    last, graph, node_index, prop_set)
                connected = list(connected)
                conn_keys = {c[0] for c in connected}          # 官方 :698-700 并行推理候选
                for ip in initial_props:
                    if ip not in conn_keys and ip != last and ip not in path["propositions"]:
                        connected.append((ip, [{"type": "parallel", "entity1": ip,
                                                "entity2": ip, "similarity": 1.0}]))
                for nxt, conn in connected:
                    if nxt in path["propositions"]:
                        continue
                    fs = frozenset(path["propositions"] + [nxt])
                    if fs in seen or fs in seen_here:
                        continue
                    seen_here.add(fs)
                    new_candidates.append({
                        "propositions": path["propositions"] + [nxt],
                        "connections": path["connections"] + [{
                            "from_prop": last, "to_prop": nxt, "entity_connections": conn}],
                        "score": 0.0})
            if not new_candidates:
                break
            for p, s in zip(new_candidates,
                            self.score_paths(new_candidates, qvec,
                                             embedding_combination=comb_mode)):
                p["score"] = float(s)
            new_candidates.sort(key=lambda x: x["score"], reverse=True)
            # 官方 :759-791：第二阶段用 concatenate 重打分并**覆盖** score
            if ssfk > 0 and len(new_candidates) > bw:
                head = new_candidates[:ssfk]
                cs = self.score_paths(head, qvec, embedding_combination="concatenate")
                for p, s in zip(head, cs):
                    p["concatenate_score"] = float(s)
                    p["score"] = float(s)
                head.sort(key=lambda x: x["score"], reverse=True)
                new_candidates = head[:ssfk]
            beam = new_candidates[:bw]
            all_paths.extend(beam)
            for p in beam:
                seen.add(frozenset(p["propositions"]))
            if depth < mpl:
                other_candidates.append(
                    sorted(all_paths, key=lambda x: x["score"], reverse=True)[:bw])

        processed = []
        for p in all_paths:
            q = dict(p)
            q["proposition_keys"] = list(p["propositions"])
            q["proposition_texts"] = [self.rag.prop_text(k) for k in p["propositions"]]
            q["length"] = len(p["propositions"])
            # 官方后处理把 entity key 换回**文本**（get_entities_from_paths 依赖这点）
            q["connections"] = [
                {"from_prop": c["from_prop"], "to_prop": c["to_prop"],
                 "entity_connections": [
                     {"type": d["type"],
                      "entity1": d["entity1"][len(_ENTITY_PREFIX):],
                      "entity2": d["entity2"][len(_ENTITY_PREFIX):],
                      "similarity": d["similarity"]} for d in c["entity_connections"]]}
                for c in p["connections"]]
            processed.append(q)
        processed.sort(key=lambda x: x["score"], reverse=True)
        return processed, other_candidates, [dict(p) for p in beam]

    # ── 官方 get_entities_from_paths :924-961 ──
    def get_entities_from_paths(self, paths: list[dict]):
        scores: dict[str, list[float]] = {}
        for path in paths:
            ps = float(path["score"])
            # processed 路径有 proposition_keys；raw（return 的第三项 initial_paths）没有 → 兼容
            pkeys = path.get("proposition_keys") or path.get("propositions") or []
            for conn in path["connections"]:
                for d in conn["entity_connections"]:
                    if d["entity1"] != d["entity2"]:        # 官方：给被连接到的实体（entity2）加分
                        scores.setdefault(entity_key(d["entity2"]), []).append(ps)
            for pk in pkeys:
                for etext in self.rag.prop_entities.get(pk, []):
                    scores.setdefault(entity_key(etext), []).append(ps)
        return sorted(scores.items(), key=lambda x: float(np.sum(x[1])), reverse=True)


# ══════════════════════════════════════════════════════════════════════════════
# 适配器主体
# ══════════════════════════════════════════════════════════════════════════════
class PropRAG(BaseMethod):
    """语料级：build() 抽命题建图（落盘可复用），query() 走官方两阶段 PPR + beam。"""

    name = "proprag"

    def __init__(self):
        self.dataset = ""
        self.work_dir = None
        self.passage_weight = PROPRAG_PASSAGE_WEIGHT
        self._qcache: dict = {}
        self._lock = threading.Lock()
        self._P: dict = {}
        # 检索期结构
        self.passages: list[str] = []
        self.passage_locs: list[list] = []
        self.chunk_vecs: np.ndarray | None = None
        self.prop_ids: list[str] = []
        self.prop_texts: list[str] = []
        self.prop_vecs: np.ndarray | None = None
        self.prop_entities: dict[str, list[str]] = {}
        self.entity_to_props: dict[str, list[str]] = {}
        self.entity_ids: list[str] = []
        self.entity_vecs: np.ndarray | None = None
        self.chunk_propositions: dict[str, list[str]] = {}
        self.graph = None
        self.node_name_to_vertex_idx: dict[str, int] = {}
        self._pk2pos: dict[str, int] = {}
        self._beam: BeamSearchPathFinder | None = None      # 兼容占位（实际用每线程实例）
        self._beam_gen = 0

    # ──────────────────────────────────────────────────────────────────────
    # 建图
    # ──────────────────────────────────────────────────────────────────────
    def build(self, corpus_contexts: list[dict], working_dir: str = "") -> "PropRAG":
        from pathlib import Path
        repo = require_repo(PROPRAG_REPO_CANDIDATES, self.name,
                            "官方仓库（src/proprag/prompts/templates/*.py）",
                            "把官方仓库整个目录放到 baselines/PropRAG-main/（含 .py）")
        self._P = _load_official_prompts(repo)
        try:
            import igraph  # noqa: F401
        except Exception as e:                      # noqa: BLE001
            raise SourceNotReady(
                f"[proprag] 缺 python-igraph（PPR 用）：{e}\n"
                f"  → pip install python-igraph>=0.10（requirements.txt 里已有）") from e

        # 语料单元 = 整段（不切块，title\ntext 原样，与 hipporag 适配器同口径）
        passages, locs = self._chunk_corpus(corpus_contexts)
        if not passages:
            raise RuntimeError("[proprag] 语料为空")
        self.passages, self.passage_locs = passages, locs

        base = Path(working_dir) if working_dir else \
            Path("data") / self.name / (self.dataset or "unknown") / "v1"
        self.work_dir = base / PROPRAG_ARTIFACT_DIRNAME
        self.work_dir.mkdir(parents=True, exist_ok=True)
        art, meta = self.work_dir / "artifacts.npz", self.work_dir / "meta.json"

        if art.exists() and meta.exists() and self._artifact_ok(meta, len(passages)):
            print(f"[proprag] 复用已建命题图：{self.work_dir}（不重新抽命题）", flush=True)
            self._load_artifacts(art, meta)
            # passage_locs 不进产物：按同一套切块参数**重算一遍**（纯本地、无 LLM），
            # 否则复载路径下 evidence_locs 全空、命中率指标直接坏掉。
            self.passage_locs = self._chunk_locs(corpus_contexts)
            if len(self.passage_locs) != len(self.passages):
                print(f"[proprag] ⚠ 复载的 passage 数（{len(self.passages)}）与重新切块"
                      f"（{len(self.passage_locs)}）不一致 → 产物可能来自别的切块参数，"
                      f"删掉 {self.work_dir} 重建", flush=True)
                self.passage_locs = [[] for _ in self.passages]
            return self

        print(f"[proprag] 建图：{len(passages)} 个 passage（整段，不切块）→ 官方 prompt 抽 "
              f"NER + 命题（每块 2 次 LLM，workers={PROPRAG_EXTRACT_WORKERS}；"
              f"生成上限 NER={PROPRAG_NER_MAX_NEW_TOKENS} / 命题={PROPRAG_PROP_MAX_NEW_TOKENS}，"
              f"逐条对齐官方 openie_vllm_offline.py:38/:45）；"
              f"样例={passages[0]!r}", flush=True)
        openie = self._extract_propositions(passages)
        self._assemble(openie)
        self._build_graph()
        self._save_artifacts(art, meta)
        return self

    def _chunk_corpus(self, corpus_contexts: list[dict]) -> tuple[list[str], list[list]]:
        """语料 → (passage 文本, 每篇的 (title, sent_idx) 定位)。

        与 hipporag / hypergraphrag / lightrag 同口径：交给 `build_chunks` 拼
        `title\\ntext`；切块口径按数据集解析（triviaqa 1200/100，其余整段不切）。

        ★ **不要在这里自己拼 `title\\ntext`**：`build_chunks` 内部已经拼了
          （`standard_rag.py` 的 `head = f"{title}\\n" if title else ""`）。早先这里
          把 `full` 同时塞进 `text` 与 `sentences`，`build_chunks` 再拼一次标题 →
          **passage 文本里标题出现两遍**（`'Doc A\\nDoc A\\nA is a person...'`）。
          后果是多花 embedding/抽取 token、且正文被重复标题稀释；实测输出里能看到。
          同理也不需要把 sentences 置成 `[full]` —— 上游 `merged_contexts` 给的
          context 本来就带真实 `sentences`（`corpus.py:46-63` 原样透传），
          丢掉它反而让 `sent_locs` 只剩一段。

        ★ **不是 `@staticmethod`**：它要读 `self.dataset` 才能选对切块口径。
          写 `@staticmethod` 时这一行会抛 `NameError: name 'self' is not defined`，
          而 `run.py:464` 只打印 `[proprag] 不可用：...`（不写 traceback）→
          表现就是"proprag 整跑不起来，只留一行 not defined"。
        """
        passages, locs = [], []
        for c in corpus_contexts:
            # 返回值是 (title, start_char, chunk_text, locs) —— 第 3 项才是正文！
            for _title, _cs, text, lc in build_chunks([c], dataset=getattr(self, "dataset", "")):
                if text.strip():
                    passages.append(text)
                    locs.append(list(lc))
        return passages, locs

    def _chunk_locs(self, corpus_contexts: list[dict]) -> list[list]:
        return self._chunk_corpus(corpus_contexts)[1]

    def _artifact_ok(self, meta, n_passages: int) -> bool:
        from config import EXTRACT_CHUNK_TOKENS, EXTRACT_CHUNK_OVERLAP
        try:
            m = json.loads(meta.read_text(encoding="utf-8"))
        except Exception:                       # noqa: BLE001
            return False
        return (m.get("version") == PROPRAG_ARTIFACT_VERSION
                and m.get("n_passages") == n_passages
                and m.get("syn_threshold") == PROPRAG_SYN_THRESHOLD
                and (m.get("chunk_tokens"), m.get("chunk_overlap"))
                == (EXTRACT_CHUNK_TOKENS, EXTRACT_CHUNK_OVERLAP))

    # ── 官方两步抽取（EnhancedOpenIE.ner → PropositionExtractor.extract_propositions）──
    def _extract_one(self, text: str) -> dict:
        """官方两步抽取：`EnhancedOpenIE.ner` → `PropositionExtractor.extract_propositions`。

        ★ 重试口径（`PROPRAG_RETRY=2` → 封顶 3 次）**只对解析失败生效**，两个守卫：
          ① **解析成功即算成功，哪怕结果空列表** —— `temperature=0`（`common/llm.py:479`）
             下同一 prompt 复现同一份输出，空结果重抽变不出实体来，纯白烧。
             旧写法 `if ents:` / `if props:` 把"抽出来是空的"也当失败去重抽，
             与"**解析失败**最多再抽两次"的口径不符（那是本项目的声明口径）。
          ② **命题层的输入实体为空时不做重抽** —— 命题 prompt 拿的是
             `named_entities=json.dumps(ents)`（见下），`ents` 为空就是发 `"[]"` 进去，
             **结构性注定失败**（与 hipporag 的 NER→三元组同一个坑）。
             首抽仍照官方发出（不改官方调用次数），只省掉后面那几次。
        """
        ner_tpl = self._P["ner_expanded"].get("prompt_template") or []
        ents: list[str] = []
        err: str = ""                               # ★ 记异常：否则系统性 TypeError 会被静默吞掉
        for _ in range(PROPRAG_RETRY + 1):
            try:
                with role("ner"):
                    out = _chat(_render(ner_tpl, passage=text), PROPRAG_LLM_TIMEOUT,
                                PROPRAG_NER_MAX_NEW_TOKENS)
            except Exception as e:                   # noqa: BLE001  传输/超时也算一次
                out = None
                err = err or f"ner {type(e).__name__}: {e}"
            data = _json_from_text(out)
            if isinstance(data, dict):              # ★ 守卫①：解析成功 → 停（空列表也是成功）
                ents = [str(e).strip() for e in (data.get("entities") or []) if str(e).strip()]
                break
        ents = list(dict.fromkeys(ents))            # 官方 ner() 的 `dict.fromkeys` 去重

        prop_tpl = self._P["proposition_extraction"].get("prompt_template") or []
        props: list[dict] = []
        _tries = (PROPRAG_RETRY + 1) if ents else 1   # ★ 守卫②：实体为空 → 只发首抽
        for _ in range(_tries):
            try:
                with role("proposition"):
                    out = _chat(_render(prop_tpl, passage=text, named_entities=json.dumps(ents)),
                                PROPRAG_LLM_TIMEOUT, PROPRAG_PROP_MAX_NEW_TOKENS)
            except Exception as e:                  # noqa: BLE001  同上
                out = None
                err = err or f"prop {type(e).__name__}: {e}"
            data = _json_from_text(out)
            if isinstance(data, dict):              # ★ 守卫①：解析成功 → 停
                for p in (data.get("propositions") or []):
                    if not isinstance(p, dict):
                        continue
                    pt = str(p.get("text") or "").strip()
                    if not pt:
                        continue
                    props.append({"text": pt,
                                  "entities": [str(e).strip() for e in (p.get("entities") or [])
                                               if str(e).strip()]})
                break
        return {"entities": ents, "propositions": props, "error": err}

    def _extract_propositions(self, passages: list[str]) -> list[dict]:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        out: list = [{"entities": [], "propositions": []} for _ in passages]
        with phase("build"):
            with ThreadPoolExecutor(max_workers=PROPRAG_EXTRACT_WORKERS) as ex:
                futs = {ex.submit(self._extract_one, t): i for i, t in enumerate(passages)}
                done = 0
                for fut in as_completed(futs):
                    i = futs[fut]
                    try:
                        out[i] = fut.result()
                    except Exception as e:                  # noqa: BLE001
                        out[i] = {"entities": [], "propositions": [], "error": str(e)}
                    done += 1
                    if done % 200 == 0 or done == len(passages):
                        print(f"[proprag] 抽取 {done}/{len(passages)} 块，命题 "
                              f"{sum(len(o.get('propositions') or []) for o in out)} 条",
                              flush=True)
        empty = sum(1 for o in out if not (o.get("propositions") or []))
        if empty:
            # ★ 把异常原因带出来：否则「全库 0 命题」只能靠猜（曾有一个 `flatten=` TypeError
            #   被 except 吞掉，整份 2wiki 跑完才从"命题 0 条"看出不对）。
            errs = [o.get("error") for o in out if o.get("error")]
            tail = f"；其中 {len(errs)} 块有异常，例如 → {errs[0]}" if errs else ""
            print(f"[proprag] ⚠ {empty} 块没抽出命题（这些块在图上只有 passage 顶点）{tail}",
                  flush=True)
        return out

    # ── 官方 index() 的 :203-268：三张表 ──
    def _assemble(self, openie: list[dict]):
        self.prop_ids, self.prop_texts, self.prop_entities = [], [], {}
        self.entity_to_props, self.chunk_propositions = {}, {}
        prop_ent_lists: list[list[str]] = []
        for text, info in zip(self.passages, openie):
            ck = chunk_key(text)
            keys: list[str] = []
            for p in (info.get("propositions") or []):
                pk = prop_key(p["text"])
                keys.append(pk)                 # 同名命题跨 passage 合并（官方用 dict 天然去重）
                if pk in self.prop_entities:
                    continue
                self.prop_ids.append(pk)
                self.prop_texts.append(p["text"])
                self.prop_entities[pk] = list(dict.fromkeys(p["entities"]))
                prop_ent_lists.append(self.prop_entities[pk])
            self.chunk_propositions[ck] = keys
        # 官方 `extract_proposition_entities`：实体表 = np.unique(所有命题实体)（排序）
        flat = [e for lst in prop_ent_lists for e in lst]
        self.entity_ids = [entity_key(str(e)) for e in np.unique(np.array(flat, dtype=object))] \
            if flat else []
        for pk, ents in self.prop_entities.items():
            for e in ents:
                self.entity_to_props.setdefault(entity_key(e), []).append(pk)
        print(f"[proprag] passage {len(self.passages)} / 命题 {len(self.prop_ids)} / "
              f"实体 {len(self.entity_ids)}", flush=True)

    def _build_graph(self):
        """官方 E1（:944-995）+ E2（:569-611）+ E3（:613-674）+ augment（:806-892）。

        顶点顺序官方是「实体在前、passage 在后」（`add_new_nodes` :819-849）。
        """
        import igraph as ig
        from config import EMBED_MODEL_SERVED          # ★ 别再写死 bge-large（早换 Qwen3 了）
        print(f"[proprag] 编码 passage / 实体 / 命题向量（{EMBED_MODEL_SERVED}，文档侧不加指令）",
              flush=True)
        with phase("build"):
            self.chunk_vecs = embed_batch([fit_embed(t) for t in self.passages])
            self.entity_vecs = embed_batch([fit_embed(e[len(_ENTITY_PREFIX):])
                                            for e in self.entity_ids]) if self.entity_ids \
                else np.zeros((0, self.chunk_vecs.shape[1]), dtype=np.float32)
            self.prop_vecs = embed_batch([fit_embed(t) for t in self.prop_texts]) \
                if self.prop_texts else np.zeros((0, self.chunk_vecs.shape[1]), dtype=np.float32)

        print("[proprag] 建图（E1 命题内实体全连接 / E2 passage-实体 / E3 同义实体）", flush=True)
        stats: dict[tuple, float] = {}
        # E1：命题内实体两两 +1（两个方向都写 ⇒ 无向图里两条平行边）
        for pk in self.prop_ids:
            ekeys = [entity_key(e) for e in self.prop_entities[pk]]
            for i in range(len(ekeys)):
                for j in range(i + 1, len(ekeys)):
                    a, b = ekeys[i], ekeys[j]
                    stats[(a, b)] = stats.get((a, b), 0.0) + 1
                    stats[(b, a)] = stats.get((b, a), 0.0) + 1
        # E2：passage → 它的全部实体，权重 1.0
        for text in self.passages:
            ck = chunk_key(text)
            ents = set()
            for pk in self.chunk_propositions.get(ck, []):
                ents.update(self.prop_entities.get(pk, []))
            for e in ents:
                stats[(ck, entity_key(e))] = 1.0
        # E3：同义实体（cosine ≥ 阈值、每点 ≤101 邻居），权重 = cosine
        n_syn = 0
        syn_pairs: dict[tuple, float] = {}          # 显式记「哪些边是同义边、cosine 多少」
        if len(self.entity_ids):
            ne = len(self.entity_ids)
            for s in range(0, ne, PROPRAG_KNN_BLOCK):
                e = min(ne, s + PROPRAG_KNN_BLOCK)
                sims = self.entity_vecs[s:e] @ self.entity_vecs.T     # 都已 L2 归一化
                for r in range(e - s):
                    key = self.entity_ids[s + r]
                    if len(_ALNUM_RE.sub("", key[len(_ENTITY_PREFIX):])) <= 2:   # 官方 :655
                        continue
                    row = sims[r]
                    k = min(PROPRAG_SYN_TOPK, len(row))
                    cand = np.argpartition(row, -k)[-k:]
                    cand = cand[row[cand] >= PROPRAG_SYN_THRESHOLD]
                    cand = cand[np.argsort(-row[cand])]
                    taken = 0
                    for j in cand:
                        if taken >= PROPRAG_SYN_MAX_NN:                 # 官方 `num_nns > 100`
                            break
                        other = self.entity_ids[int(j)]
                        if other == key:
                            continue
                        sc = float(row[int(j)])
                        stats[(key, other)] = stats.get((key, other), 0.0) + sc
                        stats[(other, key)] = stats.get((other, key), 0.0) + sc
                        # 两个方向都调用了 find_synonymous_entities，所以这里也双向登记
                        syn_pairs[(key, other)] = sc
                        syn_pairs[(other, key)] = sc
                        taken += 1
                        n_syn += 1
        print(f"[proprag] 同义边 {n_syn} 条（cosine ≥ {PROPRAG_SYN_THRESHOLD}）", flush=True)

        names = self.entity_ids + [chunk_key(t) for t in self.passages]
        idx = {nm: i for i, nm in enumerate(names)}
        # ★ 边类型要能判定：同义边 = cosine float；E1 计数 / E2 是整数或 1.0。
        #   官方靠「权重非整数」来区分，但 cos=1.0（或双向累加后成 2.0）会被误判成 E1 ——
        #   本文件把同义边显式写成**真实 cosine**（同义边的 cosine 永远 < 1，
        #   因为 1.0 只会是「自己和自己」；官方也在 :661 排除 nn == node_key），
        #   既保持官方「非整数 = 同义」的判定语义，又不落进那个坑。
        pairs = []
        for (a, b), w in stats.items():
            if a not in idx or b not in idx or a == b:
                continue
            pairs.append((idx[a], idx[b], float(syn_pairs.get((a, b), w))))
        g = ig.Graph(n=len(names), edges=[(a, b) for a, b, _ in pairs], directed=False)
        g.vs["name"] = names
        g.es["weight"] = [w for _, _, w in pairs]      # 平行边**保留**（官方不 simplify）
        self.graph = g
        self.node_name_to_vertex_idx = idx
        self._pk2pos = {k: i for i, k in enumerate(self.prop_ids)}
        print(f"[proprag] 图：{g.vcount()} 顶点（实体 {len(self.entity_ids)} + passage "
              f"{len(self.passages)}）/ {g.ecount()} 边（含平行边）", flush=True)
        self._make_beam()

    # ── 落盘 / 复载（官方只存 parquet+GraphML；这里用 npz，免 pandas 依赖）──
    def _save_artifacts(self, art, meta):
        np.savez_compressed(
            art,
            chunks=np.array(self.passages, dtype=object),
            chunk_vecs=self.chunk_vecs,
            entity_ids=np.array(self.entity_ids, dtype=object),
            entity_vecs=self.entity_vecs,
            prop_ids=np.array(self.prop_ids, dtype=object),
            prop_texts=np.array(self.prop_texts, dtype=object),
            prop_vecs=self.prop_vecs,
            graph_vertices=np.array(self.graph.vs["name"], dtype=object),
            graph_edges=np.array(self.graph.get_edgelist(), dtype=np.int64),
            graph_weights=np.array(self.graph.es["weight"], dtype=np.float64),
            allow_pickle=True)
        (self.work_dir / "prop_entities.json").write_text(
            json.dumps(self.prop_entities, ensure_ascii=False), encoding="utf-8")
        (self.work_dir / "chunk_props.json").write_text(
            json.dumps(self.chunk_propositions, ensure_ascii=False), encoding="utf-8")
        meta.write_text(json.dumps({
            "version": PROPRAG_ARTIFACT_VERSION, "dataset": self.dataset,
            "n_passages": len(self.passages), "n_props": len(self.prop_ids),
            "n_entities": len(self.entity_ids), "n_edges": self.graph.ecount(),
            "syn_threshold": PROPRAG_SYN_THRESHOLD,
            "chunk_tokens": EXTRACT_CHUNK_TOKENS,
            "chunk_overlap": EXTRACT_CHUNK_OVERLAP,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[proprag] 落盘 {art}（{art.stat().st_size / 1e6:.1f} MB）", flush=True)

    def _load_artifacts(self, art, meta):
        import igraph as ig
        d = np.load(art, allow_pickle=True)
        self.passages = [str(x) for x in d["chunks"]]
        self.chunk_vecs = d["chunk_vecs"]
        self.entity_ids = [str(x) for x in d["entity_ids"]]
        self.entity_vecs = d["entity_vecs"]
        self.prop_ids = [str(x) for x in d["prop_ids"]]
        self.prop_texts = [str(x) for x in d["prop_texts"]]
        self.prop_vecs = d["prop_vecs"]
        vnames = [str(x) for x in d["graph_vertices"]]
        pairs = [(int(a), int(b)) for a, b in d["graph_edges"]]
        g = ig.Graph(n=len(vnames), edges=pairs, directed=False)
        g.vs["name"] = vnames
        g.es["weight"] = [float(w) for w in d["graph_weights"]]
        self.graph = g
        self.node_name_to_vertex_idx = {nm: i for i, nm in enumerate(vnames)}
        self._pk2pos = {k: i for i, k in enumerate(self.prop_ids)}
        for fn, attr in (("prop_entities.json", "prop_entities"),
                         ("chunk_props.json", "chunk_propositions")):
            p = self.work_dir / fn
            if not p.exists():
                raise RuntimeError(f"[proprag] 产物缺 {fn}（{self.work_dir}）→ 删掉该目录重建")
            setattr(self, attr, json.loads(p.read_text(encoding="utf-8")))
        self.entity_to_props = {}
        for pk, ents in self.prop_entities.items():
            for e in ents:
                self.entity_to_props.setdefault(entity_key(e), []).append(pk)
        print(f"[proprag] 复载：{len(self.passages)} passage / {len(self.prop_ids)} 命题 / "
              f"{len(self.entity_ids)} 实体 / {self.graph.ecount()} 边", flush=True)
        self._make_beam()

    def _make_beam(self):
        """准备一个**世代号** + 每线程一个 finder 实例（`threading.local`）。

        ★ 为什么每线程一个：`find_paths` 里的两个缓存（同义边 / 相连命题）是**写**进
          finder 的 dict 的；作答期 32 个 worker 共用一个实例时会互相覆盖、把对方刚
          算好的 memo 冲掉（白算）。每线程一个之后缓存按线程独立、跨题长期有效。

        ★ 为什么用世代号而不是直接 `clear_caches()`：finder 是**每线程**的，主线程
          清不到别的线程的缓存；图换了之后那些线程的缓存就全是旧的。`_beam_gen` 每次
          建图/复载 +1，各线程首次用 finder 时发现自己的世代落后就自清。
        """
        self._beam_gen = int(getattr(self, "_beam_gen", 0)) + 1
        self._beam_tls = threading.local()

    def _beam_for_thread(self) -> BeamSearchPathFinder:
        """取本线程的 finder（首次访问时建；世代落后则自清缓存）。"""
        b = getattr(self._beam_tls, "beam", None)
        if b is None:
            b = BeamSearchPathFinder(
                self, beam_width=PROPRAG_BEAM_WIDTH,
                max_path_length=PROPRAG_MAX_PATH_LENGTH,
                embedding_combination="average", sim_threshold=PROPRAG_SYN_THRESHOLD,
                second_stage_filter_k=PROPRAG_SECOND_STAGE_K)
            self._beam_tls.beam = b
            self._beam_tls.gen = -1
        if getattr(self._beam_tls, "gen", -1) != self._beam_gen:
            b.clear_caches()                      # 图换过了 → 本线程缓存作废
            self._beam_tls.gen = self._beam_gen
        return b

    # ── BeamSearchPathFinder 需要的 duck-typed 接口（官方 rag.* 同名语义）──
    def prop_vectors(self, keys) -> np.ndarray:
        return np.array([self.prop_vecs[self._pk2pos[k]] for k in keys], dtype=np.float32)

    def prop_text(self, key) -> str:
        pos = self._pk2pos.get(key)
        return self.prop_texts[pos] if pos is not None else ""

    def batch_encode(self, texts, instruction: str = "") -> np.ndarray:
        """官方 `embedding_model.batch_encode(texts, ...)`：**文档侧不加指令**。"""
        if isinstance(texts, str):
            texts = [texts]
        return embed_batch([fit_embed(t) for t in texts])

    def query_vector(self, query: str) -> np.ndarray:
        """官方 `get_query_embeddings`：`Instruct: {query_to_passage}\\nQuery: ` + 问题。"""
        with self._lock:
            v = self._qcache.get(query)
        if v is not None:
            return v
        with phase("query"):
            full = f"Instruct: {_QUERY_INSTRUCTION}\nQuery: " + query
            v = _normalize_rows(_embed_request([full], timeout=60))[0]
        with self._lock:
            self._qcache[query] = v
        return v

    # ──────────────────────────────────────────────────────────────────────
    # PPR（官方 `run_ppr` :1728-1781）
    # ──────────────────────────────────────────────────────────────────────
    def _ppr(self, graph, reset: np.ndarray, damping: float) -> np.ndarray:
        reset = np.where(np.isnan(reset) | (reset < 0), 0, np.asarray(reset, dtype=np.float64))
        if float(np.sum(reset)) == 0:
            n = graph.vcount()
            scores = np.ones(n) / n if n else np.zeros(0)
            return scores
        pr = graph.personalized_pagerank(
            vertices=range(graph.vcount()), damping=damping, directed=False, weights="weight",
            reset=reset.tolist(), implementation="prpack")
        return np.asarray(pr, dtype=np.float64)

    # ──────────────────────────────────────────────────────────────────────
    # 检索（官方 `graph_search_with_proposition_entities` :1478-1726）
    # ──────────────────────────────────────────────────────────────────────
    def _dpr(self, query: str) -> np.ndarray:
        return min_max_normalize(self.chunk_vecs @ self.query_vector(query))

    def retrieve(self, query: str, want_paths: bool = True):
        """返回 (doc_ids 降序, doc_scores 同序, 第二阶段路径, focused 实体)。

        **下标空间**：igraph 的 PPR 分数是「图顶点空间」，而对外返回的 doc id 是
        「passage 空间」（`self.passages` 的下标，官方 = `passage_node_keys` 下标）。
        两者必须显式换算，否则 top-50 会取到实体顶点（官方靠 passage 顶点连续排在
        实体顶点之后才不出错；我们显式做，避免依赖顶点顺序的隐式假设）。
        """
        n = len(self.passages)
        if n == 0:
            return np.zeros(0, dtype=int), np.zeros(0), [], []
        dpr = self._dpr(query)
        qv = self.query_vector(query)
        nv = self.graph.vcount()
        ckeys = [chunk_key(t) for t in self.passages]
        # passage i ↔ 图顶点 vidx[i]；顶点 → passage 反向表
        vidx: list = [self.node_name_to_vertex_idx.get(ck) for ck in ckeys]
        vert2doc = {v: i for i, v in enumerate(vidx) if v is not None}

        def _doc_scores(vertex_scores: np.ndarray) -> np.ndarray:
            """图顶点分数 → passage 空间分数。"""
            return np.array([vertex_scores[v] if v is not None else 0.0 for v in vidx])

        def _stage1_ret():
            sc = _doc_scores(ppr1)
            order = np.argsort(sc)[::-1]
            return (order, sc[order],
                    paths[:PROPRAG_FOCUS_PATH_TOP], ents[:PROPRAG_FOCUS_PATH_TOP])

        # 1) 第一阶段 beam（长度 1）→ 二值 phrase 权重（官方 :1516-1529）
        beam = self._beam_for_thread()
        paths, _other, _init = beam.find_paths(query, query_vector=qv)
        ents = beam.get_entities_from_paths(paths[:PROPRAG_STAGE1_PATH_TOP])[
            :PROPRAG_STAGE1_ENTITY_TOP]
        phrase_w = np.zeros(nv)
        for ekey, _sc in ents:
            v = self.node_name_to_vertex_idx.get(ekey)
            if v is not None:
                phrase_w[v] = 1.0
        if float(phrase_w.max(initial=0.0)) > 0:
            phrase_w = np.nan_to_num(min_max_normalize(phrase_w), nan=0.0)
        # 2) passage 重启权重（官方 :1531-1540）
        passage_w = np.zeros(nv)
        for i, v in enumerate(vidx):
            if v is not None:
                passage_w[v] = dpr[i] * self.passage_weight
        node_w = phrase_w + passage_w
        if float(np.sum(node_w)) == 0:                      # 官方 fallback：退回 DPR 排序
            order = np.argsort(dpr)[::-1]
            return order, dpr[order], [], []
        # 3) 第一阶段 PPR（damping 0.75）
        ppr1 = self._ppr(self.graph, node_w, PROPRAG_PPR_DAMPING_1)
        s1 = _doc_scores(ppr1)
        stage1_docs = np.argsort(s1)[::-1]

        # 4) 聚焦子图 = top-50 passage ∪ 它们的 entity 邻居（官方 :1557-1598）
        top_keys = [ckeys[i] for i in stage1_docs[:PROPRAG_FOCUS_TOP_K]]
        top_verts = [self.node_name_to_vertex_idx[k] for k in top_keys
                     if k in self.node_name_to_vertex_idx]
        if not top_verts:
            return _stage1_ret()
        keep = set(top_verts)
        for v in top_verts:
            for nb in self.graph.neighbors(v, mode="all"):
                if self.graph.vs[nb]["name"].startswith(_ENTITY_PREFIX):
                    keep.add(nb)
        sub = self.graph.induced_subgraph(sorted(keep))
        if sub.vcount() == 0:
            return _stage1_ret()
        sub_idx = {v["name"]: i for i, v in enumerate(sub.vs)}
        orig_to_sub = {self.node_name_to_vertex_idx[nm]: i for nm, i in sub_idx.items()
                       if nm in self.node_name_to_vertex_idx}
        sub_to_name = {i: nm for nm, i in sub_idx.items()}

        # 5) 第二阶段 beam（宽度 4 / 长度 3 / prop_set = 子图那 50 篇的命题）
        #    ★ 图/下标表/beam 参数**全部作为参数传进去**，不再改 `self._beam` 上的状态
        #      再 `finally` 还原 —— 作答期 32 个 worker 共用一个 finder，改共享状态
        #      会被别的线程看到（A 还原成子图、B 正要拿它当全图用）。
        prop_set = set()
        for k in top_keys:
            prop_set.update(self.chunk_propositions.get(k, []))
        fpaths, _fo, finit = beam.find_paths(
            query, prop_set=prop_set, query_vector=qv,
            graph=sub, node_index=sub_idx,
            beam_width=PROPRAG_BEAM_WIDTH,
            max_path_length=PROPRAG_MAX_PATH_LENGTH,
            second_stage_filter_k=PROPRAG_SECOND_STAGE_K)
        fpaths = fpaths[:PROPRAG_FOCUS_PATH_TOP]
        f_ents = beam.get_entities_from_paths(fpaths)[:PROPRAG_FOCUS_PATH_TOP]
        i_ents = beam.get_entities_from_paths(finit)[:PROPRAG_FOCUS_PATH_TOP]

        # 6) 子图重启向量（官方 :1635-1682：各自 max() → 各自归一化 → 逐元素 max）
        sub_phrase = np.zeros(sub.vcount())
        sub_initial = np.zeros(sub.vcount())
        for ekey, scores in f_ents:
            v = self.node_name_to_vertex_idx.get(ekey)
            if v is not None and v in orig_to_sub and len(scores):
                sub_phrase[orig_to_sub[v]] = float(np.max(scores))
        for ekey, scores in i_ents:
            v = self.node_name_to_vertex_idx.get(ekey)
            if v is not None and v in orig_to_sub and len(scores):
                sub_initial[orig_to_sub[v]] = float(np.max(scores))
        if float(np.sum(sub_phrase)) > 0:
            sub_phrase = np.nan_to_num(min_max_normalize(sub_phrase), nan=0.0)
        if float(np.sum(sub_initial)) > 0:
            sub_initial = np.nan_to_num(min_max_normalize(sub_initial), nan=0.0)
        sub_phrase = np.maximum(sub_phrase, sub_initial)
        sub_passage = np.zeros(sub.vcount())
        for i, v in enumerate(vidx):
            if v is not None and v in orig_to_sub:
                sub_passage[orig_to_sub[v]] = dpr[i] * self.passage_weight
        sub_w = sub_phrase + sub_passage
        if float(np.sum(sub_w)) == 0:
            return _stage1_ret()

        # 7) 子图 PPR（damping 0.45）+ 合并（官方 :1688-1714）——全程 passage 空间
        sub_ppr = self._ppr(sub, sub_w, PROPRAG_PPR_DAMPING_2)
        final = s1.copy()
        lowest = float(np.min(sub_ppr)) if len(sub_ppr) else 0.0
        final = final * (lowest * 0.5)
        for si, sc in enumerate(sub_ppr):
            nm = sub_to_name.get(si, "")
            if nm.startswith(_CHUNK_PREFIX):
                d = vert2doc.get(self.node_name_to_vertex_idx.get(nm, -1))
                if d is not None:
                    final[d] = float(sc)
        order = np.argsort(final)[::-1]
        return order, final[order], (fpaths if want_paths else []), f_ents

    # ──────────────────────────────────────────────────────────────────────
    # 作答 + 对外接口
    # ──────────────────────────────────────────────────────────────────────
    def query(self, question: str) -> dict:
        if self.graph is None:
            raise RuntimeError("请先 build()")
        order, scores, paths, _ents = self.retrieve(question)
        top = [int(i) for i in order[:PROPRAG_TOP_DOCS]]
        evidence = [self.passages[i] for i in top]
        locs: list = []
        for i in top:
            if i < len(self.passage_locs):
                locs.extend(self.passage_locs[i])
        # 统一作答（唯一路径）：官方 `rag_qa_musique` 对照路径已删除
        answer = generate_answer(question, [{"text": t} for t in evidence],
                                 dataset=getattr(self, "dataset", ""))
        return {"answer": answer,
                "evidence": evidence,
                "evidence_locs": list(dict.fromkeys(locs)),
                "retrieval_paths": [p["proposition_texts"] for p in paths[:3]],
                "n_propositions": len(self.prop_ids),
                "n_entities": len(self.entity_ids),
                "top_doc_score": float(scores[0]) if len(scores) else 0.0}

    def run(self, question, contexts):
        raise NotImplementedError("PropRAG 走语料级：先 build(全部上下文) 再逐题 query")
