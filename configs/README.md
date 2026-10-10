# configs/ —— 参数管理入口

一次实验的所有**可调参数**写在 JSON 里，用 `--config` 指定；代码里的默认值只作兜底。

```bash
python run.py --config configs/default.json --method proprag --dataset hotpotqa --sample 500
python tools/build_all.py --config configs/baselines.json --methods proprag,ircot \
    --datasets hotpotqa,2wiki,musique --sample 500
```

## 优先级（高 → 低）

```
命令行显式参数  >  真实环境变量（export XXX=… 或命令行前缀）  >  配置文件  >  代码默认值
```

注入时机：`run.py` / `tools/build_all.py` 在**任何项目模块 import 之前**用一个只依赖
stdlib 的预扫描读出 `--config`，再 `apply_config_file()` 把键注入 `os.environ`。
因为 `config.py` 和各 adapter 的旋钮都是**在 import 时**读 `os.environ.get(...)` 的，
所以必须在最前面注入。已存在的环境变量不会被覆盖（临时覆盖仍然有效）。

## 键名规则

| 配置键 | 环境变量 | 谁读它 |
|---|---|---|
| `chat_model` | `CHAT_MODEL` | `config.py` |
| `extract_chunk_tokens` | `EXTRACT_CHUNK_TOKENS` | `config.py` |
| `defaults.workers` | （不注入） | `tools/build_all.py` 的 run 默认值 |
| `methods.proprag.top_docs` | `PROPRAG_TOP_DOCS` | `methods/proprag_adapter.py` |
| `methods.selfask.max_hops` | `SELFASK_MAX_HOPS` | `methods/selfask_adapter.py` |
| `methods.ircot.retrieval_count` | `IRCOT_RETRIEVAL_COUNT` | `methods/ircot_adapter.py` |

规则：扁平化 = 嵌套键用 `_` 连接、全大写；bool → `1`/`0`；list → 逗号连接；`null` 跳过。

## JSON 里哪些键真的能改？（四类键）

打开任一份 `configs/*.json` 会看到四类键，**只有前两类会起作用**：

| 类别 | 例子 | 生效吗 | 说明 |
|---|---|---|---|
| **① 顶层可注入键** | `"embed_dim": 1024` | ✅ 生效 | → `EMBED_DIM`，不跑哪个方法都生效。**这类才是你平时要改的** |
| **② `methods.<m>.*`** | `"methods": {"proprag": {"top_docs": 20}}` | ✅ 生效（仅当 `--method proprag`） | → `PROPRAG_TOP_DOCS`。前缀必须与方法的环境变量前缀一致，见下节 `method_prefixes` |
| **③ `defaults.*`** | `"defaults": {"workers": 8}` | ⚠️ 半生效 | **不注入环境变量**。只有 `tools/build_all.py` 读它当 run 级默认；`run.py` **不读**，`run.py` 的 workers/sample 必须命令行给 |
| **④ `_` 开头的键** | `"_readme"` / `"_why"` / `"_note_*"`（任意层级） | ❌ 不生效 | 纯注释，给人看的。加载器见到 `_` 开头就跳过（写 `"_note"` 当参数曾是个真 bug） |
| **⑤ 值是 `null`** | `"embed_dim": null` | ❌ 不生效 | 语义是"**跟随全局/代码默认**"。用于"只想声明这个键可调、但不想覆盖" |

自查：`python smoke_new_baselines.py` 里有一项会扫 `configs/*.json` 的每个键、确认代码里
真有人读它，**死键会被点名**（防止"改了配置却没生效"）。

## 三个容易踩的坑

1. **一次只能给一份 `--config`**。`run.py --config A --config B` 只有 A 生效（预扫描取第一个）。
   想同时要"官方对齐 + 换 embedder"，就把两边的键合到一份文件里（`configs/official-alignment.json`
   和 `configs/embedding-qwen3-8b.json` 都是完整可独立使用的，不是补丁）。
2. **已有的环境变量优先，配置文件不覆盖它**。`export EMBED_DIM=2048` 之后再 `--config …`，
   文件里的 `embed_dim` 会被忽略。所以"改了配置却没效果"先 `echo $EMBED_DIM` 看一眼。
3. **`null` 不是"设为空"，是"不注入"**。要让某方法显式跟随全局，写 `null` 或不写都行。

⚠️ **方法私有键必须长成 `<METHOD>_XXX`**（如 `methods.proprag.top_docs` → `PROPRAG_TOP_DOCS`）。
否则它会变成全局环境变量、影响别的方法 —— 这种键会被**跳过并打印警告**，不会静默生效。
方法名前缀一律用 `run.py --method` 的名字大写（`selfask` → `SELFASK_`、`s2grag` → `S2GRAG_`）。

## 跟官方数字对齐：`configs/official-alignment.json`

默认配置为了离线可跑 + 全项目同尺子，把官方 HGR/LightRAG 内部的 **tiktoken 换成了
embedding 的 tokenizer**，并按那个尺子把截断/分块调小（不切块、truncate 3000…）。
所以**默认列的绝对数字不该直接跟论文比**。要对齐官方就用这份配方，它把值改回官方原值：

```bash
python run.py --config configs/official-alignment.json --method hypergraphrag \
    --dataset hotpotqa --sample 500 --graph-version 3     # ← 必须开新 v
```

| 方法 | 官方的值（本配方已写好） |
|---|---|
| `hypergraphrag` | 内部 tiktoken(**gpt-4o**)；`truncate_list_by_token_size=4000`；分块 1200/100 |
| `lightrag` | 内部 tiktoken(gpt-4o-mini)；分块 1200/100；gleaning 1 |
| `s2grag` | `max_facts=3`（hotpotqa/2wikimultqa；triviaqa 官方是 1，且还需移植两个专属 prompt） |
| `ircot` | `max_para_words=350`（**词数**，本来就是官方值，只是显式写出来） |

⚠️ 这是**口径变更**：切块尺子一换，块边界 → 实体 → 图 → 向量全变，结果与默认配置
**不可比**，论文里必须单独标注为 alignment 实验，且必须开新 `--graph-version` 重建。
`hipporag/proprag/selfask/standardrag` 不涉及这个分歧，配方里不给它们任何键。

### `method_prefixes`：方法名 ≠ 环境变量前缀

`methods.<m>.*` 默认要求扁平名以 `<M>_` 开头。但有的适配器用的是**缩写前缀**：
`hypergraphrag` → `HGR_`。前缀对不上时那些键会被**跳过并打印警告**（而不是静默生效）。
在配置里声明一次即可：

```jsonc
{ "method_prefixes": {"hypergraphrag": "HGR"},
  "methods": {"hypergraphrag": {"tokenizer": "tiktoken"}} }   // → HGR_TOKENIZER
```

已知前缀表：`hypergraphrag`→`HGR_`、`lightrag`→`LIGHTRAG_`、`hipporag`→`HIPPORAG_`、
其余（`ircot/selfask/proprag/s2grag/standardrag`）与方法名一致。



embedding 模型决定三件事，散着改一定漏：**向量空间（维度）**、**截断尺子（tokenizer）**、
**查询侧前缀**。所以它们集中在 `config.py` 的「Embedding 模型」一节，全部可由配置/环境变量给：

| 配置键 | 环境变量 | 默认 | 作用 |
|---|---|---|---|
| `embed_model` | `EMBED_MODEL` | `BAAI/bge-large-en-v1.5` | HF 路径或本地目录；决定加载哪个 tokenizer |
| `embed_model_served` | `EMBED_MODEL_SERVED` | `bge-large-en-v1.5` | vLLM `--served-model-name`，必须逐字一致 |
| `embed_dim` | `EMBED_DIM` | 1024 | 向量维度，**必须 = 服务端真实输出**（首次嵌入会探测并告警） |
| `embed_max_tokens` | `EMBED_MAX_TOKENS` | 4096 | 单条文本喂 embedding 前的截断上限，**必须 ≤ 服务端 `--max-model-len`**（当前服务端 = 4096）。★ 不切块口径下这是**真实生效**的截断（triviaqa 整篇文章会被切到这里） |
| `embed_tokenizer` | `EMBED_TOKENIZER` | `embed` | embedding 侧尺子：`embed`（模型自带）/`tiktoken` |
| `chunk_tokenizer` | `CHUNK_TOKENIZER` | = `EMBED_TOKENIZER` | 切块/内部计数尺子；官方对账时设 `tiktoken` |
| `embed_request_tokens` | `EMBED_REQUEST_TOKENS` | 4096 | 单请求 token 预算，必须 ≤ 引擎 `--max-num-batched-tokens` |
| `embed_query_format` | `EMBED_QUERY_FORMAT` | `bge` | 查询前缀格式：`bge` / `qwen3` / `none` |
| `embed_query_task` | `EMBED_QUERY_TASK` | 英文检索任务描述 | `qwen3` 格式里 `Instruct:` 后的任务句 |
| `embed_query_template` | `EMBED_QUERY_TEMPLATE` | 空 | 非空则优先，支持 `{task}` / `{text}` 占位 |
| `embed_doc_prefix` | `EMBED_DOC_PREFIX` | 空 | 文档侧前缀（e5 的 `passage: ` 之类） |
| `embed_dim_check` | `EMBED_DIM_CHECK` | 1 | 首次嵌入探测服务端维度并比对（不一致会告警） |

现成配方：[`embedding-qwen3-8b.json`](embedding-qwen3-8b.json)（可当唯一 `--config` 直接用）。
每个 baseline 内部的尺子/维度也有自己的旋钮，默认跟随上面这组：
`HGR_EMBED_DIM` / `HGR_EMBED_MAX_TOKENS` / `HGR_TOKENIZER`、
`LIGHTRAG_EMBED_DIM` / `LIGHTRAG_EMBED_MAX_TOKENS` / `LIGHTRAG_TOKENIZER`、
（PRoH 已移除，不再有 `PROH_*` 旋钮。）
要恢复官方原版口径（官方按 tiktoken 调的 4000 截断等）就设 `chunk_tokenizer: "tiktoken"`
并把这些 `*_TOKENIZER` 设成 `tiktoken`。

⚠️ **改 `embed_dim`/`embed_max_tokens`/`embed_tokenizer`/`chunk_tokenizer` = 改向量口径，
必须重建所有向量**（开新 `--graph-version`）：新旧向量不在同一空间，混用只会静默给出垃圾结果。
`embed_tokens` 计费口径也跟着 embedder 的 tokenizer 走（换 Qwen 后同一段文本的 token 数会变），
前后的成本对比要注明这一点。

### 换 embedder 时"静默复用旧向量"是怎么被拦住的

换 embedder 最危险的不是报错，而是**不报错**：`bge`(1024 维) → `Qwen3-Embedding-8B` 截到
1024 维后，`.npy`/`faiss` 的形状完全一致，旧向量会被直接复用 —— 检索结果不可信，日志一片正常。
所以向量口径被写进了三处缓存判定：

1. `common.run_record.embed_fingerprint()` = 模型 + 维度 + 单条上限 + 两把尺子 + 查询格式的
   sha256 前 16 位；
2. `data/corpus/{ds}/v{n}/meta.json` 与 `methods/factalign.py` 的
   `{ent,window}_components.json` 都记这个指纹，**不一致就拒绝复用并报错**（不是警告）；
3. `extract.py` 的逐题抽取缓存 key 里也带上它（`basename|指纹`）。

外部 baseline 自己的索引目录（`data/{method}/{ds}/v{n}/`）**没有**这层判定：它们各自
按 marker 重建，不认 embedder 指纹。所以规矩很简单：

> **换 embedding 模型时，一律开新 `--graph-version`。** 不要指望同一个 `v` 号里的
> baseline 索引会自己发现 embedder 变了。

自检：换完先用 `python tools/check_embedder.py --config <新配置> --embed` 看配置与服务端是否一致
（退出码 1 = 维度不一致，2 = 服务端不可达），跑起来后再确认日志里有
`[embed] 服务端维度 N 与配置一致`。

## 文件

| 文件 | 用途 |
|---|---|
| `default.json` | 全局默认：后端/模型地址、切块口径、embedding 口径、重试与并发闸、run 级默认（workers/sample/version） |
| `baselines.json` | 12 个 baseline 的稳定旋钮（默认值 = 各官方仓库取值） |
| `official-alignment.json` | ★ 官方口径对齐（tiktoken 尺子 / 官方分块与截断 / 官方生成上限），用于跟论文数字对账 |
| `embedding-qwen3-8b.json` | 换 Qwen3-Embedding-8B 的完整配方（含 vLLM 启动命令与重建提示） |
| `smoke.json` | 10 题冒烟（小样本 + trace 全开） |

## 改了参数会不会重建图？

会。图/索引目录是 `data/{method}/{dataset}/v{n}/`，`v` 由 `--graph-version` 决定，
同一 `v` 内参数变化由各 adapter 的 marker 检测并重建。**要保留旧图就开新 `v`**：

```bash
python run.py --config configs/default.json --method proprag --dataset hotpotqa \
    --graph-version 2 --retrieval 1        # 新图
```

配方里的参数也会一起写进 `data/{method}/{dataset}/v{n}/build/config.json`，
所以事后能回答「这张图是拿哪份配置建的」。
