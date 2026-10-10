# FactAlign-RAG

以「自包含命题（事实句）」为最小检索单元，并把下一跳查询写成**同形态的开放槽命题**
（`"Cleopatra IV's father is ___"`）驱动迭代检索 —— 让第 t+1 跳检索退化为对单条事实的
近复述匹配。方法定稿与实验矩阵见 `METHOD.md` / `EXPERIMENT_PLAN.md`。

（旧名 HyperfactRAG 已退役：当年的 `full` / `no_head` 自带语料图路线与
`methods/hyperfactrag.py` 一并删除，主线改为本方法 + `test_layer.py` 实验台。）

## 目录结构

```
├── common/          # 数据读取 / 指标 / LLM 封装（含分阶段日志）
├── methods/         # 方法实现与各 baseline 适配器
│   ├── factalign.py         # 本方法：命题索引 + 开放槽命题查询（run.py --method factalign）
│   ├── standard_rag.py      # standardrag（纯稠密分块检索）
│   ├── ircot_adapter.py     # IRCoT
│   ├── lightrag_adapter.py / hypergraphrag_adapter.py / hipporag_adapter.py
│   ├── s2grag_adapter.py
│   └── __init__.py          # 注册表（懒加载）
├── baselines/       # 原版 baseline 仓库（git clone，不提交）
├── prompts/         # 生成 prompt
├── data/            # samples（题集）/ corpus（语料图，缓存）
├── results/         # results/{method}/{dataset}/v{n}r{m}.jsonl + summary.jsonl
├── logs/            # logs/{method}/…（建图段 / 检索段分开）
├── test_layer.py    # 本方法实验台（消融全在这里跑）
├── run.py           # 主入口：方法 × 数据集 评测（断点续）
├── corpus.py / extract.py / search.py    # 语料建图 / 命题抽取 / 检索
└── tools/           # 对比与审计脚本
```

## 数据集

| 数据集 | 来源 |
|---|---|
| HotpotQA | HF 自带上下文 |
| 2WikiMultiHopQA | HF 自带上下文 |
| MuSiQue | HF 自带上下文 |

**最终指标：EM / F1（答案正确性）。** R@k 仅作开发诊断，最终分析不报。

## 方法

- `factalign`（**本方法**）：命题（自包含事实句）索引 + 开放槽命题查询；语料级 build 一次 + 逐题 query
- `standardrag`：分块（滑动窗口）+ **纯稠密**检索 top-5（无 BM25、无 RRF）（地基基线）
- `ircot`：检索-推理交替（官方 prompt）
- `lightrag` / `hypergraphrag` / `hipporag` / `s2grag`：外部 baseline 适配器

## 用法

```bash
python sample_datasets.py --sample 500                        # 下载数据
python run.py --method factalign --dataset 2wiki --sample 500 --workers 16   # 本方法
python run.py --method standardrag --dataset hotpotqa --sample 500 --workers 8
python test_layer.py --dataset 2wiki --sample 500 --limit 500 --workers 16 \
  --decompose --decompose_stmt --rerank hybrid --tag q_stmt_he   # 实验台 / 消融
```

- `--graph-version`（v）= 第几次建图，`--retrieval`（r）= 同一张图上的第几次检索；
  换参数开新号，结果互不覆盖，同号重跑自动断点续。
- 日志按阶段分两份：`logs/{method}/{ds}_{n}_v{n}r{m}_build_*.log`（建图段）与
  `..._query_*.log`（检索段到最终汇总），两段拼起来就是完整输出。
- 消融（查询形态 / 索引单元 / 拆解步数 / 精排 / 图传播 / 证据过滤）统一走 `test_layer.py`。

## 关键约定

- 生成口径统一：所有方法最后都调 `methods.base.generate_answer`；prompt 按**数据集**分派
  （hotpotqa / musique / 2wiki → `prompts/rag_prompt.txt`；triviaqa → `prompts/rag_prompt_triviaqa.txt`），
  无证据路径（`zeroshot`）用 `prompts/rag_prompt_nocontext.txt`，**不按数据集分派**
  （无证据时没有"抄哪个 span"的问题，单跳/多跳口径对它不存在）；
- EM / F1 全项目只此一份（`common/metrics.py`）；
- baseline 原版仓库放 `baselines/`，适配器在 `methods/*_adapter.py`；
- 语料切块：**默认不切块**（1 段上下文 = 1 个整段 passage），只有超过 2000 token 的块才切
  （`config.EXTRACT_CHUNK_TOKENS=0` → `normalize_chunk_tokens()` 映射为 2000，重叠 0）。
