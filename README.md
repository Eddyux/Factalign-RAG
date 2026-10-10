# FactAlign-RAG

**English** | [中文](README_ch.md)

A multi-hop retrieval-augmented generation (RAG) benchmark harness whose smallest
retrieval unit is a **self-contained proposition** — an atomic fact stated as a
complete sentence. Each follow-up query is issued in the same canonical form, an
**open-slot proposition** (`"Cleopatra IV's father is ___"`), which drives the
iterative retrieval loop and reduces hop *t+1* to a near-paraphrase match against a
single fact.

> The former name **HyperfactRAG** is retired. The earlier self-built corpus-graph
> route (`full` / `no_head`, together with `methods/hyperfactrag.py`) has been
> removed; the main line is now this method plus the `test_layer.py` experiment bench.

## Directory layout

```
├── common/          # data loading / metrics / LLM wrappers (with phased logging)
├── methods/         # method implementations and per-baseline adapters
│   ├── factalign.py         # this method: proposition index + open-slot queries
│   ├── standard_rag.py      # standardrag (pure dense chunk retrieval)
│   ├── ircot_adapter.py     # IRCoT
│   ├── lightrag_adapter.py / hypergraphrag_adapter.py / hipporag_adapter.py
│   ├── s2grag_adapter.py
│   └── __init__.py          # lazy method registry
├── baselines/       # upstream baseline repositories (git clone, not committed)
├── prompts/         # generation prompts
├── data/            # samples (question sets) / corpus (corpus graphs, cached)
├── results/         # results/{method}/{dataset}/v{n}r{m}.jsonl + summary.jsonl
├── logs/            # logs/{method}/… (build phase and query phase split)
├── test_layer.py    # experiment bench for this method (all ablations live here)
├── run.py           # main entry: evaluate method × dataset (resumable)
├── corpus.py / extract.py / search.py    # corpus build / proposition extraction / retrieval
└── tools/           # comparison and audit scripts
```

## Datasets

| Dataset | Source |
|---|---|
| HotpotQA | contexts shipped with the HF dataset |
| 2WikiMultiHopQA | contexts shipped with the HF dataset |
| MuSiQue | contexts shipped with the HF dataset |

**Final metric: EM / F1 (answer correctness).** R@k is used only for development
diagnostics and is not reported in the final analysis.

## Methods

- `factalign` (**this method**): proposition (self-contained fact sentence) index +
  open-slot proposition queries; one corpus-level build, then a per-question query
- `standardrag`: chunking (sliding window) + **pure dense** retrieval of top-5
  (no BM25, no RRF) — the floor baseline
- `ircot`: alternating retrieval and reasoning (official prompts)
- `lightrag` / `hypergraphrag` / `hipporag` / `s2grag`: external baseline adapters

## Usage

```bash
# build the corpus and the sampled question sets
python tools/build_all.py --datasets hotpotqa --sample 500 \
    --steps raw,samples,corpus

# this method
python run.py --method factalign --dataset 2wiki --sample 500 --workers 16

# floor baseline
python run.py --method standardrag --dataset hotpotqa --sample 500 --workers 8

# experiment bench / ablations
python test_layer.py --dataset 2wiki --sample 500 --limit 500 --workers 16 \
  --decompose --decompose_stmt --rerank hybrid --tag q_stmt_he
```

- `--graph-version` (`v`) is which corpus build, `--retrieval` (`r`) is which query
  pass over that same build. Bumping a number starts a fresh set of results so runs
  never overwrite each other; re-running the same number resumes automatically.
- Logs are split by phase: `logs/{method}/{ds}_{n}_v{n}r{m}_build_*.log` (corpus
  build) and `..._query_*.log` (retrieval through the final summary). Concatenating
  the two gives the complete output.
- Every ablation (query form / retrieval unit / decomposition steps / reranking /
  graph propagation / evidence filtering) is run through `test_layer.py`.

## Conventions

- **Unified generation path**: every method finally calls
  `methods.base.generate_answer`. The prompt is dispatched per **dataset**
  (`prompts/rag_prompt.txt` for hotpotqa / musique / 2wiki;
  `prompts/rag_prompt_triviaqa.txt` for triviaqa). The no-evidence path
  (`zeroshot`) uses `prompts/rag_prompt_nocontext.txt` and is **not** dispatched by
  dataset — with no evidence there is no "which span to copy" problem, so the
  single-hop / multi-hop distinction does not apply to it.
- **One implementation of EM / F1** for the whole project: `common/metrics.py`.
- Upstream baseline repositories live in `baselines/`; the adapters are in
  `methods/*_adapter.py`.
- **Chunking: none by default** (one context = one whole passage). Only blocks above
  2000 tokens are split (`config.EXTRACT_CHUNK_TOKENS=0` is mapped to 2000 by
  `normalize_chunk_tokens()`, with zero overlap).