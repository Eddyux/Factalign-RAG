"""构建评测集：随机抽取 N 条/数据集（默认 500），统一 item 格式。

产物：data/samples/{name}_{N}.json
      {"dataset", "sample_size", "random_seed", "built_at", "builder", "items": [...]}

item 格式（全项目统一，run.py / corpus.py 只认这个）：
    {"id", "question", "answer", "answer_aliases", "contexts", "supporting_facts"}
    contexts: [{"title", "text", "sentences"}]

数据集（`--datasets` 可选，默认全部 5 个）：
    hotpotqa  HF hotpotqa/hotpot_qa (distractor, validation)  每题 10 段（2 金 + 8 干扰）
    2wiki     HF ohjoonhee/2WikiMultihopQA (validation)       该源给金段落（无干扰）
    musique   HF bdsaglam/musique (answerable, validation)    2-4 跳支撑段 + 干扰
    popqa     HF Atipico1/popQA (test)                        **单跳**长尾实体，每题 10 段预检索段落
    triviaqa  HF mandarjoshi/trivia_qa (rc.wikipedia)         每题自带 wiki 页面

都是**公开数据集的原生 split**，脚本只做「固定 seed 随机抽样 + 统一 item 格式」，
不额外筛选/不改动题目与上下文；语料规模（段数/token/块数）在构建后打印，用于估算建图成本。

抽样逻辑与旧脚本逐字一致（`_backup_scripts/old_data_build_sample_datasets.py`）：
    rng = random.Random(seed); idxs = sorted(rng.sample(range(len(ds)), min(k, len(ds))))
→ 同一个 seed + 同一个数据集 = **每次抽到完全相同的题**（可复现）。

用法:
  python tools/build_eval_samples.py                              # 5 个数据集各 500
  python tools/build_eval_samples.py --sample 500 --datasets hotpotqa,2wiki,musique,popqa
  python tools/build_eval_samples.py --datasets triviaqa --sample 500
  python tools/build_eval_samples.py --seed 42 --force            # 覆盖已有文件
"""

import argparse
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from config import SAMPLES_DIR, RANDOM_SEED, SAMPLE_SIZE

BUILDER_VERSION = "build_eval_samples/2026-09"
DEFAULT_DATASETS = ("hotpotqa", "2wiki", "musique", "popqa", "triviaqa")
TRIVIAQA_SPLITS = os.environ.get("TRIVIAQA_SPLITS", "validation").split(",")


def _load_original_builders():
    """加载旧构建脚本（备份里那份 = 原样逻辑），复用它的 loader 与抽样函数。"""
    p = ROOT / "_backup_scripts" / "old_data_build_sample_datasets.py"
    if not p.exists():
        p = ROOT / "_backup_scripts" / "sample_datasets.py"
    spec = importlib.util.spec_from_file_location("_orig_sample_datasets", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ORIG = _load_original_builders()
_split_sentences = ORIG._split_sentences
_sample_indices = ORIG._sample_indices


# ── TriviaQA（官方 HF 数据，rc.wikipedia 配置） ──────────────────────────────
def _iter_triviaqa_docs(example):
    """产出该题自带的 (title, text)：entity_pages 优先，再补 search_results（去重）。

    字段名在 HF 上有两套历史命名，两套都兼容（与 S2G-RAG 的 `parse_page_collection` 同思路）：
      entity_pages:  {"wiki_context": [...], "title": [...]}
      search_results:{"search_context": [...], "title": [...]}
    """
    seen = set()
    for key, text_key in (("entity_pages", "wiki_context"),
                          ("search_results", "search_context")):
        pages = example.get(key) or {}
        titles = texts = []
        if isinstance(pages, dict):
            titles = pages.get("title") or []
            texts = pages.get(text_key) or pages.get("context") or []
        elif isinstance(pages, list):
            for it in pages:
                if isinstance(it, dict) and it.get("title"):
                    titles.append(it["title"])
                    texts.append(it.get(text_key) or it.get("text") or "")
        for t, x in zip(titles, texts):
            t, x = str(t or "").strip(), str(x or "").strip()
            if not t or not x or (t, x) in seen:
                continue
            seen.add((t, x))
            yield t, x


def load_triviaqa(spec, k, seed):
    """按 `rc.wikipedia` 抽 k 题，把每题自带的 wiki 页面转成 contexts。"""
    from datasets import load_dataset
    ds = load_dataset(spec["path"], spec["config"], split=spec["split"])
    items = []
    for i in _sample_indices(len(ds), k, seed):
        item = ds[int(i)]
        contexts = []
        for title, text in _iter_triviaqa_docs(item):
            contexts.append({"title": title, "text": text,
                             "sentences": _split_sentences(text)})
        ans = item.get("answer") or {}
        if isinstance(ans, dict):
            value = str(ans.get("normalized_value") or ans.get("value") or "").strip()
            aliases = [str(a).strip() for a in (ans.get("aliases") or []) if str(a).strip()]
        else:
            value, aliases = str(ans).strip(), []
        if value and value not in aliases:
            aliases = [value] + aliases
        items.append({"id": str(item.get("question_id", i)),
                      "question": str(item.get("question", "")).strip(),
                      "answer": value or (aliases[0] if aliases else ""),
                      "answer_aliases": aliases,
                      "contexts": contexts,
                      "supporting_facts": []})   # TriviaQA 无段落级标注
    return items


LOADERS = {
    "hotpotqa": ORIG.load_hotpotqa,
    "2wiki": ORIG.load_2wiki,
    "musique": ORIG.load_musique,
    "popqa": ORIG.load_popqa,
    "triviaqa": load_triviaqa,
}
SPECS = {
    "hotpotqa": {"path": "hotpotqa/hotpot_qa", "config": "distractor", "split": "validation"},
    "musique": {"path": "bdsaglam/musique", "config": "answerable", "split": "validation"},
    "2wiki": {"path": "ohjoonhee/2WikiMultihopQA", "config": None, "split": "validation"},
    "popqa": {"path": "Atipico1/popQA", "config": None, "split": "test"},
    "triviaqa": {"path": "mandarjoshi/trivia_qa", "config": "rc.wikipedia",
                 "split": TRIVIAQA_SPLITS[0]},
}


# ── 规模体检（构建时印出来，用于估算建图成本） ─────────────────────────────
def describe(name: str, items: list[dict]) -> dict:
    """题数 / 段落数 / 内容 token 估算（语料规模 → 建图耗时与成本）。"""
    n = len(items)
    if not n:
        return {"n": 0}
    ctx_counts = [len(it["contexts"]) for it in items]
    # 去重后的段落数（title+text 去重，与 corpus.merged_contexts 的合并口径接近）
    uniq = set()
    chars = 0
    for it in items:
        for c in it["contexts"]:
            key = (c.get("title", ""), c.get("text", ""))
            if key in uniq:
                continue
            uniq.add(key)
            chars += len(c.get("text", ""))
    return {
        "n": n,
        "avg_ctx_per_q": sum(ctx_counts) / n,
        "paras_uniq": len(uniq),
        "chars": chars,
        "tok_est": int(chars / 4),          # 粗估：4 字符 ≈ 1 token
        "blocks_est": int(chars / 4 / 360) + 1,   # 整段的块数粗估
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=500,
                    help=f"每个数据集抽多少题（默认 500；config.SAMPLE_SIZE={SAMPLE_SIZE}）")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--datasets", type=str, default="",
                    help=f"逗号分隔（默认 {','.join(DEFAULT_DATASETS)}）")
    ap.add_argument("--force", action="store_true", help="已有同名文件也覆盖（默认跳过并提示）")
    a = ap.parse_args()

    names = ([n.strip() for n in a.datasets.split(",") if n.strip()]
             if a.datasets else list(DEFAULT_DATASETS))
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"评测集构建：{len(names)} 个数据集 × {a.sample} 题，seed={a.seed}")
    print(f"输出目录：{SAMPLES_DIR}\n")

    summary = {}
    for name in names:
        if name not in LOADERS:
            print(f"[SKIP] {name}：不支持（可选 {list(LOADERS)}）")
            continue
        # 文件名带上 seed：换 seed 就是换一批题，**不能**覆盖已有样本
        # （默认 seed 保持 {name}_{N}.json 这个老命名，run.py/iter_questions 按它找文件）
        stem = f"{name}_{a.sample}" if a.seed == RANDOM_SEED else f"{name}_{a.sample}_seed{a.seed}"
        out = SAMPLES_DIR / f"{stem}.json"
        if out.exists() and not a.force:
            print(f"[SKIP] {name}：{out.name} 已存在（要覆盖加 --force）")
            continue
        print(f"[{name}] 下载 {SPECS[name]['path']}"
              f"{'/' + SPECS[name]['config'] if SPECS[name]['config'] else ''} "
              f"({SPECS[name]['split']}) …", flush=True)
        t0 = time.time()
        try:
            items = LOADERS[name](SPECS[name], a.sample, a.seed)
        except Exception as e:
            print(f"  [FAIL] {name}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        if not items:
            print(f"  [FAIL] {name}: 抽到 0 题")
            continue
        info = describe(name, items)
        payload = {"dataset": name, "sample_size": len(items), "random_seed": a.seed,
                   "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "builder": BUILDER_VERSION, "source": SPECS[name], "stats": info,
                   "items": items}
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        summary[name] = info
        print(f"  [OK] {len(items)} 题 -> {out.name}（{time.time() - t0:.0f}s）")
        print(f"       平均上下文 {info['avg_ctx_per_q']:.1f} 段 | 去重后 {info['paras_uniq']} 段 "
              f"| ≈{info['tok_est']:,} token | 切块粗估 {info['blocks_est']:,} 块")

    if summary:
        print("\n=== 汇总（建图前先看这张表：决定建图规模/耗时） ===")
        print(f"{'数据集':<10}{'题数':>6}{'段/题':>7}{'段落(去重)':>11}"
              f"{'≈token':>11}{'≈块数':>9}")
        for k, v in summary.items():
            print(f"{k:<10}{v['n']:>6}{v['avg_ctx_per_q']:>7.1f}{v['paras_uniq']:>11,}"
                  f"{v['tok_est']:>11,}{v['blocks_est']:>9,}")
        print("\n下一步：python run.py --method factalign --dataset <ds> --sample "
              f"{a.sample} --workers 16")


if __name__ == "__main__":
    main()
