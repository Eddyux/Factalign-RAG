"""读取统一格式的数据集样本。

统一 item 格式:
{
  "id": str, "question": str, "answer": str, "answer_aliases": [str],
  "contexts": [{"title": str, "text": str, "sentences": [str]}],
  "supporting_facts": [[title, sent_idx], ...],
}
"""

import json

from config import SAMPLES_DIR


def sample_path(dataset: str, n: int = 0):
    """定位样本文件：data/samples/{dataset}_{n}.json；取不到时退回最大样本数的那份。

    ★ 单独抽成公开函数：**读数据**与**记录"读的是哪份数据"必须走同一条解析逻辑**。
    以前 run.py 把样本路径硬编码成 `{dataset}_{sample}.json` 写进 trace 头，
    `--sample 0` 或文件名不严格等于样本数时就会记错文件（等于自己给自己造假凭证）。
    """
    if n > 0:
        path = SAMPLES_DIR / f"{dataset}_{n}.json"
        if path.exists():
            return path
    # n<=0，或 n>0 但没有同名文件 → 退回"最大样本数"口径（与旧行为一致）。
    # 排除 {dataset}_{n}.meta.json 这类旁挂文件，否则排序取末位会拿到没有 "items" 的元信息。
    candidates = sorted(p for p in SAMPLES_DIR.glob(f"{dataset}_*.json")
                        if not p.name.endswith(".meta.json"))
    if not candidates:
        raise FileNotFoundError(f"找不到 {dataset}_*.json in {SAMPLES_DIR}")
    # 注意：这里**不因为请求的 n 与文件尺寸不符而报错**（旧行为也不报）；真实用的是哪份、
    # 里面多少题，由 sample_provenance() 记进 manifest，事后一眼能看出来。
    return candidates[-1]


def load_samples(dataset: str, n: int = 0) -> list[dict]:
    """读 data/samples/{dataset}_{n}.json；n<=0 时自动找最大样本数的文件。"""
    path = sample_path(dataset, n)
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return data["items"]


def iter_questions(dataset: str, n: int = 0):
    for item in load_samples(dataset, n):
        yield item
