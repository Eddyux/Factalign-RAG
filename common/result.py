"""版本化的图/结果目录结构。

图版本 v：第几次建图（--graph-version，默认 1）。换图参数（模型/切块）时
手动开新 v 保留旧图；同一 v 内参数变化由各方法自己的 marker 检测并重建。
检索次数 r：同一张图上的第几次验证检索（--retrieval，默认 1）。换查询参数
（prompt / native-generator / rerank 等）时开新 r；同 v+r 重跑断点续。

  data/{method}/{dataset}/v{n}/              图/索引目录（自带方法共享图在 data/corpus/）
  results/{method}/{dataset}/v{n}r{m}.jsonl  逐题检索结果（每次检索一个文件，互不覆盖）
  results/summary.jsonl                      汇总（每行带 graph_version / retrieval）
"""

from config import DATA_DIR, RESULTS_DIR


def graph_dir(method: str, dataset: str, graph_version: int):
    """图/索引目录：data/{method}/{dataset}/v{n}/。"""
    return DATA_DIR / method / dataset / f"v{graph_version}"


def result_path(method: str, dataset: str, graph_version: int, retrieval: int):
    """检索结果文件：results/{method}/{dataset}/v{n}r{m}.jsonl。"""
    return RESULTS_DIR / method / dataset / f"v{graph_version}r{retrieval}.jsonl"
