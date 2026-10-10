#!/usr/bin/env bash
# ============================================================
# 清理所有「可再生的实验产物」，推倒重来。
# 保留：代码、data/samples/（下载的数据，别删，重下很慢）、baselines/
# 删除：results/、data/cache/、data/corpus/、外部 baseline 索引、logs/
# ============================================================
set -u

echo "== 清理实验产物（保留 data/samples/ 和代码）=="

# 结果（含 summary.jsonl）
if [ -d results ]; then
  rm -rf results/*
  echo "  已清空 results/"
fi

# 自带方法的图缓存（v7 图会自动重建）
if [ -d data/cache ]; then rm -rf data/cache; echo "  已删除 data/cache/"; fi
if [ -d data/corpus ]; then rm -rf data/corpus; echo "  已删除 data/corpus/"; fi

# 外部 baseline 的索引目录（data/hipporag_*、lightrag_*、graphrag_*、hypergraphrag_*、graph_r1_*）
for p in data/hipporag_* data/lightrag_* data/graphrag_* data/hypergraphrag_* data/graph_r1_*; do
  if [ -d "$p" ]; then rm -rf "$p"; echo "  已删除 $p"; fi
done

# 日志
if [ -d logs ]; then rm -rf logs; echo "  已删除 logs/"; fi

echo ""
echo "清理完成。保留的内容："
echo "  - data/samples/（数据样本）"
echo "  - 代码 + baselines/（第三方仓库）"
echo ""
echo "重新开始：先下数据 -> bash run_baselines.sh"
