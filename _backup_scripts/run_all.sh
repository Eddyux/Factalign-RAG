#!/bin/bash
# 跑全部「方法 × 数据集」，结果按 results/{method}_{dataset}.jsonl 分开记，断点续。
#
# 用法:
#   bash run_all.sh                                  # 全部 7 方法 × 4 数据集
#   METHODS="standardrag full" bash run_all.sh       # 只跑部分方法
#   DATASETS="hotpotqa" bash run_all.sh              # 只跑部分数据集
#   SKIP_METHODS="hypergraphrag graph_r1" bash run_all.sh   # 跳过未实现的 baseline
#   SAMPLE=100 WORKERS=4 bash run_all.sh             # 改样本数/并发
#   nohup bash run_all.sh &                          # 后台跑，日志在 logs/
#
# 说明:
#   - 每个「方法×数据集」单独一条命令，中途停了重跑同命令自动续（跳过已完成题）
#   - hypergraphrag / graph_r1 适配器未实现，会快速报"不可用"跳过，不影响其他方法
#   - full 和 no_head 共用同一张语料图（data/corpus/），第一个跑会建图，第二个直接复用
#   - 失败的运行会记进 logs/failures.txt

set -u

SAMPLE=${SAMPLE:-300}
WORKERS=${WORKERS:-8}
# 默认不含 graph_r1（需 RL 训练 checkpoint，官方未放出）；准备好后可显式指定:
#   METHODS="... graph_r1" bash run_all.sh
METHODS=${METHODS:-"standardrag no_head full lightrag graphrag hypergraphrag"}
DATASETS=${DATASETS:-"hotpotqa musique popqa 2wiki"}
SKIP_METHODS=${SKIP_METHODS:-}

mkdir -p logs

for method in $METHODS; do
  if [[ " $SKIP_METHODS " == *" $method "* ]]; then
    echo "========== 跳过 $method（SKIP_METHODS） =========="
    continue
  fi
  for dataset in $DATASETS; do
    echo "========== $method @ $dataset ($(date '+%H:%M:%S')) =========="
    python run.py --method "$method" --dataset "$dataset" \
      --sample "$SAMPLE" --workers "$WORKERS" 2>&1 | tee "logs/${method}_${dataset}.log"
    rc=${PIPESTATUS[0]}
    if [ "$rc" -ne 0 ]; then
      echo "[$method @ $dataset] 退出码 $rc" >> logs/failures.txt
    fi
  done
done

echo "========== 全部跑完。结果: results/ 日志: logs/ =========="
