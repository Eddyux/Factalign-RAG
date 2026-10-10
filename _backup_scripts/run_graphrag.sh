#!/usr/bin/env bash
# ============================================================
# 单独跑 graphrag（最重的 baseline：实体/关系抽取 + 社区摘要 + 向量索引）
# 4 数据集 × 300 题，每条独立命令 + 独立 log，失败不中断后续。
#
# ⚠️ graphrag 建索引很慢、很吃内存（300 题语料可能几小时），建议挂后台：
#     nohup bash run_graphrag.sh > graphrag_run.log 2>&1 &
#     然后 tail -f graphrag_run.log 看进度
#
# 每个 run.py 跑完自动写：
#   results/graphrag_{dataset}.jsonl   （每题一行，断点续）
#   results/summary.jsonl              （EM/F1 + build/query token + 耗时，追加）
# ============================================================
set -u
set -o pipefail

SAMPLE=300
WORKERS=8
DATASETS=(hotpotqa musique 2wiki popqa)

LOGDIR=logs
mkdir -p "$LOGDIR"
FAILED=()

echo "== 依赖预检 =="
if python -c "import graphrag" >/dev/null 2>&1; then
  echo "  [OK] graphrag 已装"
else
  echo "  [缺] 先 pip install 'graphrag>=3.0'（本适配器只支持 3.x）"
fi

echo "== 数据预检（缺 300 题样本的数据集会跳过）=="
for DS in "${DATASETS[@]}"; do
  if ls "data/samples/${DS}_${SAMPLE}.json" >/dev/null 2>&1; then
    echo "  [OK] ${DS}_${SAMPLE}.json"
  else
    echo "  [缺] ${DS}_${SAMPLE}.json -> python sample_datasets.py --datasets ${DS} --sample ${SAMPLE}"
  fi
done
echo ""

for DS in "${DATASETS[@]}"; do
  if ! ls "data/samples/${DS}_${SAMPLE}.json" >/dev/null 2>&1; then
    echo "[跳过] graphrag @ ${DS}：数据缺失，先下数据"
    continue
  fi
  LOG="$LOGDIR/graphrag_${DS}.log"
  echo "=============================================================="
  echo "[$(date '+%F %T')] 开始 graphrag @ ${DS} (n=${SAMPLE})"
  echo "  日志: ${LOG}"
  echo "=============================================================="
  if python run.py --method graphrag --dataset "$DS" --sample "$SAMPLE" --workers "$WORKERS" 2>&1 | tee "$LOG"; then
    echo "[完成] graphrag @ ${DS}"
  else
    echo "[失败] graphrag @ ${DS}（详见 ${LOG}），继续下一个"
    FAILED+=("graphrag @ ${DS}")
  fi
  echo ""
done

echo "=============================================================="
echo "graphrag 全部跑完。"
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "失败项："
  for f in "${FAILED[@]}"; do echo "  - $f"; done
else
  echo "无失败项。"
fi
echo ""
echo "token/耗时汇总: results/summary.jsonl"
echo "出对比表:       python report_tokens.py"
