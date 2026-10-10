#!/usr/bin/env bash
# ============================================================
# 批量评测外部 baseline：hipporag / standardrag / lightrag × 4 数据集 × 300 题
#
# 每个 `python run.py` 调用结束时会自动做两件事（无需额外保存）：
#   1. 写每题结果   -> results/{method}_{dataset}.jsonl（断点续，重跑跳过已完成题）
#   2. 追加汇总行   -> results/summary.jsonl（EM/F1 + build/query 分桶 token + 耗时）
#   （token 消耗由 report_tokens.py 读 summary.jsonl 出对比表）
#
# 每个「方法 × 数据集」单独一条命令、单独一个 log；某个失败不中断后续。
# ============================================================
set -u
set -o pipefail   # 让 python 的退出码穿透 | tee，失败能被抓到

SAMPLE=300
WORKERS=8
DATASETS=(hotpotqa musique 2wiki popqa)
METHODS=(hipporag standardrag lightrag)

LOGDIR=logs
mkdir -p "$LOGDIR"
FAILED=()

# 终端只显示这些进度/结果行（完整输出仍全部写进 LOG，tee 存全量 + grep 过滤终端）
PROGRESS='extracting stage|merging stage|processing file|em=|f1=|llm\(|建图|检索|traceback|error|失败|不可用|抽取 '

echo "== 数据预检（缺 300 题样本的数据集会跳过）=="
for DS in "${DATASETS[@]}"; do
  if ls "data/samples/${DS}_${SAMPLE}.json" >/dev/null 2>&1; then
    echo "  [OK] ${DS}_${SAMPLE}.json"
  else
    echo "  [缺] ${DS}_${SAMPLE}.json  -> 先跑: python sample_datasets.py --datasets ${DS} --sample ${SAMPLE}"
  fi
done
echo ""

for METHOD in "${METHODS[@]}"; do
  for DS in "${DATASETS[@]}"; do
    if ! ls "data/samples/${DS}_${SAMPLE}.json" >/dev/null 2>&1; then
      echo "[跳过] ${METHOD} @ ${DS}：数据缺失，先下数据"
      continue
    fi
    LOG="$LOGDIR/${METHOD}_${DS}.log"
    echo "=============================================================="
    echo "[$(date '+%F %T')] 开始 ${METHOD} @ ${DS} (n=${SAMPLE})"
    echo "  日志: ${LOG}"
    echo "=============================================================="
    python run.py --method "$METHOD" --dataset "$DS" --sample "$SAMPLE" --workers "$WORKERS" 2>&1 \
      | tee "$LOG" \
      | grep --line-buffered -iE "$PROGRESS"
    status=${PIPESTATUS[0]}
    if [ "$status" -eq 0 ]; then
      echo "[完成] ${METHOD} @ ${DS}"
    else
      echo "[失败] ${METHOD} @ ${DS}（详见 ${LOG}），继续下一个"
      FAILED+=("${METHOD} @ ${DS}")
    fi
    echo ""
  done
done

echo "=============================================================="
echo "全部跑完。"
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "失败项："
  for f in "${FAILED[@]}"; do echo "  - $f"; done
else
  echo "无失败项。"
fi
echo ""
echo "token/耗时汇总: results/summary.jsonl"
echo "出对比表:       python report_tokens.py"
