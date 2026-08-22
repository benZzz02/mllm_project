#!/usr/bin/env bash
set -euo pipefail

# Parallel DGM4 inference + evaluation.
#
# Example:
#   DATA_DIR=/data/nfs_data/mllm_project/generated \
#   ADAPTER=outputs/sft_lora \
#   NAME=sft \
#   GPU_IDS=0,1,2 \
#   bash scripts/run_parallel_eval.sh

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${DATA_DIR:-data/generated}"
ADAPTER="${ADAPTER:-outputs/sft_lora}"
NAME="${NAME:-sft}"
MODEL="${MODEL:-Qwen/Qwen3-VL-2B-Instruct}"
GPU_IDS="${GPU_IDS:-0,1,2}"
SPLITS="${SPLITS:-val test}"
ATTN="${ATTN:-sdpa}"
DTYPE="${DTYPE:-auto}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-192}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SCORE_MODE="${SCORE_MODE:-full}"
BERT_TOKENIZER="${BERT_TOKENIZER:-bert-base-uncased}"
PREDICTION_DIR="${PREDICTION_DIR:-predictions}"
RESULT_DIR="${RESULT_DIR:-results}"
LOG_DIR="${LOG_DIR:-logs}"
SHARD_DIR="${SHARD_DIR:-.eval_shards}"
RESUME="${RESUME:-1}"
TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

export TRANSFORMERS_VERBOSITY
export PYTORCH_CUDA_ALLOC_CONF

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
NUM_SHARDS="${#GPUS[@]}"

if [[ "$NUM_SHARDS" -lt 1 ]]; then
  echo "GPU_IDS must contain at least one GPU id" >&2
  exit 1
fi

mkdir -p "$PREDICTION_DIR" "$RESULT_DIR" "$LOG_DIR" "$SHARD_DIR"

split_jsonl() {
  local dataset="$1"
  local split_dir="$2"
  local shard

  mkdir -p "$split_dir"
  rm -f "$split_dir"/shard_*.jsonl
  for ((shard = 0; shard < NUM_SHARDS; shard++)); do
    awk -v n="$NUM_SHARDS" -v s="$shard" '(NR - 1) % n == s {print}' "$dataset" > "$split_dir/shard_${shard}.jsonl"
  done
}

validate_prediction_count() {
  local dataset="$1"
  local predictions="$2"
  "$PYTHON_BIN" - "$dataset" "$predictions" <<'PY'
import json
import sys
from pathlib import Path

dataset = Path(sys.argv[1])
predictions = Path(sys.argv[2])

expected = sum(1 for line in dataset.open(encoding="utf-8") if line.strip())
ids = []
for line in predictions.open(encoding="utf-8"):
    if line.strip():
        ids.append(json.loads(line)["id"])

unique = len(set(ids))
if len(ids) != expected or unique != expected:
    raise SystemExit(
        f"Prediction count/id check failed: expected={expected}, rows={len(ids)}, unique_ids={unique}"
    )
print(f"Prediction count/id check ok: {expected}")
PY
}

run_split() {
  local split="$1"
  local dataset="${DATA_DIR}/dgm4_${split}.jsonl"
  local split_shard_dir="${SHARD_DIR}/${NAME}_${split}"
  local pred_shard_dir="${PREDICTION_DIR}/shards/${NAME}_${split}"
  local merged_predictions="${PREDICTION_DIR}/${NAME}_${split}.jsonl"
  local metrics_output="${RESULT_DIR}/${NAME}_${split}_metrics.json"
  local badcases_output="${RESULT_DIR}/${NAME}_${split}_badcases.jsonl"
  local shard gpu log_file pid failed
  local pids=()

  if [[ ! -f "$dataset" ]]; then
    echo "Missing dataset: $dataset" >&2
    exit 1
  fi

  mkdir -p "$pred_shard_dir"
  split_jsonl "$dataset" "$split_shard_dir"

  echo "===== ${NAME} ${split}: launching ${NUM_SHARDS} shards on GPUs ${GPU_IDS} ====="
  for ((shard = 0; shard < NUM_SHARDS; shard++)); do
    gpu="${GPUS[$shard]}"
    log_file="${LOG_DIR}/${NAME}_${split}_gpu${gpu}_shard${shard}.log"

    (
      cmd=(
        "$PYTHON_BIN" scripts/infer_qwen3vl_with_scores.py
        --dataset "${split_shard_dir}/shard_${shard}.jsonl"
        --output "${pred_shard_dir}/shard_${shard}.jsonl"
        --model "$MODEL"
        --device-map "$DEVICE_MAP"
        --dtype "$DTYPE"
        --max-new-tokens "$MAX_NEW_TOKENS"
        --batch-size "$BATCH_SIZE"
        --score-mode "$SCORE_MODE"
      )
      if [[ -n "$ADAPTER" ]]; then
        cmd+=(--adapter "$ADAPTER")
      fi
      if [[ -n "$ATTN" ]]; then
        cmd+=(--attn-implementation "$ATTN")
      fi
      if [[ "$RESUME" == "1" ]]; then
        cmd+=(--resume)
      fi
      CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}"
    ) > "$log_file" 2>&1 &
    pid="$!"
    pids+=("$pid")
    echo "Shard ${shard} -> GPU ${gpu}, pid ${pid}, log ${log_file}"
  done

  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [[ "$failed" -ne 0 ]]; then
    echo "At least one ${split} shard failed. Check ${LOG_DIR}/${NAME}_${split}_gpu*_shard*.log" >&2
    exit 1
  fi

  : > "$merged_predictions"
  for ((shard = 0; shard < NUM_SHARDS; shard++)); do
    cat "${pred_shard_dir}/shard_${shard}.jsonl" >> "$merged_predictions"
  done
  validate_prediction_count "$dataset" "$merged_predictions"

  echo "===== ${NAME} ${split}: evaluating merged predictions ====="
  eval_cmd=(
    "$PYTHON_BIN" scripts/evaluate_dgm4_predictions.py
    --ground-truth "$dataset"
    --predictions "$merged_predictions"
    --output "$metrics_output"
    --badcases-output "$badcases_output"
  )
  if [[ -n "$BERT_TOKENIZER" ]]; then
    eval_cmd+=(--bert-tokenizer "$BERT_TOKENIZER")
  fi
  "${eval_cmd[@]}" > "${LOG_DIR}/${NAME}_${split}_evaluate.log" 2>&1

  "$PYTHON_BIN" - "$metrics_output" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
metrics = json.loads(path.read_text(encoding="utf-8"))
print(f"Saved metrics: {path}")
print(json.dumps(metrics["official_metrics_percent"], indent=2, ensure_ascii=False))
PY
}

echo "Parallel eval config:"
echo "  DATA_DIR=${DATA_DIR}"
echo "  ADAPTER=${ADAPTER:-<base model>}"
echo "  NAME=${NAME}"
echo "  MODEL=${MODEL}"
echo "  GPU_IDS=${GPU_IDS}"
echo "  SPLITS=${SPLITS}"
echo "  BATCH_SIZE=${BATCH_SIZE}"
echo "  SCORE_MODE=${SCORE_MODE}"
echo "  RESUME=${RESUME}"

for split in $SPLITS; do
  run_split "$split"
done

echo "All requested splits finished."
