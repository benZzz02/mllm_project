#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

ACCELERATE="${ACCELERATE:-accelerate}"
DATA_DIR="${DATA_DIR:-/data/nfs_data/mllm_project/generated}"
MODEL="${MODEL:-Qwen/Qwen3-VL-2B-Instruct}"
SFT_ADAPTER="${SFT_ADAPTER:-outputs/sft_lora}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/sft_grpo_lora}"
GPU_IDS="${GPU_IDS:-0,1,2}"
NUM_GENERATIONS="${NUM_GENERATIONS:-2}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACC="${GRAD_ACC:-8}"
LR="${LR:-5e-6}"
EPOCHS="${EPOCHS:-1.0}"
MAX_STEPS="${MAX_STEPS:--1}"
MAX_SAMPLES="${MAX_SAMPLES:-2000}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-192}"
DTYPE="${DTYPE:-auto}"
ATTN="${ATTN:-sdpa}"
BETA="${BETA:-0.01}"
USE_VLLM="${USE_VLLM:-0}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.35}"
REPORT_TO="${REPORT_TO:-tensorboard}"

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
NUM_PROCESSES="${#GPUS[@]}"
if [[ "$NUM_PROCESSES" -lt 1 ]]; then
  echo "GPU_IDS must contain at least one GPU id" >&2
  exit 1
fi

cmd=(
  "$ACCELERATE" launch
  --num_processes "$NUM_PROCESSES"
  scripts/train_grpo.py
  --dataset "${DATA_DIR}/dgm4_preference_pool.jsonl"
  --weights-from "${DATA_DIR}/dgm4_sft_train.jsonl"
  --model "$MODEL"
  --adapter "$SFT_ADAPTER"
  --output-dir "$OUTPUT_DIR"
  --num-generations "$NUM_GENERATIONS"
  --per-device-train-batch-size "$BATCH_SIZE"
  --gradient-accumulation-steps "$GRAD_ACC"
  --learning-rate "$LR"
  --num-train-epochs "$EPOCHS"
  --max-steps "$MAX_STEPS"
  --max-prompt-length "$MAX_PROMPT_LENGTH"
  --max-completion-length "$MAX_COMPLETION_LENGTH"
  --dtype "$DTYPE"
  --attn-implementation "$ATTN"
  --beta "$BETA"
  --report-to "$REPORT_TO"
)

if [[ "$MAX_SAMPLES" != "0" && "$MAX_SAMPLES" != "-1" ]]; then
  cmd+=(--max-samples "$MAX_SAMPLES")
fi
if [[ "$USE_VLLM" == "1" ]]; then
  cmd+=(--use-vllm --vllm-gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION")
fi

echo "GRPO output: ${OUTPUT_DIR}"
echo "GRPO default uses MAX_SAMPLES=${MAX_SAMPLES}; set MAX_SAMPLES=0 to use the full preference pool."
CUDA_VISIBLE_DEVICES="$GPU_IDS" "${cmd[@]}"
