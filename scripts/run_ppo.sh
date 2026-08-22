#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
LLAMAFACTORY="${LLAMAFACTORY:-llamafactory-cli}"
DATA_DIR="${DATA_DIR:-/data/nfs_data/mllm_project/generated}"
SFT_ADAPTER="${SFT_ADAPTER:-outputs/sft_lora}"
RM_OUTPUT="${RM_OUTPUT:-outputs/reward_lora}"
PPO_OUTPUT="${PPO_OUTPUT:-outputs/sft_ppo_lora}"
GPU_IDS="${GPU_IDS:-0,1,2}"
MAX_PAIRS="${MAX_PAIRS:-5000}"
INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-2}"
REBUILD_PREFS="${REBUILD_PREFS:-0}"
REBUILD_PREDICTIONS="${REBUILD_PREDICTIONS:-0}"
REBUILD_PPO_PROMPTS="${REBUILD_PPO_PROMPTS:-0}"
FORCE_RM="${FORCE_RM:-0}"
INCLUDE_PRISTINE_ERRORS="${INCLUDE_PRISTINE_ERRORS:-0}"
PREF_RUN_NAME="${PREF_RUN_NAME:-sft_pref}"
PREF_DATASET_NAME="${PREF_DATASET_NAME:-dgm4_badcase_preference}"
PPO_DATASET_NAME="${PPO_DATASET_NAME:-dgm4_ppo_train}"
PREF_FILE="${DATA_DIR}/${PREF_DATASET_NAME}.jsonl"
PPO_FILE="${DATA_DIR}/${PPO_DATASET_NAME}.jsonl"
PREF_PREDICTIONS="${PREF_PREDICTIONS:-predictions/${PREF_RUN_NAME}_preference_pool.jsonl}"
RUNTIME_CONFIG_DIR="${RUNTIME_CONFIG_DIR:-logs/runtime_configs}"

mkdir -p "$RUNTIME_CONFIG_DIR" logs predictions results

if [[ "$REBUILD_PREFS" == "1" || ! -f "$PREF_FILE" ]]; then
  if [[ "$REBUILD_PREDICTIONS" == "1" || ! -f "$PREF_PREDICTIONS" ]]; then
    DATA_DIR="$DATA_DIR" \
    ADAPTER="$SFT_ADAPTER" \
    NAME="$PREF_RUN_NAME" \
    GPU_IDS="$GPU_IDS" \
    BATCH_SIZE="$INFER_BATCH_SIZE" \
    SPLITS="preference_pool" \
    SCORE_MODE="generated" \
    bash scripts/run_parallel_eval.sh
  fi

  build_cmd=(
    "$PYTHON_BIN" scripts/build_preference_pairs.py
    --pool "${DATA_DIR}/dgm4_preference_pool.jsonl"
    --predictions "$PREF_PREDICTIONS"
    --output "$PREF_FILE"
    --max-pairs "$MAX_PAIRS"
  )
  if [[ "$INCLUDE_PRISTINE_ERRORS" == "1" ]]; then
    build_cmd+=(--include-pristine-errors)
  fi
  "${build_cmd[@]}"
fi

if [[ "$REBUILD_PPO_PROMPTS" == "1" || ! -f "$PPO_FILE" ]]; then
  "$PYTHON_BIN" scripts/prepare_ppo_prompts.py \
    --source "${DATA_DIR}/dgm4_preference_pool.jsonl" \
    --output "$PPO_FILE" \
    --dataset-info "${DATA_DIR}/dataset_info.json" \
    --dataset-name "$PPO_DATASET_NAME"
fi

rm_config="${RUNTIME_CONFIG_DIR}/rm_$(date +%Y%m%d_%H%M%S).yaml"
"$PYTHON_BIN" scripts/render_llamafactory_config.py \
  --template configs/rm_lora.yaml \
  --output "$rm_config" \
  --set "dataset_dir=${DATA_DIR}" \
  --set "dataset=${PREF_DATASET_NAME}" \
  --set "adapter_name_or_path=${SFT_ADAPTER}" \
  --set "output_dir=${RM_OUTPUT}"

if [[ "$FORCE_RM" == "1" || ! -d "$RM_OUTPUT" ]]; then
  echo "RM config: ${rm_config}"
  CUDA_VISIBLE_DEVICES="$GPU_IDS" FORCE_TORCHRUN=1 "$LLAMAFACTORY" train "$rm_config"
else
  echo "Skip RM because ${RM_OUTPUT} already exists. Set FORCE_RM=1 to retrain."
fi

ppo_config="${RUNTIME_CONFIG_DIR}/ppo_$(date +%Y%m%d_%H%M%S).yaml"
"$PYTHON_BIN" scripts/render_llamafactory_config.py \
  --template configs/ppo_lora.yaml \
  --output "$ppo_config" \
  --set "dataset_dir=${DATA_DIR}" \
  --set "dataset=${PPO_DATASET_NAME}" \
  --set "adapter_name_or_path=${SFT_ADAPTER}" \
  --set "reward_model=${RM_OUTPUT}" \
  --set "output_dir=${PPO_OUTPUT}"

echo "PPO config: ${ppo_config}"
echo "PPO eval adapter after training: ${SFT_ADAPTER},${PPO_OUTPUT}"
CUDA_VISIBLE_DEVICES="$GPU_IDS" FORCE_TORCHRUN=1 "$LLAMAFACTORY" train "$ppo_config"
