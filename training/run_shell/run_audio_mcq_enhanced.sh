#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${TRAINING_DIR}/.." && pwd)"
DATA_FILE="${TRAINING_DIR}/datasets/audio-mcq-strongac-gemini-cot/train.jsonl"

if [ ! -f "${DATA_FILE}" ]; then
  echo "Dataset not found at ${DATA_FILE}. Running preparation script..."
  bash "${SCRIPT_DIR}/prepare_audio_mcq_sft.sh"
fi

cd "${TRAINING_DIR}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# ---------------------------------------------------------------------------
# Default hyperparameters (matching the original fast config)
# ---------------------------------------------------------------------------
MODEL_PATH="${MODEL_PATH:-../pretrained_models/Fun-Audio-Chat-8B}"
OUTPUT_DIR="${OUTPUT_DIR:-saves/Fun-Audio-Chat-8B/audio_mcq_enhanced_sft}"
DATA_PATH="${DATA_PATH:-datasets/audio-mcq-strongac-gemini-cot/train.jsonl}"

# Training hyperparameters
NUM_EPOCHS="${NUM_EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACC="${GRAD_ACC:-4}"
LR="${LR:-2e-4}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
MAX_AUDIO_SECONDS="${MAX_AUDIO_SECONDS:-30}"
SEED="${SEED:-42}"

# Counterfactual / margin-loss switches
USE_COUNTERFACTUAL="${USE_COUNTERFACTUAL:-false}"
USE_SILENCE="${USE_SILENCE:-true}"
USE_MISMATCH="${USE_MISMATCH:-true}"
USE_PERMUTED="${USE_PERMUTED:-false}"
AUDIO_MARGIN_WEIGHT="${AUDIO_MARGIN_WEIGHT:-0.0}"
AUDIO_MARGIN="${AUDIO_MARGIN:-0.2}"
PERMUTATION_LOSS_WEIGHT="${PERMUTATION_LOSS_WEIGHT:-0.0}"
PERMUTATION_MARGIN="${PERMUTATION_MARGIN:-0.0}"

# Resume support
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

# ---------------------------------------------------------------------------
# Build command-line flags
# ---------------------------------------------------------------------------
EXTRA_ARGS=()

if [ "${USE_COUNTERFACTUAL}" = "true" ] || [ "${AUDIO_MARGIN_WEIGHT}" != "0.0" ]; then
  EXTRA_ARGS+=("--use_counterfactual_audio")
fi

if [ "${USE_SILENCE}" = "true" ]; then
  EXTRA_ARGS+=("--use_silence_view")
fi

if [ "${USE_MISMATCH}" = "true" ]; then
  EXTRA_ARGS+=("--use_mismatch_view")
fi

if [ "${USE_PERMUTED}" = "true" ] || [ "${PERMUTATION_LOSS_WEIGHT}" != "0.0" ]; then
  EXTRA_ARGS+=("--use_permuted_view")
fi

if [ -n "${RESUME_FROM_CHECKPOINT}" ]; then
  EXTRA_ARGS+=("--resume_from_checkpoint" "${RESUME_FROM_CHECKPOINT}")
fi

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
echo "========================================"
echo "Fun-Audio-Chat Enhanced AudioMCQ SFT"
echo "========================================"
echo "Model:      ${MODEL_PATH}"
echo "Data:       ${DATA_PATH}"
echo "Output:     ${OUTPUT_DIR}"
echo "Epochs:     ${NUM_EPOCHS}"
echo "Batch size: ${BATCH_SIZE} (grad_acc=${GRAD_ACC})"
echo "LR:         ${LR}"
echo "LoRA:       r=${LORA_R}, alpha=${LORA_ALPHA}"
echo "Margin:     weight=${AUDIO_MARGIN_WEIGHT}, margin=${AUDIO_MARGIN}"
echo "Permutation: weight=${PERMUTATION_LOSS_WEIGHT}, margin=${PERMUTATION_MARGIN}"
echo "========================================"

python "${TRAINING_DIR}/train_audio_mcq_enhanced.py" \
  --model_name_or_path "${MODEL_PATH}" \
  --data_path "${DATA_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --num_train_epochs "${NUM_EPOCHS}" \
  --per_device_train_batch_size "${BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRAD_ACC}" \
  --learning_rate "${LR}" \
  --lora_r "${LORA_R}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_dropout "${LORA_DROPOUT}" \
  --max_audio_seconds "${MAX_AUDIO_SECONDS}" \
  --save_steps 500 \
  --save_total_limit 2 \
  --logging_steps 10 \
  --seed "${SEED}" \
  --bf16 \
  --gradient_checkpointing \
  --audio_margin_weight "${AUDIO_MARGIN_WEIGHT}" \
  --audio_margin "${AUDIO_MARGIN}" \
  --permutation_loss_weight "${PERMUTATION_LOSS_WEIGHT}" \
  --permutation_margin "${PERMUTATION_MARGIN}" \
  "${EXTRA_ARGS[@]}"
