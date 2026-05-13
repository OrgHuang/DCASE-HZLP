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
# Configuration
# ---------------------------------------------------------------------------
CONFIG_FILE="${CONFIG_FILE:-configs/audio_mcq_qlora_sft_enhanced.yaml}"

yaml_config_value() {
  python -c 'import sys, yaml; data = yaml.safe_load(open(sys.argv[1], encoding="utf-8")) or {}; value = data.get(sys.argv[2]); print("" if value is None else value)' "$CONFIG_FILE" "$1"
}

LOG_DIR="${LOG_DIR:-$(yaml_config_value log_dir)}"
LOG_DIR="${LOG_DIR:-logs}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(yaml_config_value cuda_visible_devices)}"
if [ -n "${CUDA_VISIBLE_DEVICES}" ]; then
  export CUDA_VISIBLE_DEVICES
fi
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_NAME=$(basename "${CONFIG_FILE}" .yaml)
LOG_FILE="${LOG_DIR}/${EXPERIMENT_NAME}_${TIMESTAMP}_RANK${RANK:-0}.log"

# ---------------------------------------------------------------------------
# Optional environment-variable overrides
# ---------------------------------------------------------------------------
EXTRA_ARGS=("--config_file" "${CONFIG_FILE}")

append_value_override() {
  local env_name="$1"
  local arg_name="$2"
  local value="${!env_name:-}"

  if [ -n "${value}" ]; then
    EXTRA_ARGS+=("--${arg_name}" "${value}")
  fi
}

append_bool_override() {
  local env_name="$1"
  local arg_name="$2"
  local value="${!env_name:-}"

  if [ -z "${value}" ]; then
    return
  fi

  if [ "${value}" = "true" ]; then
    EXTRA_ARGS+=("--${arg_name}")
  elif [ "${value}" = "false" ]; then
    EXTRA_ARGS+=("--no-${arg_name}")
  else
    echo "Error: ${env_name} must be true or false, got: ${value}"
    exit 1
  fi
}

append_value_override MODEL_PATH model_name_or_path
append_value_override ATTN_IMPLEMENTATION attn_implementation
append_value_override DATA_PATH data_path
append_value_override OUTPUT_DIR output_dir
append_value_override MAX_SAMPLES max_samples
append_value_override MAX_AUDIO_SECONDS max_audio_seconds
append_value_override NUM_EPOCHS num_train_epochs
append_value_override BATCH_SIZE per_device_train_batch_size
append_value_override GRAD_ACC gradient_accumulation_steps
append_value_override LR learning_rate
append_value_override WARMUP_RATIO warmup_ratio
append_value_override LOGGING_STEPS logging_steps
append_value_override SAVE_STEPS save_steps
append_value_override SAVE_TOTAL_LIMIT save_total_limit
append_value_override SEED seed
append_value_override LORA_R lora_r
append_value_override LORA_ALPHA lora_alpha
append_value_override LORA_DROPOUT lora_dropout
append_value_override LORA_TARGET lora_target
append_value_override AUDIO_MARGIN_WEIGHT audio_margin_weight
append_value_override AUDIO_MARGIN audio_margin
append_value_override PERMUTATION_LOSS_WEIGHT permutation_loss_weight
append_value_override PERMUTATION_MARGIN permutation_margin
append_value_override RESUME_FROM_CHECKPOINT resume_from_checkpoint

append_bool_override BF16 bf16
append_bool_override FP16 fp16
append_bool_override GRADIENT_CHECKPOINTING gradient_checkpointing
append_bool_override USE_COUNTERFACTUAL use_counterfactual_audio
append_bool_override USE_SILENCE use_silence_view
append_bool_override USE_MISMATCH use_mismatch_view
append_bool_override USE_PERMUTED use_permuted_view
append_bool_override ALLOW_CPU allow_cpu

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
echo "========================================"
echo "Fun-Audio-Chat Enhanced AudioMCQ SFT"
echo "========================================"
echo "Config:     ${CONFIG_FILE}"
echo "Overrides:  ${EXTRA_ARGS[*]:2}"
echo "CUDA GPUs:  ${CUDA_VISIBLE_DEVICES:-all}"
echo "Log file:   ${LOG_FILE}"
echo "========================================"

python "${TRAINING_DIR}/train_audio_mcq_enhanced.py" \
  "${EXTRA_ARGS[@]}" 2>&1 | tee "${LOG_FILE}"

TRAINING_EXIT_CODE=${PIPESTATUS[0]}
if [ ${TRAINING_EXIT_CODE} -ne 0 ]; then
  echo "Training failed with exit code: ${TRAINING_EXIT_CODE}" | tee -a "${LOG_FILE}"
  echo "Log saved to: ${LOG_FILE}" | tee -a "${LOG_FILE}"
  exit ${TRAINING_EXIT_CODE}
fi

echo "Training completed successfully." | tee -a "${LOG_FILE}"
echo "Log saved to: ${LOG_FILE}" | tee -a "${LOG_FILE}"
