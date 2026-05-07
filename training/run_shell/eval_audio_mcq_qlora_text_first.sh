#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${TRAINING_DIR}/.." && pwd)"
EVAL_SCRIPT="${PROJECT_ROOT}/evaluate/DCASE2026/eval_task5_with_lora.py"

DEFAULT_ADAPTER_DIR="${TRAINING_DIR}/saves/Fun-Audio-Chat-8B/audio_mcq_qlora_sft_text_first"
ADAPTER_PATH="${ADAPTER_PATH:-${DEFAULT_ADAPTER_DIR}}"
DATASET_ROOT="${DATASET_ROOT:-/home/org/DCASE/Harland/DCASE2026-Task5-DevSet}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/dcase2026_task5_eval_text_first}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [ ! -f "${EVAL_SCRIPT}" ]; then
  echo "Missing eval script: ${EVAL_SCRIPT}"
  exit 1
fi

if [ ! -d "${ADAPTER_PATH}" ]; then
  echo "Missing adapter directory: ${ADAPTER_PATH}"
  echo "Train the text-first SFT adapter first, or override ADAPTER_PATH."
  exit 1
fi

if [ ! -d "${DATASET_ROOT}" ]; then
  echo "Missing dataset root: ${DATASET_ROOT}"
  exit 1
fi

RESOLVED_ADAPTER_PATH="${ADAPTER_PATH}"
if [ ! -f "${RESOLVED_ADAPTER_PATH}/adapter_config.json" ]; then
  LATEST_CHECKPOINT="$(find "${ADAPTER_PATH}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1)"
  if [ -n "${LATEST_CHECKPOINT}" ] && [ -f "${LATEST_CHECKPOINT}/adapter_config.json" ]; then
    RESOLVED_ADAPTER_PATH="${LATEST_CHECKPOINT}"
  else
    echo "Could not find adapter_config.json in ${ADAPTER_PATH} or its checkpoint-* subdirectories."
    exit 1
  fi
fi

cd "${PROJECT_ROOT}"

CMD=(
  "${PYTHON_BIN}" "${EVAL_SCRIPT}"
  "--adapter-path" "${RESOLVED_ADAPTER_PATH}"
  "--dataset-root" "${DATASET_ROOT}"
  "--output-dir" "${OUTPUT_DIR}"
)

if [ -n "${DATASET_FILE:-}" ]; then
  CMD+=("--dataset-file" "${DATASET_FILE}")
fi

if [ -n "${BASE_MODEL_PATH:-}" ]; then
  CMD+=("--base-model-path" "${BASE_MODEL_PATH}")
fi

if [ -n "${LOG_FILE:-}" ]; then
  CMD+=("--log-file" "${LOG_FILE}")
fi

if [ -n "${LOG_EVERY:-}" ]; then
  CMD+=("--log-every" "${LOG_EVERY}")
fi

if [ -n "${MAX_SAMPLES:-}" ]; then
  CMD+=("--max-samples" "${MAX_SAMPLES}")
fi

if [ -n "${MAX_NEW_TOKENS:-}" ]; then
  CMD+=("--max-new-tokens" "${MAX_NEW_TOKENS}")
fi

if [ -n "${DEVICE:-}" ]; then
  CMD+=("--device" "${DEVICE}")
fi

if [ "${LOAD_IN_4BIT:-1}" = "0" ]; then
  CMD+=("--no-load-in-4bit")
fi

if [ "${SHORT_ANSWER_HINT:-0}" = "1" ]; then
  CMD+=("--short-answer-hint")
fi

echo "Evaluating adapter: ${RESOLVED_ADAPTER_PATH}"
echo "Dataset root: ${DATASET_ROOT}"
echo "Output dir: ${OUTPUT_DIR}"

"${CMD[@]}" "$@"
