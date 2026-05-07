#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${TRAINING_DIR}/.." && pwd)"

INPUT_JSONL="${INPUT_JSONL:-/home/org/DCASE/AudioMCQ-StrongAC-GeminiCoT-complete/data_acoustic_cot_no_teacher.jsonl}"
DATASET_ROOT="${DATASET_ROOT:-/home/org/DCASE/AudioMCQ-StrongAC-GeminiCoT-complete}"
OUTPUT_FILE="${OUTPUT_FILE:-${TRAINING_DIR}/datasets/audio-mcq-strongac-gemini-cot-text-first/train.jsonl}"
TARGET_MODE="${TARGET_MODE:-cot_answer}"
ANSWER_FORMAT="${ANSWER_FORMAT:-letter_and_text}"

cd "${PROJECT_ROOT}"
python training/process/convert_audio_mcq_sft.py \
  --input-jsonl "${INPUT_JSONL}" \
  --dataset-root "${DATASET_ROOT}" \
  --output-file "${OUTPUT_FILE}" \
  --target-mode "${TARGET_MODE}" \
  --answer-format "${ANSWER_FORMAT}" \
  --audio-position text_first \
  --shuffle
