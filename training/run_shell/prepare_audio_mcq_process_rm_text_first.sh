#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${TRAINING_DIR}/.." && pwd)"

INPUT_JSONL="${INPUT_JSONL:-/home/org/DCASE/AudioMCQ-StrongAC-GeminiCoT-complete/data_acoustic_cot_no_teacher.jsonl}"
DATASET_ROOT="${DATASET_ROOT:-/home/org/DCASE/AudioMCQ-StrongAC-GeminiCoT-complete}"
OUTPUT_FILE="${OUTPUT_FILE:-${TRAINING_DIR}/datasets/audio-mcq-strongac-process-rm-text-first/train.jsonl}"
REASONING_FIELD="${REASONING_FIELD:-gemini_cot}"
FALLBACK_REASONING_FIELD="${FALLBACK_REASONING_FIELD:-}"
ANSWER_FORMAT="${ANSWER_FORMAT:-letter_and_text}"
AUDIO_POSITION="${AUDIO_POSITION:-text_first}"
PAIRS_PER_SAMPLE="${PAIRS_PER_SAMPLE:-2}"

cd "${PROJECT_ROOT}"
CMD=(
  python training/process/convert_audio_mcq_process_rm.py
  --input-jsonl "${INPUT_JSONL}"
  --dataset-root "${DATASET_ROOT}"
  --output-file "${OUTPUT_FILE}"
  --reasoning-field "${REASONING_FIELD}"
  --answer-format "${ANSWER_FORMAT}"
  --audio-position "${AUDIO_POSITION}"
  --pairs-per-sample "${PAIRS_PER_SAMPLE}"
  --shuffle
)

if [ -n "${FALLBACK_REASONING_FIELD}" ]; then
  CMD+=(--fallback-reasoning-field "${FALLBACK_REASONING_FIELD}")
fi

"${CMD[@]}"
