#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_FILE="${TRAINING_DIR}/datasets/audio-mcq-strongac-gemini-cot-text-first/train.jsonl"

if [ ! -f "${DATA_FILE}" ]; then
  echo "Preparing text-first AudioMCQ Fun-Audio-Chat SFT dataset at ${DATA_FILE}"
  bash "${SCRIPT_DIR}/prepare_audio_mcq_sft_text_first.sh"
fi

cd "${TRAINING_DIR}"
CONFIG_FILE="${CONFIG_FILE:-configs/audio_mcq_qlora_sft_text_first.yaml}" bash "${SCRIPT_DIR}/run.sh"
