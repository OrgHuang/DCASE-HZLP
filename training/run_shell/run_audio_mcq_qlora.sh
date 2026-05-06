#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${TRAINING_DIR}/.." && pwd)"
DATA_FILE="${TRAINING_DIR}/datasets/audio-mcq-strongac-gemini-cot/train.jsonl"

if [ ! -f "${DATA_FILE}" ]; then
  echo "Preparing AudioMCQ Fun-Audio-Chat SFT dataset at ${DATA_FILE}"
  bash "${SCRIPT_DIR}/prepare_audio_mcq_sft.sh"
fi

cd "${TRAINING_DIR}"
CONFIG_FILE="${CONFIG_FILE:-configs/audio_mcq_qlora_sft.yaml}" bash "${SCRIPT_DIR}/run.sh"
