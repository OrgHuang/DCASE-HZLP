#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_FILE="${TRAINING_DIR}/datasets/audio-mcq-teacher-lite-cot/train.jsonl"

if [ ! -f "${DATA_FILE}" ]; then
  echo "Preparing teacher-lite AudioMCQ CoT dataset at ${DATA_FILE}"
  bash "${SCRIPT_DIR}/prepare_audio_mcq_teacher_lite_cot.sh"
fi

cd "${TRAINING_DIR}"
CONFIG_FILE="${CONFIG_FILE:-configs/audio_mcq_teacher_lite_qlora_sft.yaml}" bash "${SCRIPT_DIR}/run.sh"
