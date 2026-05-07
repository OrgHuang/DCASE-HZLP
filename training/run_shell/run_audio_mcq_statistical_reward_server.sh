#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${TRAINING_DIR}/.." && pwd)"

INPUT_JSONL="${INPUT_JSONL:-/home/org/DCASE/AudioMCQ-StrongAC-GeminiCoT-complete/data_acoustic_cot_no_teacher.jsonl}"
REWARD_SERVER_HOST="${REWARD_SERVER_HOST:-127.0.0.1}"
REWARD_SERVER_PORT="${REWARD_SERVER_PORT:-8001}"

cd "${PROJECT_ROOT}"
python training/process/audio_mcq_statistical_reward_server.py \
  --input-jsonl "${INPUT_JSONL}" \
  --host "${REWARD_SERVER_HOST}" \
  --port "${REWARD_SERVER_PORT}"
