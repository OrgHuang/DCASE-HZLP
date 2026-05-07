#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
POLICY_ADAPTER_DIR="${TRAINING_DIR}/saves/Fun-Audio-Chat-8B/audio_mcq_qlora_sft_text_first"
REWARD_INPUT_JSONL="${REWARD_INPUT_JSONL:-/home/org/DCASE/AudioMCQ-StrongAC-GeminiCoT-complete/data_acoustic_cot_no_teacher.jsonl}"
REWARD_SERVER_HOST="${REWARD_SERVER_HOST:-127.0.0.1}"
REWARD_SERVER_PORT="${REWARD_SERVER_PORT:-8001}"
REWARD_SERVER_URL="${REWARD_SERVER_URL:-http://${REWARD_SERVER_HOST}:${REWARD_SERVER_PORT}/reward}"
START_REWARD_SERVER="${START_REWARD_SERVER:-1}"
BASE_CONFIG_FILE="${CONFIG_FILE:-configs/audio_mcq_process_ppo_text_first.yaml}"

if [ ! -d "${POLICY_ADAPTER_DIR}" ]; then
  echo "Missing SFT adapter directory: ${POLICY_ADAPTER_DIR}"
  echo "Train the text-first SFT adapter first, or update ${BASE_CONFIG_FILE}."
  exit 1
fi

if [ ! -f "${REWARD_INPUT_JSONL}" ]; then
  echo "Missing reward reference jsonl: ${REWARD_INPUT_JSONL}"
  exit 1
fi

cd "${TRAINING_DIR}"

RESOLVED_POLICY_ADAPTER="${POLICY_ADAPTER_DIR}"
if [ ! -f "${RESOLVED_POLICY_ADAPTER}/adapter_config.json" ]; then
  LATEST_CHECKPOINT="$(find "${POLICY_ADAPTER_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1)"
  if [ -n "${LATEST_CHECKPOINT}" ] && [ -f "${LATEST_CHECKPOINT}/adapter_config.json" ]; then
    RESOLVED_POLICY_ADAPTER="${LATEST_CHECKPOINT}"
  else
    echo "Could not find adapter_config.json in ${POLICY_ADAPTER_DIR} or its checkpoint-* subdirectories."
    exit 1
  fi
fi

SERVER_PID=""
cleanup() {
  if [ -n "${SERVER_PID}" ] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  if [ -n "${TEMP_CONFIG_FILE:-}" ] && [ -f "${TEMP_CONFIG_FILE}" ]; then
    rm -f "${TEMP_CONFIG_FILE}"
  fi
}
trap cleanup EXIT

if [ "${START_REWARD_SERVER}" = "1" ]; then
  echo "Starting statistical reward server at ${REWARD_SERVER_URL}"
  INPUT_JSONL="${REWARD_INPUT_JSONL}" \
  REWARD_SERVER_HOST="${REWARD_SERVER_HOST}" \
  REWARD_SERVER_PORT="${REWARD_SERVER_PORT}" \
  /bin/bash "${SCRIPT_DIR}/run_audio_mcq_statistical_reward_server.sh" \
    >/tmp/audio_mcq_statistical_reward_server.log 2>&1 &
  SERVER_PID=$!
  sleep 2
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "Statistical reward server failed to start. See /tmp/audio_mcq_statistical_reward_server.log"
    exit 1
  fi
fi

TEMP_CONFIG_FILE="$(mktemp /tmp/audio_mcq_process_ppo_text_first.XXXX.yaml)"
python - "${BASE_CONFIG_FILE}" "${TEMP_CONFIG_FILE}" "${REWARD_SERVER_URL}" "${RESOLVED_POLICY_ADAPTER}" <<'PY'
import pathlib
import re
import sys

src_path, dst_path, reward_url, adapter_path = sys.argv[1:]
text = pathlib.Path(src_path).read_text(encoding="utf-8")
text = re.sub(r"(?m)^reward_model:\s*.*$", f"reward_model: {reward_url}", text)
text = re.sub(r"(?m)^adapter_name_or_path:\s*.*$", f"adapter_name_or_path: {adapter_path}", text)
pathlib.Path(dst_path).write_text(text, encoding="utf-8")
PY

CONFIG_FILE="${TEMP_CONFIG_FILE}" bash "${SCRIPT_DIR}/run.sh"
