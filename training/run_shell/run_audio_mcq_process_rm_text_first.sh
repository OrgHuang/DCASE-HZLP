#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_FILE="${TRAINING_DIR}/datasets/audio-mcq-strongac-process-rm-text-first/train.jsonl"
POLICY_ADAPTER_DIR="${TRAINING_DIR}/saves/Fun-Audio-Chat-8B/audio_mcq_qlora_sft_text_first"
BASE_CONFIG_FILE="${CONFIG_FILE:-configs/audio_mcq_process_rm_text_first.yaml}"

if [ ! -f "${DATA_FILE}" ]; then
  echo "Preparing AudioMCQ process-RM dataset at ${DATA_FILE}"
  bash "${SCRIPT_DIR}/prepare_audio_mcq_process_rm_text_first.sh"
fi

if [ ! -d "${POLICY_ADAPTER_DIR}" ]; then
  echo "Missing SFT adapter directory: ${POLICY_ADAPTER_DIR}"
  echo "Train the text-first SFT adapter first, or update ${BASE_CONFIG_FILE}."
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

TEMP_CONFIG_FILE="$(mktemp /tmp/audio_mcq_process_rm_text_first.XXXX.yaml)"
cleanup() {
  if [ -n "${TEMP_CONFIG_FILE:-}" ] && [ -f "${TEMP_CONFIG_FILE}" ]; then
    rm -f "${TEMP_CONFIG_FILE}"
  fi
}
trap cleanup EXIT

python - "${BASE_CONFIG_FILE}" "${TEMP_CONFIG_FILE}" "${RESOLVED_POLICY_ADAPTER}" <<'PY'
import pathlib
import re
import sys

src_path, dst_path, adapter_path = sys.argv[1:]
text = pathlib.Path(src_path).read_text(encoding="utf-8")
text = re.sub(r"(?m)^adapter_name_or_path:\s*.*$", f"adapter_name_or_path: {adapter_path}", text)
pathlib.Path(dst_path).write_text(text, encoding="utf-8")
PY

CONFIG_FILE="${TEMP_CONFIG_FILE}" bash "${SCRIPT_DIR}/run.sh"
