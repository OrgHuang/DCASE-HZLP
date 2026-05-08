#!/bin/bash
#SBATCH --job-name=eval_fun_audio_chat_sft
#SBATCH --partition=VM-GPU-L
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --output=/root/model/Fun-Audio-Chat/eval_scripts/outputs/eval_sft_%j.log
#SBATCH --error=/root/model/Fun-Audio-Chat/eval_scripts/outputs/eval_sft_%j.log

set -eo pipefail

PROJECT_DIR="/root/model/Fun-Audio-Chat"
cd "$PROJECT_DIR"

RUN_ID="${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}"

MODEL_PATH="/root/model/Fun-Audio-Chat/pretrained_models/Fun-Audio-Chat-8B"
ADAPTER_PATH="/root/model/Fun-Audio-Chat/training/saves/Fun-Audio-Chat-8B/audio_mcq_qlora_sft_fast/checkpoint-3654"
FUN_REPO_PATH="/root/model/Fun-Audio-Chat"
INPUT_JSONL="/root/data/Audio/datasets/Eval/dev.jsonl"
DATA_ROOT="/root/data/Audio/datasets/Eval"
OUTPUT_DIR="/root/model/Fun-Audio-Chat/eval_scripts/outputs/sft_eval_${RUN_ID}"
EVAL_SCRIPT="/root/model/Fun-Audio-Chat/eval_scripts/eval_sft_devset.py"

LIMIT="${LIMIT:--1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
DTYPE="${DTYPE:-bfloat16}"

mkdir -p "$OUTPUT_DIR"

echo "Working directory: $(pwd)"
echo "Run ID        : $RUN_ID"
echo "Model path    : $MODEL_PATH"
echo "Adapter path  : $ADAPTER_PATH"
echo "Input JSONL   : $INPUT_JSONL"
echo "Data root     : $DATA_ROOT"
echo "Output dir    : $OUTPUT_DIR"
echo "Limit         : $LIMIT"
echo "Max new tokens: $MAX_NEW_TOKENS"
echo "Dtype         : $DTYPE"

test -d "$FUN_REPO_PATH"
test -d "$MODEL_PATH"
test -d "$ADAPTER_PATH"
test -f "$INPUT_JSONL"
test -f "$EVAL_SCRIPT"

export PYTHONPATH="${FUN_REPO_PATH}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

python "$EVAL_SCRIPT" \
  --model_path "$MODEL_PATH" \
  --adapter_path "$ADAPTER_PATH" \
  --fun_repo_path "$FUN_REPO_PATH" \
  --input_jsonl "$INPUT_JSONL" \
  --data_root "$DATA_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --limit "$LIMIT" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --dtype "$DTYPE" \
  2>&1 | tee "${OUTPUT_DIR}/eval_${RUN_ID}.log"
