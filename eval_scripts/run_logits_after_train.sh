#!/bin/bash
# 等待训练完成，然后自动运行 logits eval
# 用法: bash run_logits_after_train.sh <train_pid> <output_dir_name>

set -euo pipefail

PROJECT_DIR="/root/model/Fun-Audio-Chat"
cd "$PROJECT_DIR"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

RUN_ID="train_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${PROJECT_DIR}/eval_scripts/outputs_logits_after_train_${RUN_ID}"
EVAL_SCRIPT="${PROJECT_DIR}/eval_scripts/eval_base_logits.py"
MODEL_PATH="${PROJECT_DIR}/pretrained_models/Fun-Audio-Chat-8B"
INPUT_JSONL="/root/data/Audio/datasets/Eval/dev.jsonl"
DATA_ROOT="/root/data/Audio/datasets/Eval"

echo "=" * 60
echo "Waiting for training process to finish..."
echo "Monitoring PID: $(pgrep -f "train_audio_mcq_enhanced_v2" | paste -sd ' ' -)"
echo "=" * 60

# 等待训练进程结束
while pgrep -f "train_audio_mcq_enhanced_v2" > /dev/null 2>&1; do
    NOW=$(date +"%H:%M:%S")
    echo "[$NOW] Training still running... GPU memory: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 'N/A') MiB"
    sleep 60
done

echo ""
echo "Training finished at $(date)!"

# 找到最新的 checkpoint
SAVES_DIR="${PROJECT_DIR}/training/saves/Fun-Audio-Chat-8B"
LATEST_RUN=$(ls -dt "${SAVES_DIR}"/*/ 2>/dev/null | head -1)
LATEST_CKPT=""

if [ -d "${LATEST_RUN}" ]; then
    # 找最新的 checkpoint (数字最大的)
    LATEST_CKPT=$(ls -d "${LATEST_RUN}/checkpoint-"*/ 2>/dev/null | sort -t'-' -k2 -n | tail -1)
fi

# 如果 checkpoint 子目录里有 adapter，直接用；否则用根目录
if [ -n "${LATEST_CKPT}" ] && [ -f "${LATEST_CKPT}/adapter_model.safetensors" ]; then
    ADAPTER_PATH="${LATEST_CKPT%/}"
else
    ADAPTER_PATH="${LATEST_RUN%/}"
fi

echo "Latest training run: ${LATEST_RUN}"
echo "Adapter path       : ${ADAPTER_PATH}"

# 先跑 logits eval on the new SFT model
echo ""
echo "Running Yes-logit eval on new SFT model..."
mkdir -p "${OUTPUT_DIR}/sft_logits"

python "${EVAL_SCRIPT}" \
    --model_path "${MODEL_PATH}" \
    --input_jsonl "${INPUT_JSONL}" \
    --data_root "${DATA_ROOT}" \
    --output_dir "${OUTPUT_DIR}/sft_logits" \
    2>&1 | tee "${OUTPUT_DIR}/sft_logits/eval.log"

echo ""
echo "Done! Results saved to: ${OUTPUT_DIR}/sft_logits/"
