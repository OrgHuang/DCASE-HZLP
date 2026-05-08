# Enhanced AudioMCQ SFT for Fun-Audio-Chat

This training script upgrades the original AudioMCQ SFT with **multi-view counterfactual data augmentation**, inspired by StepAudio2's robust audio understanding training.

## What's New

### Multi-View Training
Each training sample is expanded into multiple **views**:

| View | Audio | Text | Loss |
|------|-------|------|------|
| `positive_sft` | Original | Original Q+A | Standard CE loss |
| `positive_score` | Original | Original Q+A | Sequence score only |
| `silence_score` | Silence | Original Q+A | Sequence score only |
| `mismatch_score` | Other audio | Original Q+A | Sequence score only |
| `permuted_score` | Original | Shuffled choices + A | Sequence score only |

### Margin Losses
- **Audio margin loss**: The model must assign a higher answer likelihood score to the **real audio** than to **silence** or a **mismatched audio**.
- **Permutation consistency loss**: The answer score should remain stable when option order is shuffled.

These losses are computed on **sequence-level log-probabilities** (averaged over answer tokens), not token-level CE, so they guide the model's confidence calibration without changing the answer itself.

## Quick Start

### 1. Prepare Data (same as before)

```bash
cd /root/model/Fun-Audio-Chat/training
bash run_shell/prepare_audio_mcq_sft.sh
```

### 2. Run Enhanced Training

**Standard QLoRA SFT** (no counterfactual, identical behavior to original):
```bash
cd /root/model/Fun-Audio-Chat/training
bash run_shell/run_audio_mcq_enhanced.sh
```

**With counterfactual audio + margin loss**:
```bash
cd /root/model/Fun-Audio-Chat/training
USE_COUNTERFACTUAL=true AUDIO_MARGIN_WEIGHT=0.5 bash run_shell/run_audio_mcq_enhanced.sh
```

**With all views + margin + permutation consistency**:
```bash
cd /root/model/Fun-Audio-Chat/training
USE_COUNTERFACTUAL=true \
AUDIO_MARGIN_WEIGHT=0.5 \
USE_PERMUTED=true \
PERMUTATION_LOSS_WEIGHT=0.3 \
bash run_shell/run_audio_mcq_enhanced.sh
```

### 3. Resume from Checkpoint

```bash
RESUME_FROM_CHECKPOINT=saves/Fun-Audio-Chat-8B/audio_mcq_enhanced_sft/checkpoint-500 \
bash run_shell/run_audio_mcq_enhanced.sh
```

## Hyperparameters (Environment Variables)

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `../pretrained_models/Fun-Audio-Chat-8B` | Base model path |
| `OUTPUT_DIR` | `saves/Fun-Audio-Chat-8B/audio_mcq_enhanced_sft` | Output directory |
| `NUM_EPOCHS` | `3` | Training epochs |
| `BATCH_SIZE` | `1` | Per-device batch size |
| `GRAD_ACC` | `16` | Gradient accumulation steps |
| `LR` | `2e-4` | Learning rate |
| `LORA_R` | `16` | LoRA rank |
| `LORA_ALPHA` | `32` | LoRA alpha |
| `USE_COUNTERFACTUAL` | `false` | Enable counterfactual views |
| `USE_SILENCE` | `true` | Enable silence view |
| `USE_MISMATCH` | `true` | Enable mismatch view |
| `USE_PERMUTED` | `false` | Enable permuted choices view |
| `AUDIO_MARGIN_WEIGHT` | `0.0` | Weight for audio margin loss |
| `AUDIO_MARGIN` | `0.2` | Margin threshold |
| `PERMUTATION_LOSS_WEIGHT` | `0.0` | Weight for permutation consistency loss |
| `PERMUTATION_MARGIN` | `0.0` | Permutation consistency margin |

## Direct Python Usage

```bash
cd /root/model/Fun-Audio-Chat/training
export PYTHONPATH="/root/model/Fun-Audio-Chat:${PYTHONPATH}"

python train_audio_mcq_enhanced.py \
  --model_name_or_path ../pretrained_models/Fun-Audio-Chat-8B \
  --data_path datasets/audio-mcq-strongac-gemini-cot/train.jsonl \
  --output_dir saves/Fun-Audio-Chat-8B/audio_mcq_enhanced_sft \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 2e-4 \
  --lora_r 16 \
  --lora_alpha 32 \
  --bf16 \
  --gradient_checkpointing \
  --use_counterfactual_audio \
  --audio_margin_weight 0.5 \
  --audio_margin 0.2
```

## Architecture Notes

- The script uses **Transformers Trainer** + **QLoRA** directly (no LLaMA-Factory dependency for training).
- Speech decoder is **disabled** (`disable_speech=True`) since AudioMCQ is a speech-to-text task.
- Model forward is called **without labels** to avoid NaN from all-IGNORE score views; CE loss is computed manually in the trainer.
- Audio features (Mel spectrograms) are extracted on-the-fly by the DataCollator using the model's WhisperFeatureExtractor.

## Files

| File | Purpose |
|------|---------|
| `train_audio_mcq_enhanced.py` | Main training script |
| `run_shell/run_audio_mcq_enhanced.sh` | Bash launcher with env-variable controls |
