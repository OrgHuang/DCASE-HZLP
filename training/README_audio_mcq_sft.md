# AudioMCQ SFT for Fun-Audio-Chat

This setup converts `/home/org/DCASE/AudioMCQ-StrongAC-GeminiCoT-complete` into the Fun-Audio-Chat ShareGPT-style audio format and trains a single-GPU-friendly LoRA adapter with 4-bit QLoRA.

## 1. Prepare the training jsonl

```bash
cd /home/org/DCASE/Fun-Audio-Chat/model/Fun-Audio-Chat
conda run -n FunAudioChat bash training/run_shell/prepare_audio_mcq_sft.sh
```

The converted dataset is written to:

```text
training/datasets/audio-mcq-strongac-gemini-cot/train.jsonl
```

By default, each sample uses:

- System prompt: `You are asked to generate text tokens.`
- User prompt:
  - `<|audio_bos|><|AUDIO|><|audio_eos|>`
  - `question +` the `(A) ... (D) ...` multiple-choice block
- Assistant target:
  - Gemini CoT
  - `Final answer: (X) answer text`

If you want answer-only supervision:

```bash
TARGET_MODE=answer_only conda run -n FunAudioChat bash training/run_shell/prepare_audio_mcq_sft.sh
```

## 2. Install training dependencies in `FunAudioChat`

At minimum, the environment needs local `llamafactory` and `peft`.

```bash
cd /home/org/DCASE/Fun-Audio-Chat/model/Fun-Audio-Chat/third_party/LLaMA-Factory
conda run -n FunAudioChat pip install -e . --no-build-isolation
```

`flash-attn` is required if you want the provided AudioMCQ configs to run with `flash_attn: fa2`.

## 3. Run a smoke test

```bash
cd /home/org/DCASE/Fun-Audio-Chat/model/Fun-Audio-Chat/training
CONFIG_FILE=configs/audio_mcq_qlora_sft_smoke.yaml \
conda run -n FunAudioChat bash run_shell/run_audio_mcq_qlora.sh
```

## 4. Run the full QLoRA SFT

```bash
cd /home/org/DCASE/Fun-Audio-Chat/model/Fun-Audio-Chat/training
conda run -n FunAudioChat bash run_shell/run_audio_mcq_qlora.sh
```

## Notes

- The full config is tuned for a single 24GB-class GPU, so it uses `finetuning_type: lora` plus `quantization_bit: 4`.
- The default configs keep `do_eval: false` for stability in single-GPU QLoRA smoke runs. If you want validation, re-enable the commented eval block in the yaml.
- The training launcher auto-registers the Fun-Audio-Chat processor and multimodal plugin through `training/plugin/sitecustomize.py`.
- Output checkpoints are written under `training/saves/Fun-Audio-Chat-8B/`.
