# AudioMCQ Process RL

This setup adds a lightweight process-reward pipeline on top of the current text-first AudioMCQ LoRA training path.

## Pipeline

1. Train or reuse a text-first SFT adapter.
2. Run PPO starting from the SFT adapter.
3. Use a local statistical reward server instead of training a reward model.

An older RM-based path is still kept in the repo, but PPO now defaults to the statistical reward server.

## Default data choices

- Source file: `/home/org/DCASE/AudioMCQ-StrongAC-GeminiCoT-complete/data_acoustic_cot_no_teacher.jsonl`
- Preferred reasoning field: `gemini_cot`
- Fallback reasoning field: none
- Prompt order: `text_first`

This version now uses only `gemini_cot` by default.

## How scoring works

The PPO reward is now computed by a local HTTP server using statistical scoring, not by a learned reward model.

Main positive terms:

- answer correctness
- sentence-level Gemini CoT coverage similarity
- final-answer format stability
- reasoning and answer consistency

Main penalties:

- overlong reasoning
- repeated bigrams / repetition
- conflicting answer mentions
- multiple final choice signals

The score used by the server is:

```text
R_total
= 4.0 * answer_component
+ 1.5 * cot_similarity
+ 0.5 * format_component
+ 0.7 * consistency_component
- 0.5 * length_penalty
- 0.8 * conflict_penalty
```

Where:

- `answer_component` is `+1.0` for correct, `-0.5` for parseable but wrong, `-1.0` for unparseable.
- `cot_similarity` greedily matches each reference Gemini CoT sentence to one generated sentence.
- `format_component` rewards `Final answer:` plus a parseable answer near the end.
- `consistency_component` rewards reasoning that supports the chosen answer.
- `length_penalty` grows when reasoning is much longer than the reference or highly repetitive.
- `conflict_penalty` grows when multiple choices are simultaneously indicated.

The code lives in:

- `training/process/audio_mcq_statistical_reward.py`
- `training/process/audio_mcq_statistical_reward_server.py`
- `training/run_shell/run_audio_mcq_statistical_reward_server.sh`

## Run PPO with the statistical reward server

```bash
cd /home/org/DCASE/Fun-Audio-Chat/model/Fun-Audio-Chat/training
conda run -n FunAudioChat bash run_shell/run_audio_mcq_process_ppo_text_first.sh
```

The launcher will:

- reuse `audio_mcq_qlora_sft_text_first` as the policy start point
- auto-start the local statistical reward server on `127.0.0.1:8001`
- patch the PPO config to point to that reward URL before launching training

Default PPO policy adapter start:

```text
saves/Fun-Audio-Chat-8B/audio_mcq_qlora_sft_text_first
```

Default PPO output:

```text
saves/Fun-Audio-Chat-8B/audio_mcq_process_ppo_text_first_stat_reward
```

## Notes

- This is a pragmatic first pass, not a full custom GRPO implementation.
- The reward is deterministic and inspectable, which makes debugging easier than a learned RM.
- For this task, keeping `max_new_tokens` small in PPO is intentional. It reduces the chance that the policy learns to win reward simply by generating longer CoT.
