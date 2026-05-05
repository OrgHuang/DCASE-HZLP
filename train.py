#!/usr/bin/env python
import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

# This script is PyTorch-only. Some environments have TensorFlow + Keras 3
# installed, which makes Transformers try to import an unsupported TF stack
# when Trainer is imported.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)


STEP_AUDIO2_CODE_DIR = "/home/ubuntu/model/Step-Audio2"
if STEP_AUDIO2_CODE_DIR not in sys.path:
    sys.path.insert(0, STEP_AUDIO2_CODE_DIR)

from utils import compute_token_num, load_audio, log_mel_spectrogram, padding_mels


IGNORE_INDEX = -100
DEFAULT_SYSTEM_PROMPT = (
    "You are a careful audio question answering assistant. "
    "Listen to the audio and answer the multiple-choice question."
)


def load_rows(data_dir: str):
    jsonl_path = os.path.join(data_dir, "data.jsonl")
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
        return

    try:
        from datasets import load_from_disk
    except ImportError as exc:
        raise RuntimeError(
            "No data.jsonl was found. Install datasets to read the saved Arrow "
            "dataset, or download the full Hugging Face repo snapshot."
        ) from exc

    ds = load_from_disk(data_dir)
    split = ds["train"] if isinstance(ds, dict) or hasattr(ds, "keys") else ds
    for row in split:
        yield dict(row)


def format_question(row: Dict):
    choices = row["choices"]
    choice_lines = "\n".join(
        f"{chr(ord('A') + i)}. {choice}" for i, choice in enumerate(choices)
    )
    return (
        "Question:\n"
        f"{row['question']}\n\n"
        "Choices:\n"
        f"{choice_lines}\n\n"
        "Answer with the exact option text."
    )


def format_target(row: Dict, use_cot: bool):
    answer = str(row["answer"]).strip()
    if use_cot and row.get("gemini_cot"):
        cot = str(row["gemini_cot"]).strip()
        return f"{cot}\n\n<answer>{answer}</answer>"
    return f"<answer>{answer}</answer>"


class AudioMCQSFTDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        tokenizer,
        max_samples: Optional[int],
        max_audio_seconds: Optional[float],
        use_cot: bool,
        skip_missing_audio: bool,
    ):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_audio_samples = (
            int(max_audio_seconds * 16000) if max_audio_seconds else None
        )
        self.use_cot = use_cot
        self.rows = []
        total_rows = 0
        skipped_missing = 0

        for row in load_rows(data_dir):
            total_rows += 1
            audio_path = os.path.join(data_dir, row["audio_path"])
            if skip_missing_audio and not os.path.exists(audio_path):
                skipped_missing += 1
                continue
            self.rows.append(row)
            if max_samples and len(self.rows) >= max_samples:
                break

        if not self.rows:
            raise RuntimeError(f"No training rows found under {data_dir}")

        self.audio_start = "<audio_start>"
        self.audio_patch = "<audio_patch>"
        self.audio_end = "<audio_end>"
        print(
            "Loaded training rows: "
            f"{len(self.rows)} / scanned {total_rows} "
            f"(skipped missing audio: {skipped_missing})"
        )

    def __len__(self):
        return len(self.rows)

    def encode_audio(self, audio_path: str):
        audio = load_audio(audio_path, max_length=self.max_audio_samples)
        mels = []
        placeholders = []
        chunk_size = 16000 * 25
        for start in range(0, audio.shape[0], chunk_size):
            mel = log_mel_spectrogram(audio[start : start + chunk_size], n_mels=128)
            mels.append(mel)
            num_patches = compute_token_num(mel.shape[1])
            placeholders.append(
                self.audio_start
                + (self.audio_patch * num_patches)
                + self.audio_end
            )
        return "".join(placeholders), mels

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        audio_abs_path = os.path.join(self.data_dir, row["audio_path"])
        audio_tokens, mels = self.encode_audio(audio_abs_path)

        user_content = f"{audio_tokens}\n{format_question(row)}"
        prompt = (
            f"<|BOT|>system\n{DEFAULT_SYSTEM_PROMPT}<|EOT|>"
            f"<|BOT|>human\n{user_content}<|EOT|>"
            f"<|BOT|>assistant\n"
        )
        target = format_target(row, self.use_cot) + "<|EOT|>"

        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        target_ids = self.tokenizer(target, add_special_tokens=False)["input_ids"]
        input_ids = prompt_ids + target_ids
        labels = [IGNORE_INDEX] * len(prompt_ids) + target_ids

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "mels": mels,
        }


@dataclass
class DataCollatorForStepAudio2:
    tokenizer: object

    def __call__(self, features):
        pad_id = self.tokenizer.pad_token_id
        max_len = max(len(x["input_ids"]) for x in features)

        input_ids, labels, attention_mask = [], [], []
        all_mels = []
        for feat in features:
            ids = feat["input_ids"]
            labs = feat["labels"]
            pad_len = max_len - ids.numel()

            input_ids.append(F.pad(ids, (0, pad_len), value=pad_id))
            labels.append(F.pad(labs, (0, pad_len), value=IGNORE_INDEX))
            attention_mask.append(
                F.pad(torch.ones_like(ids, dtype=torch.long), (0, pad_len), value=0)
            )
            all_mels.extend(feat["mels"])

        batch = {
            "input_ids": torch.stack(input_ids),
            "labels": torch.stack(labels),
            "attention_mask": torch.stack(attention_mask),
        }
        if all_mels:
            wavs, wav_lens = padding_mels(all_mels)
            batch["wavs"] = wavs
            batch["wav_lens"] = wav_lens
        return batch


class StepAudio2SFTTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=IGNORE_INDEX,
        )
        return (loss, outputs) if return_outputs else loss


def check_stepaudio2_tokenizer(tokenizer):
    tokens = ["<audio_start>", "<audio_patch>", "<audio_end>"]

    for token in tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        print(f"{token}: {token_id}")
        if token_id is None or token_id == tokenizer.unk_token_id:
            raise RuntimeError(f"{token} is not found in tokenizer.")

    probe = "<audio_start>" + ("<audio_patch>" * 3) + "<audio_end>"
    probe_ids = tokenizer(probe, add_special_tokens=False)["input_ids"]
    print("probe_ids:", probe_ids)

    if len(probe_ids) != 5:
        raise RuntimeError(
            f"Audio placeholder tokenization mismatch. Got {probe_ids}."
        )

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        default="/home/ubuntu/model/Step-Audio2/Step-Audio-2-mini",
    )
    parser.add_argument(
        "--data_dir",
        default="/home/ubuntu/data/AudioMCQ-StrongAC-GeminiCoT",
    )
    parser.add_argument(
        "--output_dir",
        default="/home/ubuntu/cp/stepaudio2-mini-audiomcq-lora",
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_audio_seconds", type=float, default=30.0)
    parser.add_argument("--use_cot", action="store_true")
    parser.add_argument(
        "--skip_missing_audio",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip rows whose audio file is missing.",
    )
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument(
        "--allow_cpu",
        action="store_true",
        help="Allow CPU training. This is only practical for tiny debugging runs.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.fp16 and args.bf16:
        raise ValueError("Use either --fp16 or --no-bf16, not both fp16 and bf16.")
    set_seed(args.seed)

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError(
            "CUDA is not available, so Step-Audio-2-mini LoRA training would run "
            "on CPU and appear to hang. Fix the NVIDIA driver/CUDA runtime or pass "
            "--allow_cpu only for tiny debugging runs."
        )

    model_dtype = (
        torch.bfloat16 if args.bf16 else torch.float16
    ) if torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True, padding_side="right"
    )
    tokenizer.eos_token = "<|EOT|>"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = "<|endoftext|>"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=model_dtype,
    )
    model.config.eos_token_id = tokenizer.convert_tokens_to_ids("<|EOT|>")
    for config in (model.config, getattr(model.config, "text_config", None)):
        if config is not None and hasattr(config, "use_cache"):
            config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    check_stepaudio2_tokenizer(tokenizer)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[x.strip() for x in args.lora_target_modules.split(",")],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = AudioMCQSFTDataset(
        data_dir=args.data_dir,
        tokenizer=tokenizer,
        max_samples=args.max_samples,
        max_audio_seconds=args.max_audio_seconds,
        use_cot=args.use_cot,
        skip_missing_audio=args.skip_missing_audio,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        fp16=args.fp16,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
    )
    effective_batch_size = (
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        * training_args.world_size
    )
    estimated_steps_per_epoch = math.ceil(
        len(train_dataset)
        / (
            training_args.per_device_train_batch_size
            * training_args.gradient_accumulation_steps
            * training_args.world_size
        )
    )
    print(
        "Training examples: "
        f"{len(train_dataset)} | per_device_batch_size: "
        f"{training_args.per_device_train_batch_size} | gradient_accumulation_steps: "
        f"{training_args.gradient_accumulation_steps} | world_size: "
        f"{training_args.world_size} | effective_batch_size: {effective_batch_size} | "
        f"estimated optimizer steps per epoch: {estimated_steps_per_epoch}"
    )

    trainer = StepAudio2SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForStepAudio2(tokenizer),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
