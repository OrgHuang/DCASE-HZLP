#!/usr/bin/env python
import argparse
import csv
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


STEP_AUDIO2_CODE_DIR = "/home/ubuntu/model/Step-Audio2"
if STEP_AUDIO2_CODE_DIR not in sys.path:
    sys.path.insert(0, STEP_AUDIO2_CODE_DIR)

from utils import compute_token_num, load_audio, log_mel_spectrogram, padding_mels


DEFAULT_SYSTEM_PROMPT = (
    "You are a careful audio question answering assistant. "
    "Listen to the audio and answer the multiple-choice question."
)


def load_dev_rows(data_dir: str, max_samples: Optional[int] = None) -> List[Dict]:
    path = os.path.join(data_dir, "dev.jsonl")
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if max_samples and len(rows) >= max_samples:
                    break
    return rows


def format_question(row: Dict) -> str:
    choice_lines = "\n".join(
        f"{chr(ord('A') + i)}. {choice}"
        for i, choice in enumerate(row["multi_choice"])
    )
    return (
        "Question:\n"
        f"{row['question_text']}\n\n"
        "Choices:\n"
        f"{choice_lines}\n\n"
        "Answer with the exact option text."
    )


def encode_audio(audio_path: str, max_audio_seconds: Optional[float]):
    max_audio_samples = int(max_audio_seconds * 16000) if max_audio_seconds else None
    audio = load_audio(audio_path, max_length=max_audio_samples)
    mels = []
    placeholders = []
    chunk_size = 16000 * 25
    for start in range(0, audio.shape[0], chunk_size):
        mel = log_mel_spectrogram(audio[start : start + chunk_size], n_mels=128)
        mels.append(mel)
        placeholders.append(
            "<audio_start>"
            + ("<audio_patch>" * compute_token_num(mel.shape[1]))
            + "<audio_end>"
        )
    return "".join(placeholders), mels


def build_prompt(row: Dict, audio_tokens: str) -> str:
    user_content = f"{audio_tokens}\n{format_question(row)}"
    return (
        f"<|BOT|>system\n{DEFAULT_SYSTEM_PROMPT}<|EOT|>"
        f"<|BOT|>human\n{user_content}<|EOT|>"
        f"<|BOT|>assistant\n"
    )


def normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def extract_answer(text: str, choices: List[str]) -> Tuple[str, str]:
    raw = text.strip()
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", raw, flags=re.I | re.S)
    candidate = match.group(1).strip() if match else raw

    candidate = candidate.split("<|EOT|>")[0].strip()
    candidate = candidate.strip(" \n\t\"'`")

    norm_candidate = normalize(candidate)
    for choice in choices:
        if normalize(choice) == norm_candidate:
            return choice, candidate

    for choice in choices:
        if normalize(choice) in norm_candidate:
            return choice, candidate

    letter_match = re.match(r"^\(?([A-Da-d])\)?[\s.:)-]*", candidate)
    if letter_match:
        idx = ord(letter_match.group(1).upper()) - ord("A")
        if 0 <= idx < len(choices):
            return choices[idx], candidate

    return candidate, candidate


def load_model(args):
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError(
            "CUDA is not available. Pass --allow_cpu only for a tiny debug run."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.bf16 and device.type == "cuda" else torch.float32

    tokenizer_path = args.lora_path if os.path.exists(
        os.path.join(args.lora_path, "tokenizer_config.json")
    ) else args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=True, padding_side="right"
    )
    tokenizer.eos_token = "<|EOT|>"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = "<|endoftext|>"

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    model = PeftModel.from_pretrained(base_model, args.lora_path)
    model.config.eos_token_id = tokenizer.convert_tokens_to_ids("<|EOT|>")
    for config in (model.config, getattr(model.config, "text_config", None)):
        if config is not None and hasattr(config, "use_cache"):
            config.use_cache = True
    model.to(device)
    model.eval()
    return model, tokenizer, device


@torch.inference_mode()
def predict_one(model, tokenizer, device, row: Dict, args) -> Tuple[str, str]:
    audio_path = os.path.join(args.data_dir, row["audio_path"])
    audio_tokens, mels = encode_audio(audio_path, args.max_audio_seconds)
    prompt = build_prompt(row, audio_tokens)

    input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")[
        "input_ids"
    ].to(device)
    attention_mask = torch.ones_like(input_ids, device=device)

    wavs, wav_lens = padding_mels(mels)
    wavs = wavs.to(device)
    wav_lens = wav_lens.to(device)
    if args.bf16 and device.type == "cuda":
        wavs = wavs.bfloat16()

    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        wavs=wavs,
        wav_lens=wav_lens,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.convert_tokens_to_ids("<|EOT|>"),
    )
    new_ids = output_ids[0, input_ids.shape[-1] :]
    text_ids = [token_id for token_id in new_ids.tolist() if token_id < 151688]
    raw_text = tokenizer.decode(text_ids, skip_special_tokens=False)
    pred, extracted = extract_answer(raw_text, row["multi_choice"])
    return pred, extracted


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        default="/home/ubuntu/model/Step-Audio2/Step-Audio-2-mini",
    )
    parser.add_argument("--lora_path", default="/home/ubuntu/out")
    parser.add_argument("--data_dir", default="/home/ubuntu/data/DCASE2026-Task5-DevSet")
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_audio_seconds", type=float, default=30.0)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow_cpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = load_dev_rows(args.data_dir, args.max_samples)
    model, tokenizer, device = load_model(args)

    correct = 0
    results = []
    for row in tqdm(rows, desc="Evaluating"):
        pred, raw_pred = predict_one(model, tokenizer, device, row, args)
        gold = row["answer"].strip()
        is_correct = normalize(pred) == normalize(gold)
        correct += int(is_correct)
        results.append(
            {
                "question": row["id"],
                "answer": pred,
                "gold": gold,
                "correct": is_correct,
                "raw_prediction": raw_pred,
            }
        )

    total = len(rows)
    accuracy = correct / total if total else 0.0
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.6f} ({accuracy * 100:.2f}%)")

    if args.output_csv:
        with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["question", "answer", "gold", "correct", "raw_prediction"],
            )
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
