#!/usr/bin/env python3
"""Low-VRAM Yes-logit eval: uses 4-bit quantization so it won't clash with training."""

import os, sys, json, re, argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np
import librosa
from tqdm import tqdm


def load_jsonl(path): return [json.loads(line) for line in open(path) if line.strip()]
def write_jsonl(data, path):
    with open(path, "w") as f:
        for item in data: f.write(json.dumps(item, ensure_ascii=False) + "\n")

def get_question_text(item):
    for k in ["question_text", "question", "query"]:
        if k in item and item[k]: return str(item[k]).strip()
    return ""

def get_audio_path(item, data_root):
    p = str(item.get("audio_path", ""))
    return p if os.path.isabs(p) else str(Path(data_root) / p)

def get_choices(item):
    raw = item.get("multi_choice", item.get("choices", []))
    return [str(x).strip() for x in raw] if isinstance(raw, list) else []

def get_gold_answer(item): return str(item.get("answer", "")).strip()
def normalize_text(x): return " ".join(str(x).strip().split()).lower()
def is_correct(p, g): return normalize_text(p) == normalize_text(g)


# =========================
# 模型加载（4-bit 量化）
# =========================

def load_model_cpu(model_path, fun_repo_path, adapter_path=None):
    fun_repo_path = str(Path(fun_repo_path).resolve())
    if fun_repo_path not in sys.path:
        sys.path.insert(0, fun_repo_path)

    from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoProcessor
    from funaudiochat.register import register_funaudiochat
    register_funaudiochat()

    processor = AutoProcessor.from_pretrained(model_path)
    processor.tokenizer.padding_side = "right"

    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )

    if adapter_path:
        print(f"Loading adapter: {adapter_path}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.to("cpu")

    model.eval()
    return model, processor


# =========================
# Token IDs
# =========================

def init_yes_token_ids(processor):
    tokenizer = processor.tokenizer
    variants = [" Yes", "Yes"]
    ids = []
    for v in variants:
        toks = tokenizer.encode(v, add_special_tokens=False)
        if len(toks) == 1:
            ids.append(toks[0])
            print(f'  "{v}" -> {toks[0]}')
    print(f"Yes token IDs: {ids}")
    return ids


# =========================
# Core eval
# =========================

def build_prompt(question, option):
    return (
        f"Question: {question}\n"
        f"Proposed Answer: {option}\n"
        f"Is the proposed answer correct?"
    )


@torch.inference_mode()
def evaluate_single_mcq(model, processor, audio_path, question_text, choices, gold_answer, yes_token_ids):
    from utils.constant import AUDIO_TEMPLATE

    audio = [librosa.load(audio_path, sr=16000)[0]]
    device = next(model.parameters()).device

    full_texts = []
    for opt in choices:
        conv = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": AUDIO_TEMPLATE + "\n" + build_prompt(question_text, opt)},
        ]
        text = processor.apply_chat_template([conv], add_generation_prompt=True, tokenize=False)[0]
        full_texts.append(text)

    inputs = processor(
        text=full_texts, audio=audio * len(choices),
        return_tensors="pt", padding=True, return_token_type_ids=False,
    ).to(device)

    outputs = model(**inputs)
    logits = outputs.logits  # [batch, seq_len, vocab]

    last_positions = inputs.attention_mask.sum(dim=1) - 1
    batch_size = logits.size(0)
    yes_logits_list = [
        float(np.mean([logits[i, last_positions[i], tid].item() for tid in yes_token_ids]))
        for i in range(batch_size)
    ]

    best_idx = int(np.argmax(yes_logits_list))
    prediction = choices[best_idx]
    correct = is_correct(prediction, gold_answer)

    options = [
        {"index": i, "text": c, "yes_logit": round(yl, 4), "is_best": i == best_idx}
        for i, (c, yl) in enumerate(zip(choices, yes_logits_list))
    ]
    return {"predicted_index": best_idx, "predicted_answer": prediction, "gold_answer": gold_answer, "correct": correct, "options": options}


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/root/model/Fun-Audio-Chat/pretrained_models/Fun-Audio-Chat-8B")
    parser.add_argument("--adapter_path", default="")
    parser.add_argument("--fun_repo_path", default="/root/model/Fun-Audio-Chat")
    parser.add_argument("--input_jsonl", default="/root/data/Audio/datasets/Eval/dev.jsonl")
    parser.add_argument("--data_root", default="/root/data/Audio/datasets/Eval")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--limit", type=int, default=-1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "predictions.jsonl")

    data = load_jsonl(args.input_jsonl)
    if args.limit > 0: data = data[:args.limit]
    print(f"Loaded {len(data)} samples.")

    model, processor = load_model_cpu(args.model_path, args.fun_repo_path, args.adapter_path or None)
    yes_token_ids = init_yes_token_ids(processor)

    results, correct_count = [], 0
    pbar = tqdm(data, desc="Evaluating (4-bit)")
    for item in pbar:
        qid = str(item.get("id", item.get("_line_id", 0)))
        qtext = get_question_text(item)
        choices = get_choices(item)
        gold = get_gold_answer(item)
        audio_path = get_audio_path(item, args.data_root)

        try:
            res = evaluate_single_mcq(model, processor, audio_path, qtext, choices, gold, yes_token_ids)
            correct_count += int(res["correct"])
            pbar.set_postfix({"acc": f"{correct_count}/{len(results)+1}"})
            results.append({
                "question_id": qid, "question_text": qtext, "gold_answer": gold,
                "predicted_answer": res["predicted_answer"], "correct": res["correct"],
                "yes_logits": res["options"],
            })
        except Exception as e:
            results.append({"question_id": qid, "question_text": qtext, "gold_answer": gold, "error": repr(e), "correct": False})

        if len(results) % 10 == 0:
            write_jsonl(results, log_path)

    write_jsonl(results, log_path)
    total = sum(1 for r in results if "error" not in r)
    acc = correct_count / total if total > 0 else 0
    print(f"\nTotal: {total} | Correct: {correct_count} | Wrong: {total-correct_count} | Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"Results: {log_path}")


if __name__ == "__main__":
    main()
