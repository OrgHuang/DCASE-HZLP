#!/usr/bin/env python
"""Evaluate a Fun-Audio-Chat LoRA adapter on DCASE 2026 Task 5 dev set."""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import logging
import re
import sys
import time
import unicodedata
from datetime import timedelta
from pathlib import Path
from typing import Any

import librosa
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoProcessor, BitsAndBytesConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from funaudiochat.register import register_funaudiochat
from utils.constant import AUDIO_TEMPLATE, DEFAULT_S2T_PROMPT


register_funaudiochat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/org/DCASE/Harland/DCASE2026-Task5-DevSet"),
        help="Directory containing dev.jsonl and dev_audios/.",
    )
    parser.add_argument(
        "--dataset-file",
        type=Path,
        default=None,
        help="Optional override for the dataset jsonl file. Defaults to <dataset-root>/dev.jsonl.",
    )
    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=PROJECT_ROOT / "training" / "saves" / "Fun-Audio-Chat-8B" / "audio_mcq_qlora_sft_fast",
        help="Path to the fine-tuned LoRA adapter directory.",
    )
    parser.add_argument(
        "--base-model-path",
        type=Path,
        default=None,
        help="Optional override for the base model path. If omitted, try to infer from adapter_config.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "dcase2026_task5_eval_fast",
        help="Where to save predictions, submission csv, and metrics.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional log file path. Defaults to <output-dir>/eval.log.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=20,
        help="Write a progress log every N evaluated samples.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Kept for CLI compatibility. Evaluation now always runs one sample at a time.",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for quick smoke tests.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum tokens generated for each answer.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device or device_map target, for example cuda:0 or cpu.",
    )
    parser.add_argument(
        "--no-load-in-4bit",
        action="store_true",
        help="Disable 4-bit quantized loading for the base model.",
    )
    parser.add_argument(
        "--short-answer-hint",
        action="store_true",
        help="Append an extra final-answer-only hint to the MCQ prompt. Disabled by default to match training prompts.",
    )
    parser.add_argument(
        "--no-short-answer-hint",
        action="store_true",
        help="Legacy alias kept for backward compatibility. Prompts already default to no short-answer hint.",
    )
    args = parser.parse_args()
    args.load_in_4bit = not args.no_load_in_4bit
    if args.no_short_answer_hint:
        args.short_answer_hint = False
    return args


def setup_logger(output_dir: Path, log_file: Path | None) -> tuple[logging.Logger, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if log_file is None:
        log_path = output_dir / "eval.log"
    elif log_file.is_absolute():
        log_path = log_file
    else:
        log_path = output_dir / log_file

    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("dcase2026_eval")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger, log_path


def format_duration(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    text = text.replace("**", " ")
    text = re.sub(r"`+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_for_match(text: str) -> str:
    text = normalize_text(text).lower()
    text = re.sub(r"^[\s\"'`([{<]+|[\s\"'`)\]}>.,;:!?]+$", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_question_prompt(question_text: str, choices: list[str], short_answer_hint: bool) -> str:
    lines = [f"{question_text} Choose the correct option from the following options:"]
    for idx, choice in enumerate(choices):
        lines.append(f"({chr(ord('A') + idx)}) {choice}")

    if short_answer_hint:
        lines.append("Respond with the final answer only in the format: Final answer: (A) option text.")

    return "\n".join(lines)


def load_dataset_rows(dataset_file: Path, max_samples: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with dataset_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if max_samples is not None and len(rows) >= max_samples:
                break
    return rows


def resolve_base_model_path(adapter_path: Path, base_model_path: Path | None) -> Path:
    if base_model_path is not None:
        return base_model_path.resolve()

    config_path = adapter_path / "adapter_config.json"
    if not config_path.exists():
        fallback = PROJECT_ROOT / "pretrained_models" / "Fun-Audio-Chat-8B"
        return fallback.resolve()

    adapter_config = json.loads(config_path.read_text(encoding="utf-8"))
    raw_path = adapter_config.get("base_model_name_or_path")
    if not raw_path:
        fallback = PROJECT_ROOT / "pretrained_models" / "Fun-Audio-Chat-8B"
        return fallback.resolve()

    candidate = Path(raw_path)
    candidates = []
    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        candidates.append((PROJECT_ROOT / candidate).resolve())
        candidates.append((PROJECT_ROOT / "training" / candidate).resolve())
        candidates.append((adapter_path / candidate).resolve())

    for item in candidates:
        if item.exists():
            return item

    fallback = PROJECT_ROOT / "pretrained_models" / "Fun-Audio-Chat-8B"
    return fallback.resolve()


def get_compute_dtype(device: str) -> torch.dtype:
    if device.startswith("cuda") and torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def load_model_and_processor(
    base_model_path: Path,
    adapter_path: Path,
    device: str,
    load_in_4bit: bool,
) -> tuple[AutoProcessor, PeftModel]:
    compute_dtype = get_compute_dtype(device)
    config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    if hasattr(processor, "tokenizer") and hasattr(processor.tokenizer, "padding_side"):
        processor.tokenizer.padding_side = "left"

    model_kwargs: dict[str, Any] = {
        "config": config,
        "trust_remote_code": True,
        "device_map": device,
    }
    if load_in_4bit and device.startswith("cuda"):
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
    else:
        model_kwargs["torch_dtype"] = compute_dtype

    model = AutoModelForSeq2SeqLM.from_pretrained(base_model_path, **model_kwargs)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    if hasattr(model, "generation_config"):
        model.generation_config.do_sample = False
        for attr in ("temperature", "top_p", "top_k"):
            if hasattr(model.generation_config, attr):
                setattr(model.generation_config, attr, None)

    target_model = model.get_base_model()
    if hasattr(target_model, "sp_gen_kwargs"):
        target_model.sp_gen_kwargs.update({"text_greedy": True, "disable_speech": True})
    elif hasattr(model, "sp_gen_kwargs"):
        model.sp_gen_kwargs.update({"text_greedy": True, "disable_speech": True})

    return processor, model


def build_conversation(question_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": DEFAULT_S2T_PROMPT},
        {"role": "user", "content": AUDIO_TEMPLATE + "\n" + question_prompt},
    ]


def generate_single_text(
    model: PeftModel,
    processor: AutoProcessor,
    audio_path: Path,
    question_prompt: str,
    max_new_tokens: int,
) -> str:
    audio = [librosa.load(str(audio_path), sr=16000)[0]]
    text = processor.apply_chat_template(build_conversation(question_prompt), add_generation_prompt=True, tokenize=False)
    inputs = processor(
        text=[text],
        audio=audio,
        return_tensors="pt",
        return_token_type_ids=False,
    ).to(model.device)
    with torch.inference_mode():
        generate_ids, _ = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    generate_ids = generate_ids[:, inputs.input_ids.size(1) :]
    return processor.decode(generate_ids[0], skip_special_tokens=True).strip()


def extract_letter_answer(text: str, num_choices: int) -> str | None:
    valid_letters = {chr(ord("A") + idx) for idx in range(num_choices)}
    patterns = [
        re.compile(r"final answer\s*[:：]\s*\(([A-Z])\)", re.IGNORECASE),
        re.compile(r"^\s*\(([A-Z])\)", re.IGNORECASE | re.MULTILINE),
        re.compile(r"\boption\s+([A-Z])\b", re.IGNORECASE),
        re.compile(r"\banswer(?:\s+is)?\s*[:：-]\s*([A-Z])\b", re.IGNORECASE),
    ]
    matches: list[str] = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            letter = next((group for group in match.groups() if group), "")
            letter = letter.upper()
            if letter in valid_letters:
                matches.append(letter)

    return matches[-1] if matches else None


def choice_from_letter(letter: str, choices: list[str]) -> str:
    return choices[ord(letter) - ord("A")]


def extract_candidate_segments(text: str) -> list[str]:
    segments = [text]
    final_answer_pattern = re.compile(r"final answer\s*[:：]\s*(.+)", re.IGNORECASE | re.DOTALL)
    for match in final_answer_pattern.finditer(text):
        segments.append(match.group(1).strip())

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        segments.append(lines[-1])

    return segments


def match_choice_text(text: str, choices: list[str]) -> tuple[str | None, str]:
    normalized_choices = {normalize_for_match(choice): choice for choice in choices}
    segments = extract_candidate_segments(text)

    for segment in segments:
        normalized_segment = normalize_for_match(segment)
        if normalized_segment in normalized_choices:
            return normalized_choices[normalized_segment], "exact_text"

    ranked_hits: list[tuple[int, int, str]] = []
    normalized_full_text = normalize_for_match(text)
    for normalized_choice, original_choice in normalized_choices.items():
        index = normalized_full_text.rfind(normalized_choice)
        if index != -1:
            ranked_hits.append((index, len(normalized_choice), original_choice))

    if ranked_hits:
        ranked_hits.sort()
        return ranked_hits[-1][2], "substring_text"

    candidates = [normalize_for_match(segment) for segment in segments if normalize_for_match(segment)]
    best_choice = None
    best_score = 0.0
    for candidate in candidates:
        for choice in choices:
            score = difflib.SequenceMatcher(None, candidate, normalize_for_match(choice)).ratio()
            if score > best_score:
                best_score = score
                best_choice = choice

    if best_choice is not None and best_score >= 0.55:
        return best_choice, f"fuzzy_text:{best_score:.2f}"

    return None, "unresolved"


def decode_prediction(raw_text: str, choices: list[str]) -> tuple[str, str]:
    letter = extract_letter_answer(raw_text, len(choices))
    if letter is not None:
        return choice_from_letter(letter, choices), f"letter:{letter}"

    matched_choice, matched_by = match_choice_text(raw_text, choices)
    if matched_choice is not None:
        return matched_choice, matched_by

    return choices[0], "fallback_first_choice"


def evaluate_dataset(
    rows: list[dict[str, Any]],
    dataset_root: Path,
    model: PeftModel,
    processor: AutoProcessor,
    max_new_tokens: int,
    short_answer_hint: bool,
    logger: logging.Logger,
    log_every: int,
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    correct = 0
    parsed_without_fallback = 0
    start_time = time.time()
    total_rows = len(rows)

    progress_bar = tqdm(total=total_rows, desc="Evaluating", unit="sample")
    for row in rows:
        choices = list(row["multi_choice"])
        question_prompt = build_question_prompt(
            str(row["question_text"]).strip(),
            choices,
            short_answer_hint=short_answer_hint,
        )
        raw_output = generate_single_text(
            model=model,
            processor=processor,
            audio_path=dataset_root / row["audio_path"],
            question_prompt=question_prompt,
            max_new_tokens=max_new_tokens,
        )

        predicted_answer, matched_by = decode_prediction(raw_output, choices)
        is_correct = normalize_for_match(predicted_answer) == normalize_for_match(row["answer"])
        if is_correct:
            correct += 1
        if matched_by != "fallback_first_choice":
            parsed_without_fallback += 1

        predictions.append(
            {
                "id": row["id"],
                "audio_path": row["audio_path"],
                "question_text": row["question_text"],
                "choices": choices,
                "gold_answer": row["answer"],
                "predicted_answer": predicted_answer,
                "matched_by": matched_by,
                "correct": is_correct,
                "raw_output": raw_output,
            }
        )

        progress_bar.update(1)
        index = len(predictions)
        last_prediction = predictions[-1]
        should_log_progress = index == 1 or index == total_rows or (log_every > 0 and index % log_every == 0)
        if should_log_progress:
            elapsed = time.time() - start_time
            logger.info(
                "Progress %d/%d (%.2f%%) | accuracy=%.4f | parsed=%d | batch_size=%d | last_id=%s | matched_by=%s | elapsed=%s",
                index,
                total_rows,
                100.0 * index / total_rows if total_rows else 100.0,
                correct / index if index else 0.0,
                parsed_without_fallback,
                1,
                row["id"],
                last_prediction["matched_by"],
                format_duration(elapsed),
            )

    progress_bar.close()

    metrics = {
        "num_samples": len(rows),
        "num_correct": correct,
        "accuracy": correct / len(rows) if rows else 0.0,
        "parsed_without_fallback": parsed_without_fallback,
        "fallback_count": len(rows) - parsed_without_fallback,
        "effective_batch_size": 1,
    }
    return predictions, metrics


def write_outputs(output_dir: Path, predictions: list[dict[str, Any]], metrics: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    submission_path = output_dir / "submission.csv"
    with submission_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["question", "answer"])
        writer.writeheader()
        for row in predictions:
            writer.writerow({"question": row["id"], "answer": row["predicted_answer"]})

    prediction_path = output_dir / "predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    logger, log_path = setup_logger(output_dir, args.log_file)

    dataset_root = args.dataset_root.resolve()
    dataset_file = args.dataset_file.resolve() if args.dataset_file is not None else dataset_root / "dev.jsonl"
    adapter_path = args.adapter_path.resolve()
    base_model_path = resolve_base_model_path(adapter_path, args.base_model_path)

    logger.info("Starting DCASE 2026 Task 5 evaluation.")
    logger.info("Dataset root: %s", dataset_root)
    logger.info("Dataset file: %s", dataset_file)
    logger.info("Adapter path: %s", adapter_path)
    logger.info("Base model path: %s", base_model_path)
    logger.info("Output dir: %s", output_dir)
    logger.info("Log file: %s", log_path)
    if args.batch_size != 1:
        logger.info(
            "Batch inference is disabled in this script revision. Ignoring requested batch_size=%d and using single-sample generation.",
            args.batch_size,
        )
    logger.info(
        "Options | device=%s | load_in_4bit=%s | max_new_tokens=%d | max_samples=%s | log_every=%d | batch_size=%d | short_answer_hint=%s",
        args.device,
        args.load_in_4bit,
        args.max_new_tokens,
        args.max_samples,
        args.log_every,
        1,
        args.short_answer_hint,
    )

    rows = load_dataset_rows(dataset_file, args.max_samples)
    logger.info("Loaded %d evaluation samples.", len(rows))

    processor, model = load_model_and_processor(
        base_model_path=base_model_path,
        adapter_path=adapter_path,
        device=args.device,
        load_in_4bit=args.load_in_4bit,
    )
    logger.info("Model and processor are ready. Starting inference loop.")

    predictions, metrics = evaluate_dataset(
        rows=rows,
        dataset_root=dataset_root,
        model=model,
        processor=processor,
        max_new_tokens=args.max_new_tokens,
        short_answer_hint=args.short_answer_hint,
        logger=logger,
        log_every=args.log_every,
        batch_size=args.batch_size,
    )
    write_outputs(output_dir, predictions, metrics)

    logger.info("Evaluation completed.")
    logger.info("Submission saved to: %s", output_dir / "submission.csv")
    logger.info("Predictions saved to: %s", output_dir / "predictions.jsonl")
    logger.info("Metrics saved to: %s", output_dir / "metrics.json")
    logger.info("Final metrics: %s", json.dumps(metrics, ensure_ascii=False))

    print(json.dumps(
        {
            "dataset_file": str(dataset_file),
            "adapter_path": str(adapter_path),
            "base_model_path": str(base_model_path),
            "output_dir": str(output_dir),
            "log_file": str(log_path),
            **metrics,
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
