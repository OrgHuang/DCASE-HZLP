#!/usr/bin/env python
"""Convert AudioMCQ StrongAC Gemini-CoT data into Fun-Audio-Chat SFT format."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.constant import AUDIO_PAD_TOKEN, AUDIO_TEMPLATE, DEFAULT_S2T_PROMPT, TOKEN_FPS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True, help="Path to the source jsonl file.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Root directory of the audio dataset.")
    parser.add_argument("--output-file", type=Path, required=True, help="Destination jsonl file.")
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=DEFAULT_S2T_PROMPT,
        help="System prompt stored in the SFT samples.",
    )
    parser.add_argument(
        "--target-mode",
        choices=("cot_answer", "answer_only"),
        default="cot_answer",
        help="Whether to supervise with Gemini CoT plus final answer, or answer only.",
    )
    parser.add_argument(
        "--answer-format",
        choices=("letter", "text", "letter_and_text"),
        default="letter_and_text",
        help="How the final answer is rendered in the assistant target.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on the number of converted samples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when shuffling examples.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle samples before writing.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def resolve_answer_letter(answer: str, choices: list[str]) -> str:
    normalized_answer = normalize_text(answer)
    for idx, choice in enumerate(choices):
        if normalize_text(choice) == normalized_answer:
            return chr(ord("A") + idx)

    lowered_answer = normalized_answer.casefold()
    for idx, choice in enumerate(choices):
        if normalize_text(choice).casefold() == lowered_answer:
            return chr(ord("A") + idx)

    raise ValueError(f"Could not align answer {answer!r} with choices {choices!r}.")


def build_question_prompt(question: str, choices: list[str]) -> str:
    lines = [f"{question} Choose the correct option from the following options:"]
    for idx, choice in enumerate(choices):
        lines.append(f"({chr(ord('A') + idx)}) {choice}")
    return "\n".join(lines)


def format_final_answer(answer: str, answer_letter: str, answer_format: str) -> str:
    if answer_format == "letter":
        return f"({answer_letter})"
    if answer_format == "text":
        return answer
    return f"({answer_letter}) {answer}"


def build_target(row: dict[str, Any], answer_letter: str, answer_format: str, target_mode: str) -> str:
    final_answer = format_final_answer(str(row["answer"]).strip(), answer_letter, answer_format)
    if target_mode == "answer_only":
        return final_answer

    reasoning = str(row.get("gemini_cot", "")).strip()
    if reasoning:
        return f"{reasoning}\n\nFinal answer: {final_answer}"
    return f"Final answer: {final_answer}"


def get_audio_duration_seconds(audio_path: Path) -> float:
    try:
        info = sf.info(str(audio_path))
        if info.frames and info.samplerate:
            return float(info.frames) / float(info.samplerate)
    except RuntimeError:
        pass

    import librosa

    return float(librosa.get_duration(path=str(audio_path)))


def build_audio_payload(audio_path: Path) -> str:
    duration_seconds = get_audio_duration_seconds(audio_path)
    num_audio_tokens = max(1, int(duration_seconds * TOKEN_FPS))
    audio_payload = {
        "path": str(audio_path.resolve()),
        "text": "",
        "token": AUDIO_PAD_TOKEN * num_audio_tokens,
        "ref_path": "",
        "ref_text": "",
    }
    return json.dumps(audio_payload, ensure_ascii=False, sort_keys=True)


def convert_row(row: dict[str, Any], dataset_root: Path, system_prompt: str, answer_format: str, target_mode: str) -> dict[str, Any]:
    audio_path = dataset_root / row["audio_path"]
    if not audio_path.exists():
        raise FileNotFoundError(f"Missing audio file: {audio_path}")

    answer_letter = resolve_answer_letter(str(row["answer"]), list(row["choices"]))
    prompt = build_question_prompt(str(row["question"]).strip(), list(row["choices"]))
    target = build_target(row, answer_letter, answer_format, target_mode)

    messages = [
        {"role": "user", "content": f"{AUDIO_TEMPLATE}\n{prompt}"},
        {"role": "assistant", "content": target},
    ]

    return {
        "system": system_prompt,
        "messages": messages,
        "audio": [build_audio_payload(audio_path)],
        "source_dataset": row.get("source_dataset"),
        "question_type": row.get("question_type"),
        "sample_id": row.get("id"),
    }


def main() -> None:
    args = parse_args()

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    skipped_alignment = 0

    with args.input_jsonl.open("r", encoding="utf-8") as source:
        for line_idx, line in enumerate(source, start=1):
            if not line.strip():
                continue
            raw_row = json.loads(line)
            try:
                rows.append(
                    convert_row(
                        row=raw_row,
                        dataset_root=args.dataset_root,
                        system_prompt=args.system_prompt,
                        answer_format=args.answer_format,
                        target_mode=args.target_mode,
                    )
                )
            except ValueError as exc:
                skipped_alignment += 1
                print(f"[warn] skipped line {line_idx}: {exc}")
                continue

            if args.max_samples is not None and len(rows) >= args.max_samples:
                break

    if args.shuffle:
        random.Random(args.seed).shuffle(rows)

    with args.output_file.open("w", encoding="utf-8") as sink:
        for row in rows:
            sink.write(json.dumps(row, ensure_ascii=False))
            sink.write("\n")

    metadata = {
        "input_jsonl": str(args.input_jsonl.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "output_file": str(args.output_file.resolve()),
        "num_rows": len(rows),
        "skipped_alignment": skipped_alignment,
        "system_prompt": args.system_prompt,
        "target_mode": args.target_mode,
        "answer_format": args.answer_format,
        "shuffle": args.shuffle,
        "seed": args.seed,
    }
    metadata_path = args.output_file.with_suffix(".meta.json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved {len(rows)} rows to {args.output_file}")
    print(f"Metadata written to {metadata_path}")


if __name__ == "__main__":
    main()
