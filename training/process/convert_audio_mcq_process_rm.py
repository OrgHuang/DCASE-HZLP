#!/usr/bin/env python
"""Build pairwise process-reward data for AudioMCQ RM training."""

from __future__ import annotations

import argparse
import difflib
import json
import random
import re
import sys
import unicodedata
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
        help="System prompt stored in the RM samples.",
    )
    parser.add_argument(
        "--reasoning-field",
        type=str,
        default="gemini_cot",
        help="Primary json field used as the preferred reasoning target.",
    )
    parser.add_argument(
        "--fallback-reasoning-field",
        type=str,
        default=None,
        help="Optional fallback reasoning field.",
    )
    parser.add_argument(
        "--answer-format",
        choices=("letter", "text", "letter_and_text"),
        default="letter_and_text",
        help="How the final answer is rendered in the chosen/rejected responses.",
    )
    parser.add_argument(
        "--audio-position",
        choices=("audio_first", "text_first"),
        default="text_first",
        help="Whether the user message places the audio placeholder before or after the text question block.",
    )
    parser.add_argument(
        "--pairs-per-sample",
        type=int,
        default=2,
        help="How many hard negative preference pairs to keep for each source sample.",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on the number of source rows.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used when shuffling examples.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle output preference pairs before writing.")
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def normalize_for_match(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    text = text.replace("**", " ")
    text = re.sub(r"`+", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    text = re.sub(r"^[\s\"'`([{<]+|[\s\"'`)\]}>.,;:!?]+$", "", text)
    return re.sub(r"\s+", " ", text).strip()


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


def build_user_content(prompt: str, audio_position: str) -> str:
    if audio_position == "text_first":
        return f"{prompt}\n{AUDIO_TEMPLATE}"
    return f"{AUDIO_TEMPLATE}\n{prompt}"


def format_final_answer(answer: str, answer_letter: str, answer_format: str) -> str:
    if answer_format == "letter":
        return f"({answer_letter})"
    if answer_format == "text":
        return answer
    return f"({answer_letter}) {answer}"


def pick_reasoning_text(row: dict[str, Any], reasoning_field: str, fallback_reasoning_field: str | None) -> str:
    primary = str(row.get(reasoning_field, "")).strip()
    if primary:
        return primary
    if fallback_reasoning_field:
        return str(row.get(fallback_reasoning_field, "")).strip()
    return ""


def sanitize_reasoning(reasoning: str) -> str:
    text = str(reasoning).strip()
    if not text:
        return ""

    text = text.replace("```html", "").replace("```", "").strip()
    text = re.sub(r"(?is)<answer>\s*.*?\s*</answer>", "", text)
    text = re.sub(r"(?im)^\s*final answer\s*:\s*.*$", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_target(
    row: dict[str, Any],
    answer_letter: str,
    answer_format: str,
    reasoning_field: str,
    fallback_reasoning_field: str | None,
) -> str:
    final_answer = format_final_answer(str(row["answer"]).strip(), answer_letter, answer_format)
    reasoning = sanitize_reasoning(pick_reasoning_text(row, reasoning_field, fallback_reasoning_field))
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


def get_question_text(row: dict[str, Any]) -> str:
    for key in ("question", "question_text"):
        value = row.get(key)
        if value is not None:
            return str(value)
    raise KeyError("Expected one of ['question', 'question_text'] in dataset row.")


def get_choices(row: dict[str, Any]) -> list[str]:
    for key in ("choices", "multi_choice"):
        value = row.get(key)
        if value is not None:
            return [str(item) for item in value]
    raise KeyError("Expected one of ['choices', 'multi_choice'] in dataset row.")


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

    answer_tag_pattern = re.compile(r"(?is)<answer>\s*(.+?)\s*</answer>")
    for match in answer_tag_pattern.finditer(text):
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


def decode_prediction(raw_text: str, choices: list[str]) -> tuple[str | None, str]:
    letter = extract_letter_answer(raw_text, len(choices))
    if letter is not None:
        return choice_from_letter(letter, choices), f"letter:{letter}"

    matched_choice, matched_by = match_choice_text(raw_text, choices)
    if matched_choice is not None:
        return matched_choice, matched_by

    return None, "unresolved"


def split_reasoning_and_final_answer(text: str) -> tuple[str, str]:
    stripped = str(text).strip()
    if not stripped:
        return "", ""

    match = re.search(r"(?is)(.*?)(?:\n\n|\n)?final answer\s*[:：]\s*(.+)$", stripped)
    if match:
        return match.group(1).strip(), match.group(2).strip()

    return stripped, ""


def pick_hard_negative_choice(answer: str, choices: list[str]) -> tuple[str, str]:
    best_letter = "A"
    best_choice = ""
    best_score = -1.0
    normalized_answer = normalize_for_match(answer)
    for idx, choice in enumerate(choices):
        if normalize_for_match(choice) == normalized_answer:
            continue

        score = difflib.SequenceMatcher(None, normalized_answer, normalize_for_match(choice)).ratio()
        if score > best_score:
            best_score = score
            best_letter = chr(ord("A") + idx)
            best_choice = choice

    if best_choice:
        return best_letter, best_choice

    for idx, choice in enumerate(choices):
        if normalize_for_match(choice) != normalized_answer:
            return chr(ord("A") + idx), choice

    raise ValueError("Need at least one negative choice.")


def answer_text_from_option(choice: str, letter: str, answer_format: str) -> str:
    return format_final_answer(choice, letter, answer_format)


def score_candidate(
    candidate: str,
    answer: str,
    choices: list[str],
    answer_letter: str,
    predicted_choice: str | None,
    matched_by: str,
) -> tuple[float, dict[str, float]]:
    reasoning, final_answer_span = split_reasoning_and_final_answer(candidate)
    normalized_text = normalize_for_match(candidate)
    normalized_reasoning = normalize_for_match(reasoning)
    gold_choice = answer

    breakdown: dict[str, float] = {}
    if not candidate.strip():
        breakdown["empty_penalty"] = -1.5

    if predicted_choice is None:
        breakdown["parse_penalty"] = -0.8
    else:
        breakdown["answer_present"] = 0.4
        if normalize_for_match(predicted_choice) == normalize_for_match(gold_choice):
            breakdown["answer_correct"] = 3.0
        else:
            breakdown["answer_incorrect"] = -0.6

    if "final answer:" in candidate.lower():
        breakdown["has_final_answer_tag"] = 0.3
    elif final_answer_span:
        breakdown["has_terminal_answer"] = 0.15

    if reasoning and len(reasoning) >= 40:
        breakdown["reasoning_nonempty"] = 0.25

    if "<evidence>" in candidate.lower():
        breakdown["evidence_tag"] = 0.15

    if "<reasoning>" in candidate.lower():
        breakdown["reasoning_tag"] = 0.15

    if matched_by.startswith("letter:") or matched_by in {"exact_text", "substring_text"}:
        breakdown["stable_parse"] = 0.25

    if predicted_choice is not None and normalize_for_match(predicted_choice) in normalized_reasoning:
        breakdown["reasoning_answer_consistency"] = 0.35
    elif predicted_choice is not None and reasoning:
        other_choice_mentions = sum(
            1 for choice in choices if normalize_for_match(choice) in normalized_reasoning and choice != predicted_choice
        )
        if other_choice_mentions > 0:
            breakdown["reasoning_choice_conflict"] = -0.35

    extracted_letters = re.findall(r"\(([A-Z])\)", candidate.upper())
    unique_letters = {letter for letter in extracted_letters if letter in {chr(ord('A') + i) for i in range(len(choices))}}
    if len(unique_letters) > 2:
        breakdown["multiple_answer_penalty"] = -0.3

    word_count = len(candidate.split())
    if word_count > 220:
        breakdown["verbosity_penalty"] = -0.25
    elif word_count <= 80:
        breakdown["concise_bonus"] = 0.1

    if normalize_for_match(gold_choice) in normalized_text:
        breakdown["mentions_gold_choice"] = 0.1

    if not any(normalize_for_match(choice) in normalized_text for choice in choices):
        breakdown["no_choice_surface_penalty"] = -0.15

    return sum(breakdown.values()), breakdown


def build_rejected_candidates(
    chosen_text: str,
    answer: str,
    choices: list[str],
    answer_letter: str,
    answer_format: str,
) -> list[tuple[str, str]]:
    wrong_letter, wrong_choice = pick_hard_negative_choice(answer, choices)
    wrong_answer = answer_text_from_option(wrong_choice, wrong_letter, answer_format)
    gold_answer = answer_text_from_option(answer, answer_letter, answer_format)
    reasoning, _ = split_reasoning_and_final_answer(chosen_text)

    candidates: list[tuple[str, str]] = []
    candidates.append(("wrong_short", f"Final answer: {wrong_answer}"))

    if reasoning:
        candidates.append(("contradict_final", f"{reasoning}\n\nFinal answer: {wrong_answer}"))
        generic_reasoning = (
            "The options were compared against the audible cues, and the selected option best matched the clip."
        )
        candidates.append(("generic_wrong", f"{generic_reasoning}\n\nFinal answer: {wrong_answer}"))
        candidates.append(("missing_final", reasoning))

        swapped_reasoning = reasoning
        if normalize_for_match(answer) != normalize_for_match(wrong_choice):
            swapped_reasoning = re.sub(
                re.escape(answer),
                wrong_choice,
                swapped_reasoning,
                flags=re.IGNORECASE,
            )
            swapped_reasoning = re.sub(
                re.escape(gold_answer),
                wrong_answer,
                swapped_reasoning,
                flags=re.IGNORECASE,
            )
        candidates.append(("swapped_reasoning", f"{swapped_reasoning}\n\nFinal answer: {wrong_answer}"))
    else:
        candidates.append(("wrong_text_only", wrong_answer))

    deduped: list[tuple[str, str]] = []
    seen: set[str] = set()
    for tag, text in candidates:
        key = normalize_text(text)
        if key and key not in seen:
            seen.add(key)
            deduped.append((tag, text))

    return deduped


def convert_row_to_pairs(
    row: dict[str, Any],
    dataset_root: Path,
    system_prompt: str,
    answer_format: str,
    reasoning_field: str,
    fallback_reasoning_field: str | None,
    audio_position: str,
    pairs_per_sample: int,
) -> list[dict[str, Any]]:
    audio_path = dataset_root / row["audio_path"]
    if not audio_path.exists():
        raise FileNotFoundError(f"Missing audio file: {audio_path}")

    choices = get_choices(row)
    answer = str(row["answer"]).strip()
    answer_letter = resolve_answer_letter(answer, choices)
    prompt = build_question_prompt(get_question_text(row).strip(), choices)
    chosen_text = build_target(
        row=row,
        answer_letter=answer_letter,
        answer_format=answer_format,
        reasoning_field=reasoning_field,
        fallback_reasoning_field=fallback_reasoning_field,
    )
    chosen_prediction, chosen_matched_by = decode_prediction(chosen_text, choices)
    chosen_score, chosen_breakdown = score_candidate(
        chosen_text, answer, choices, answer_letter, chosen_prediction, chosen_matched_by
    )

    ranked_negatives: list[dict[str, Any]] = []
    for variant_name, rejected_text in build_rejected_candidates(
        chosen_text, answer, choices, answer_letter, answer_format
    ):
        predicted_choice, matched_by = decode_prediction(rejected_text, choices)
        rejected_score, rejected_breakdown = score_candidate(
            rejected_text, answer, choices, answer_letter, predicted_choice, matched_by
        )
        if rejected_score >= chosen_score:
            rejected_score -= 1.0
            rejected_breakdown["force_margin_penalty"] = rejected_breakdown.get("force_margin_penalty", 0.0) - 1.0

        ranked_negatives.append(
            {
                "variant_name": variant_name,
                "text": rejected_text,
                "predicted_choice": predicted_choice,
                "matched_by": matched_by,
                "score": rejected_score,
                "breakdown": rejected_breakdown,
            }
        )

    ranked_negatives.sort(key=lambda item: item["score"], reverse=True)
    selected_negatives = ranked_negatives[: max(1, pairs_per_sample)]

    base_example = {
        "system": system_prompt,
        "messages": [{"role": "user", "content": build_user_content(prompt, audio_position)}],
        "audio": [build_audio_payload(audio_path)],
        "source_dataset": row.get("source_dataset"),
        "question_type": row.get("question_type"),
        "sample_id": row.get("id"),
    }

    rows: list[dict[str, Any]] = []
    for pair_idx, negative in enumerate(selected_negatives):
        rows.append(
            {
                **base_example,
                "chosen": {"role": "assistant", "content": chosen_text},
                "rejected": {"role": "assistant", "content": negative["text"]},
                "reward_meta": {
                    "pair_index": pair_idx,
                    "variant_name": negative["variant_name"],
                    "answer": answer,
                    "answer_letter": answer_letter,
                    "choices": choices,
                    "chosen_score": chosen_score,
                    "rejected_score": negative["score"],
                    "chosen_matched_by": chosen_matched_by,
                    "rejected_matched_by": negative["matched_by"],
                    "chosen_breakdown": chosen_breakdown,
                    "rejected_breakdown": negative["breakdown"],
                    "reasoning_field": reasoning_field,
                    "fallback_reasoning_field": fallback_reasoning_field,
                    "audio_position": audio_position,
                },
            }
        )

    return rows


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    skipped_alignment = 0
    processed_sources = 0

    with args.input_jsonl.open("r", encoding="utf-8") as source:
        for line_idx, line in enumerate(source, start=1):
            if not line.strip():
                continue
            raw_row = json.loads(line)
            try:
                rows.extend(
                    convert_row_to_pairs(
                        row=raw_row,
                        dataset_root=args.dataset_root,
                        system_prompt=args.system_prompt,
                        answer_format=args.answer_format,
                        reasoning_field=args.reasoning_field,
                        fallback_reasoning_field=args.fallback_reasoning_field,
                        audio_position=args.audio_position,
                        pairs_per_sample=args.pairs_per_sample,
                    )
                )
                processed_sources += 1
            except ValueError as err:
                if "Could not align answer" in str(err):
                    skipped_alignment += 1
                    continue
                raise RuntimeError(f"Failed on line {line_idx}: {err}") from err
            except Exception as err:
                raise RuntimeError(f"Failed on line {line_idx}: {err}") from err

            if args.max_samples is not None and processed_sources >= args.max_samples:
                break

    if args.shuffle:
        random.shuffle(rows)

    with args.output_file.open("w", encoding="utf-8") as sink:
        for item in rows:
            sink.write(json.dumps(item, ensure_ascii=False) + "\n")

    meta_path = args.output_file.with_suffix(".meta.json")
    meta = {
        "input_jsonl": str(args.input_jsonl.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "output_file": str(args.output_file.resolve()),
        "num_rows": len(rows),
        "processed_sources": processed_sources,
        "skipped_alignment": skipped_alignment,
        "system_prompt": args.system_prompt,
        "reasoning_field": args.reasoning_field,
        "fallback_reasoning_field": args.fallback_reasoning_field,
        "answer_format": args.answer_format,
        "audio_position": args.audio_position,
        "pairs_per_sample": args.pairs_per_sample,
        "shuffle": args.shuffle,
        "seed": args.seed,
        "score_design": {
            "answer_correct": 3.0,
            "answer_present": 0.4,
            "has_final_answer_tag": 0.3,
            "reasoning_nonempty": 0.25,
            "evidence_tag": 0.15,
            "reasoning_tag": 0.15,
            "stable_parse": 0.25,
            "reasoning_answer_consistency": 0.35,
            "answer_incorrect": -0.6,
            "parse_penalty": -0.8,
            "reasoning_choice_conflict": -0.35,
            "multiple_answer_penalty": -0.3,
            "verbosity_penalty": -0.25,
            "empty_penalty": -1.5,
        },
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
