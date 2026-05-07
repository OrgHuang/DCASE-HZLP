#!/usr/bin/env python
"""Statistical reward functions for AudioMCQ process PPO."""

from __future__ import annotations

import difflib
import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


AUDIO_SPECIAL_PATTERN = re.compile(r"<\|audio[^>]*\|>|<\|AUDIO\|>", re.IGNORECASE)
ASSISTANT_TAG = "<|im_start|>assistant\n"
USER_TAG = "<|im_start|>user\n"
IM_END_TAG = "<|im_end|>"
META_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"^\s*my analysis",
        r"^\s*identifying ",
        r"^\s*okay\b",
        r"^\s*let'?s\b",
        r"^\s*the task\b",
        r"^\s*first[, ]",
        r"^\s*now[, ]",
        r"^\s*therefore\b",
    ]
]


@dataclass
class ReferenceSample:
    question_key: str
    question_text: str
    answer: str
    answer_letter: str
    choices: list[str]
    reference_reasoning: str
    reference_sentences: list[str]
    reference_word_count: int


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def normalize_for_match(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    text = text.replace("**", " ")
    text = re.sub(r"`+", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    text = re.sub(r"^[\s\"'`([{<]+|[\s\"'`)\]}>.,;:!?]+$", "", text)
    return re.sub(r"\s+", " ", text).strip()


def strip_audio_markup(text: str) -> str:
    text = AUDIO_SPECIAL_PATTERN.sub(" ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


def build_question_prompt(question: str, choices: list[str]) -> str:
    lines = [f"{question} Choose the correct option from the following options:"]
    for idx, choice in enumerate(choices):
        lines.append(f"({chr(ord('A') + idx)}) {choice}")
    return "\n".join(lines)


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


def sanitize_reasoning(reasoning: str) -> str:
    text = str(reasoning).strip()
    if not text:
        return ""

    text = text.replace("```html", "").replace("```", "").strip()
    text = re.sub(r"(?is)<answer>\s*.*?\s*</answer>", "", text)
    text = re.sub(r"(?im)^\s*final answer\s*:\s*.*$", "", text)
    text = text.replace("**", " ")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_question_key_from_chat(message_text: str) -> str:
    user_match = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", message_text, re.DOTALL)
    if user_match:
        user_content = user_match.group(1)
    else:
        start = message_text.rfind(USER_TAG)
        if start == -1:
            user_content = message_text
        else:
            user_content = message_text[start + len(USER_TAG) :]
            if IM_END_TAG in user_content:
                user_content = user_content.split(IM_END_TAG, 1)[0]

    user_content = strip_audio_markup(user_content)
    return normalize_for_match(user_content)


def extract_assistant_response_from_chat(message_text: str) -> str:
    if ASSISTANT_TAG in message_text:
        assistant_content = message_text.rsplit(ASSISTANT_TAG, 1)[-1]
    else:
        assistant_content = message_text

    if IM_END_TAG in assistant_content:
        assistant_content = assistant_content.split(IM_END_TAG, 1)[0]

    return assistant_content.strip()


def split_reasoning_and_final_answer(text: str) -> tuple[str, str]:
    stripped = str(text).strip()
    if not stripped:
        return "", ""

    match = re.search(r"(?is)(.*?)(?:\n\n|\n)?final answer\s*[:：]\s*(.+)$", stripped)
    if match:
        return match.group(1).strip(), match.group(2).strip()

    return stripped, ""


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
            letter = next((group for group in match.groups() if group), "").upper()
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


def tokenize_for_overlap(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", normalize_for_match(text))


def token_f1(a: str, b: str) -> float:
    tokens_a = tokenize_for_overlap(a)
    tokens_b = tokenize_for_overlap(b)
    if not tokens_a or not tokens_b:
        return 0.0

    counter_a = Counter(tokens_a)
    counter_b = Counter(tokens_b)
    common = sum((counter_a & counter_b).values())
    if common == 0:
        return 0.0

    precision = common / max(1, sum(counter_b.values()))
    recall = common / max(1, sum(counter_a.values()))
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def sentence_similarity(a: str, b: str) -> float:
    seq = difflib.SequenceMatcher(None, normalize_for_match(a), normalize_for_match(b)).ratio()
    f1 = token_f1(a, b)
    return 0.45 * seq + 0.55 * f1


def is_meta_sentence(sentence: str) -> bool:
    stripped = normalize_text(sentence)
    if len(tokenize_for_overlap(stripped)) <= 3:
        return True
    return any(pattern.search(stripped) for pattern in META_PATTERNS)


def split_sentences(text: str, drop_meta: bool) -> list[str]:
    text = sanitize_reasoning(text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    raw_parts = re.split(r"(?<=[.!?])\s+|(?:\s*\n\s*)+", text)
    sentences: list[str] = []
    for part in raw_parts:
        sentence = normalize_text(part)
        if not sentence:
            continue
        if drop_meta and is_meta_sentence(sentence):
            continue
        sentences.append(sentence)
    return sentences


def greedy_sentence_coverage(reference_sentences: list[str], generated_sentences: list[str]) -> tuple[float, list[float]]:
    if not reference_sentences:
        return 0.0, []
    if not generated_sentences:
        return 0.0, [0.0 for _ in reference_sentences]

    remaining = set(range(len(generated_sentences)))
    matched_scores: list[float] = []
    for ref_sentence in reference_sentences:
        best_idx = None
        best_score = 0.0
        for gen_idx in remaining:
            score = sentence_similarity(ref_sentence, generated_sentences[gen_idx])
            if score > best_score:
                best_score = score
                best_idx = gen_idx

        if best_idx is not None and best_score >= 0.25:
            remaining.remove(best_idx)
            matched_scores.append(best_score)
        else:
            matched_scores.append(0.0)

    coverage = sum(matched_scores) / max(1, len(reference_sentences))
    if coverage == 0.0:
        coverage = sentence_similarity(" ".join(reference_sentences), " ".join(generated_sentences))
    return coverage, matched_scores


def repeated_bigram_penalty(text: str) -> float:
    tokens = tokenize_for_overlap(text)
    if len(tokens) < 4:
        return 0.0
    bigrams = [tuple(tokens[i : i + 2]) for i in range(len(tokens) - 1)]
    if not bigrams:
        return 0.0
    counts = Counter(bigrams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / max(1, len(bigrams))


def format_component(raw_text: str, predicted_choice: str | None) -> tuple[float, dict[str, float]]:
    final_score = 0.0
    breakdown: dict[str, float] = {}
    stripped = raw_text.strip()

    if "final answer:" in stripped.lower():
        breakdown["final_answer_tag"] = 0.4
        final_score += 0.4

    if predicted_choice is not None:
        breakdown["parseable_answer"] = 0.3
        final_score += 0.3

    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    tail = "\n".join(lines[-2:]) if lines else stripped
    if "final answer:" in tail.lower():
        breakdown["answer_at_end"] = 0.3
        final_score += 0.3

    return min(1.0, final_score), breakdown


def consistency_component(reasoning: str, predicted_choice: str | None, choices: list[str]) -> tuple[float, dict[str, float]]:
    if predicted_choice is None:
        return -0.5, {"missing_predicted_choice": -0.5}

    normalized_reasoning = normalize_for_match(reasoning)
    normalized_predicted = normalize_for_match(predicted_choice)
    mentioned_choices = [choice for choice in choices if normalize_for_match(choice) in normalized_reasoning]

    breakdown: dict[str, float] = {}
    if normalized_predicted in normalized_reasoning:
        if len(mentioned_choices) <= 1:
            breakdown["predicted_choice_supported"] = 1.0
            return 1.0, breakdown
        breakdown["predicted_choice_supported_but_competing"] = 0.3
        return 0.3, breakdown

    competing = [choice for choice in mentioned_choices if normalize_for_match(choice) != normalized_predicted]
    if competing:
        breakdown["competing_choice_supported"] = -1.0
        return -1.0, breakdown

    breakdown["weak_consistency"] = 0.0
    return 0.0, breakdown


def length_penalty_component(reasoning: str, reference_word_count: int) -> tuple[float, dict[str, float]]:
    generated_word_count = len(tokenize_for_overlap(reasoning))
    threshold = max(120, int(reference_word_count * 1.6))
    breakdown: dict[str, float] = {}
    penalty = 0.0

    if generated_word_count > threshold:
        overflow_ratio = (generated_word_count - threshold) / max(1, threshold)
        verbosity_penalty = min(1.0, overflow_ratio)
        penalty += verbosity_penalty
        breakdown["verbosity_penalty"] = verbosity_penalty

    repeat_penalty = min(1.0, repeated_bigram_penalty(reasoning) * 3.0)
    if repeat_penalty > 0:
        penalty += repeat_penalty
        breakdown["repeat_penalty"] = repeat_penalty

    return penalty, breakdown


def conflict_penalty_component(raw_text: str, final_answer_span: str, choices: list[str]) -> tuple[float, dict[str, float]]:
    extracted_letters = re.findall(r"\(([A-Z])\)", raw_text.upper())
    valid_letters = {chr(ord("A") + idx) for idx in range(len(choices))}
    unique_letters = {letter for letter in extracted_letters if letter in valid_letters}
    breakdown: dict[str, float] = {}
    penalty = 0.0

    if len(unique_letters) > 2:
        penalty += 0.6
        breakdown["multiple_letters"] = 0.6
    elif len(unique_letters) == 2:
        penalty += 0.25
        breakdown["two_letters"] = 0.25

    normalized_final = normalize_for_match(final_answer_span)
    final_mentions = sum(1 for choice in choices if normalize_for_match(choice) in normalized_final)
    if final_mentions > 1:
        penalty += 0.5
        breakdown["multiple_choices_in_final_answer"] = 0.5

    return penalty, breakdown


def answer_component(predicted_choice: str | None, gold_answer: str) -> tuple[float, dict[str, float]]:
    breakdown: dict[str, float] = {}
    if predicted_choice is None:
        breakdown["unparseable_answer"] = -1.0
        return -1.0, breakdown

    if normalize_for_match(predicted_choice) == normalize_for_match(gold_answer):
        breakdown["correct_answer"] = 1.0
        return 1.0, breakdown

    breakdown["incorrect_answer"] = -0.5
    return -0.5, breakdown


def statistical_reward(
    generated_response: str,
    reference: ReferenceSample,
) -> tuple[float, dict[str, Any]]:
    reasoning, final_answer_span = split_reasoning_and_final_answer(generated_response)
    predicted_choice, matched_by = decode_prediction(generated_response, reference.choices)
    generated_sentences = split_sentences(reasoning, drop_meta=False)
    cot_similarity, sentence_scores = greedy_sentence_coverage(reference.reference_sentences, generated_sentences)

    answer_value, answer_breakdown = answer_component(predicted_choice, reference.answer)
    format_value, format_breakdown = format_component(generated_response, predicted_choice)
    consistency_value, consistency_breakdown = consistency_component(reasoning, predicted_choice, reference.choices)
    length_penalty, length_breakdown = length_penalty_component(reasoning, reference.reference_word_count)
    conflict_penalty, conflict_breakdown = conflict_penalty_component(
        generated_response, final_answer_span, reference.choices
    )

    total = (
        4.0 * answer_value
        + 1.5 * cot_similarity
        + 0.5 * format_value
        + 0.7 * consistency_value
        - 0.5 * length_penalty
        - 0.8 * conflict_penalty
    )

    breakdown = {
        "matched_by": matched_by,
        "predicted_choice": predicted_choice,
        "answer_component": answer_value,
        "cot_similarity": cot_similarity,
        "format_component": format_value,
        "consistency_component": consistency_value,
        "length_penalty": length_penalty,
        "conflict_penalty": conflict_penalty,
        "answer_breakdown": answer_breakdown,
        "format_breakdown": format_breakdown,
        "consistency_breakdown": consistency_breakdown,
        "length_breakdown": length_breakdown,
        "conflict_breakdown": conflict_breakdown,
        "reference_sentence_scores": sentence_scores,
        "generated_sentence_count": len(generated_sentences),
        "reference_sentence_count": len(reference.reference_sentences),
    }
    return total, breakdown


def load_reference_index(input_jsonl: Path) -> dict[str, ReferenceSample]:
    index: dict[str, ReferenceSample] = {}
    with input_jsonl.open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            question = str(row.get("question", row.get("question_text", ""))).strip()
            choices = [str(item) for item in row.get("choices", row.get("multi_choice", []))]
            if not question or not choices:
                continue
            question_prompt = build_question_prompt(question, choices)
            question_key = normalize_for_match(question_prompt)
            answer = str(row["answer"]).strip()
            answer_letter = resolve_answer_letter(answer, choices)
            reference_reasoning = sanitize_reasoning(str(row.get("gemini_cot", "")))
            reference_sentences = split_sentences(reference_reasoning, drop_meta=True)
            reference_word_count = len(tokenize_for_overlap(reference_reasoning))
            index[question_key] = ReferenceSample(
                question_key=question_key,
                question_text=question_prompt,
                answer=answer,
                answer_letter=answer_letter,
                choices=choices,
                reference_reasoning=reference_reasoning,
                reference_sentences=reference_sentences,
                reference_word_count=reference_word_count,
            )
    return index


def score_message(message_text: str, reference_index: dict[str, ReferenceSample]) -> tuple[float, dict[str, Any]]:
    question_key = extract_question_key_from_chat(message_text)
    if question_key not in reference_index:
        return -3.0, {"error": "reference_not_found", "question_key": question_key}

    reference = reference_index[question_key]
    generated_response = extract_assistant_response_from_chat(message_text)
    score, breakdown = statistical_reward(generated_response, reference)
    breakdown["question_key"] = question_key
    breakdown["sample_answer"] = reference.answer
    return score, breakdown
