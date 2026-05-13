#!/usr/bin/env python3
"""Evaluate MCQ SFT model with prompt aligned to training format.

Key difference from eval_sft_devset.py:
- Prompt matches training exactly (no "Only output..." suffix)
- Expects answer format like "(A) answer text"
"""

import os
import re
import csv
import sys
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm


# =========================
# 1. I/O
# =========================

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            item["_line_id"] = line_id
            data.append(item)
    return data


def write_jsonl(data: List[Dict[str, Any]], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


# =========================
# 2. Field parsing (same as before)
# =========================

def get_question_id(item: Dict[str, Any]) -> str:
    for key in ["question_id", "id", "uid", "sample_id"]:
        if key in item and item[key] is not None:
            return str(item[key])
    return f"sample_{item.get('_line_id', 0)}"


def get_question_type(item: Dict[str, Any]) -> str:
    for key in ["question_type", "type", "category", "task_type"]:
        if key in item and item[key] is not None:
            return str(item[key])
    return "unknown"


def get_question_text(item: Dict[str, Any]) -> str:
    for key in ["question_text", "question", "query", "prompt"]:
        if key in item and item[key] is not None:
            text = str(item[key]).strip()
            if text:
                return text
    return ""


def get_audio_path(item: Dict[str, Any], data_root: Optional[str]) -> str:
    if "audio_file" in item and item["audio_file"]:
        return str(item["audio_file"])
    if "audio_path" in item and item["audio_path"]:
        p = str(item["audio_path"])
        if os.path.isabs(p):
            return p
        if data_root is None:
            raise ValueError("Found relative audio_path but --data_root is not provided.")
        return str(Path(data_root) / p)
    if "audio" in item and item["audio"]:
        audio = item["audio"]
        if isinstance(audio, str):
            if os.path.isabs(audio):
                return audio
            if data_root is None:
                raise ValueError("Found relative audio but --data_root is not provided.")
            return str(Path(data_root) / audio)
        if isinstance(audio, dict) and "path" in audio:
            p = str(audio["path"])
            if os.path.isabs(p):
                return p
            if data_root is None:
                raise ValueError("Found relative audio.path but --data_root is not provided.")
            return str(Path(data_root) / p)
    raise KeyError(f"No valid audio path found. Keys: {list(item.keys())}")


def parse_choices(choices: Any) -> List[str]:
    if choices is None:
        return []
    if isinstance(choices, list):
        return [str(x).strip() for x in choices if str(x).strip()]
    if isinstance(choices, dict):
        keys = sorted(choices.keys())
        return [str(choices[k]).strip() for k in keys if str(choices[k]).strip()]
    if isinstance(choices, str):
        s = choices.strip()
        try:
            obj = json.loads(s)
            if isinstance(obj, list):
                return [str(x).strip() for x in obj if str(x).strip()]
            if isinstance(obj, dict):
                keys = sorted(obj.keys())
                return [str(obj[k]).strip() for k in keys if str(obj[k]).strip()]
        except Exception:
            pass
        parts = re.split(r"\s+[A-Da-d][\.\)]\s+", s)
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) >= 2:
            return parts
        if "\n" in s:
            parts = [p.strip() for p in s.split("\n") if p.strip()]
            if len(parts) >= 2:
                return parts
        if "||" in s:
            parts = [p.strip() for p in s.split("||") if p.strip()]
            if len(parts) >= 2:
                return parts
        return [s]
    return [str(choices).strip()]


def get_choices(item: Dict[str, Any]) -> List[str]:
    for key in ["choices", "multi_choice", "options", "choice", "candidate_answers"]:
        if key in item:
            return parse_choices(item[key])
    return []


def normalize_text(x: Any) -> str:
    if x is None:
        return ""
    x = str(x).strip().lower()
    x = x.replace("<response>", "").replace("</response>", "")
    x = x.replace("<think>", "").replace("</think>", "")
    x = re.sub(r"^\s*\(?[a-d]\)?[\.\:\-\)]\s*", "", x)
    x = re.sub(r"\s+", " ", x)
    x = x.strip()
    x = x.strip("\"'`")
    x = x.strip(" .,:;!?()[]{}")
    return x


def normalize_gold_answer(answer: Any, choices: List[str]) -> str:
    ans = str(answer).strip()
    if len(ans) == 1 and ans.upper() in ["A", "B", "C", "D"]:
        idx = ord(ans.upper()) - ord("A")
        if 0 <= idx < len(choices):
            return choices[idx]
    m = re.match(r"^\s*\(?([A-Da-d])\)?\s*[\.\:\-\)]\s*(.+)$", ans)
    if m:
        idx = ord(m.group(1).upper()) - ord("A")
        stripped = m.group(2).strip()
        if stripped:
            return stripped
        if 0 <= idx < len(choices):
            return choices[idx]
    return ans


def get_gold_answer(item: Dict[str, Any], choices: List[str]) -> str:
    for key in ["answer", "label", "gold", "target", "correct_answer"]:
        if key in item and item[key] is not None:
            return normalize_gold_answer(item[key], choices)
    return ""


# =========================
# 3. Prompt — ALIGNED WITH TRAINING FORMAT
# =========================

def letter_name(i: int) -> str:
    return chr(ord("A") + i)


def build_fun_prompt(question: str, choices: List[str]) -> str:
    """Exact training format from convert_audio_mcq_sft.py."""
    lines = [f"{question} Choose the correct option from the following options:"]
    for i, c in enumerate(choices):
        lines.append(f"({letter_name(i)}) {c}")
    return "\n".join(lines)


# =========================
# 4. Answer extraction — enhanced for training format
# =========================

def extract_answer_from_response(
    response: str,
    choices: List[str],
    fuzzy_threshold: float = 0.72,
) -> Tuple[str, str]:
    raw = str(response).strip()

    if raw == "":
        return "", "no_prediction"

    # Strip <think>...</think> blocks
    raw_clean = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
    if raw_clean == "":
        return "", "no_prediction"

    norm_raw = normalize_text(raw_clean)
    if norm_raw == "":
        return "", "no_prediction"

    # 1. Parse "(A) answer text" format — training answer format
    m = re.match(r"^\s*\(([A-Da-d])\)\s*(.+)$", raw_clean, re.DOTALL)
    if m and len(choices) <= 4:
        idx = ord(m.group(1).upper()) - ord("A")
        if 0 <= idx < len(choices):
            return choices[idx], "ok"

    # 2. Only A/B/C/D
    m = re.match(r"^\s*(?:answer\s*[:：]\s*)?\(?([A-Da-d])\)?[\.\)]?\s*$", raw_clean)
    if m and len(choices) <= 4:
        idx = ord(m.group(1).upper()) - ord("A")
        if 0 <= idx < len(choices):
            return choices[idx], "ok"

    # 3. Output option A / choice B
    m = re.search(r"\b(?:option|choice|answer)\s*[:：]?\s*\(?([A-Da-d])\)?\b", raw_clean, flags=re.I)
    if m and len(choices) <= 4:
        idx = ord(m.group(1).upper()) - ord("A")
        if 0 <= idx < len(choices):
            return choices[idx], "ok"

    norm_choices = [normalize_text(c) for c in choices]

    # 4. Exact match
    for choice, norm_choice in zip(choices, norm_choices):
        if norm_choice and norm_raw == norm_choice:
            return choice, "ok"

    # 5. Response contains choice text
    contained = []
    for idx, norm_choice in enumerate(norm_choices):
        if norm_choice and norm_choice in norm_raw:
            contained.append((idx, len(norm_choice)))
    if contained:
        best_idx = sorted(contained, key=lambda x: x[1], reverse=True)[0][0]
        return choices[best_idx], "ok"

    # 6. Choice contains response
    for idx, norm_choice in enumerate(norm_choices):
        if norm_raw and norm_raw in norm_choice:
            return choices[idx], "ok"

    # 7. Fuzzy matching
    try:
        from difflib import SequenceMatcher
        scores = [SequenceMatcher(None, norm_raw, nc).ratio() for nc in norm_choices]
        if scores:
            best_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
            if scores[best_idx] >= fuzzy_threshold:
                return choices[best_idx], "ok"
    except Exception:
        pass

    return raw_clean, "parse_failure"


def is_correct_prediction(prediction: str, gold_answer: str) -> bool:
    return normalize_text(prediction) == normalize_text(gold_answer)


# =========================
# 5. Summary
# =========================

def init_stat_row() -> Dict[str, Any]:
    return {
        "total": 0, "correct": 0, "wrong": 0,
        "wrong_choice_errors": 0, "parse_failures": 0,
        "no_prediction_errors": 0, "runtime_errors": 0,
    }


def aggregate_by_question_type(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    stats = {}
    for r in results:
        qt = r.get("question_type", "unknown")
        if qt not in stats:
            stats[qt] = init_stat_row()
        stats[qt]["total"] += 1
        error_type = r.get("error_type", "")
        if error_type == "runtime_error":
            stats[qt]["runtime_errors"] += 1
            stats[qt]["wrong"] += 1
            continue
        if error_type == "no_prediction":
            stats[qt]["no_prediction_errors"] += 1
            stats[qt]["wrong"] += 1
            continue
        if error_type == "parse_failure":
            stats[qt]["parse_failures"] += 1
            stats[qt]["wrong"] += 1
            continue
        correct = r.get("correct", None)
        if correct is True:
            stats[qt]["correct"] += 1
        elif correct is False:
            stats[qt]["wrong"] += 1
            stats[qt]["wrong_choice_errors"] += 1
    return stats


def write_summary_csv(stats: Dict[str, Dict[str, Any]], output_csv: str):
    fieldnames = [
        "question_type", "total", "correct", "wrong", "accuracy",
        "wrong_choice_errors", "parse_failures", "no_prediction_errors", "runtime_errors",
    ]
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for qt in sorted(stats.keys()):
            s = stats[qt]
            total = s["total"]
            correct = s["correct"]
            accuracy = correct / total if total > 0 else 0.0
            writer.writerow({
                "question_type": qt, "total": total, "correct": correct,
                "wrong": s["wrong"], "accuracy": f"{accuracy:.4f}",
                "wrong_choice_errors": s["wrong_choice_errors"],
                "parse_failures": s["parse_failures"],
                "no_prediction_errors": s["no_prediction_errors"],
                "runtime_errors": s["runtime_errors"],
            })


def write_submission_csv(results: List[Dict[str, Any]], output_csv: str):
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "answer"])
        writer.writeheader()
        for r in results:
            writer.writerow({"question": r.get("question", ""), "answer": r.get("prediction", "")})


# =========================
# 6. Model loading & inference
# =========================

def load_fun_audio_chat_model(
    model_path: str,
    fun_repo_path: str,
    adapter_path: Optional[str] = None,
    dtype: str = "bfloat16",
):
    fun_repo_path = str(Path(fun_repo_path).resolve())
    if fun_repo_path not in sys.path:
        sys.path.insert(0, fun_repo_path)

    from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoProcessor
    from funaudiochat.register import register_funaudiochat
    register_funaudiochat()

    if dtype == "float16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.bfloat16

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(model_path)

    print("Loading config...")
    config = AutoConfig.from_pretrained(model_path)

    print("Loading model...")
    target_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path, config=config, torch_dtype=torch_dtype, low_cpu_mem_usage=True,
    )

    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)

    if target_device.type == "cuda":
        model = model.to(target_device)

    model.eval()
    if hasattr(model, "sp_gen_kwargs"):
        model.sp_gen_kwargs.update({"text_greedy": True, "disable_speech": True})

    return model, processor


@torch.inference_mode()
def infer_one(model, processor, audio_path: str, instruction: str, max_new_tokens: int = 64) -> str:
    from utils.constant import DEFAULT_S2T_PROMPT, AUDIO_TEMPLATE
    import librosa

    audio = [librosa.load(audio_path, sr=16000)[0]]
    conversation = [
        {"role": "system", "content": DEFAULT_S2T_PROMPT},
        {"role": "user", "content": AUDIO_TEMPLATE + "\n" + instruction},
    ]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, audio=audio, return_tensors="pt", return_token_type_ids=False).to(model.device)

    try:
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    except TypeError:
        outputs = model.generate(**inputs)

    if isinstance(outputs, tuple):
        generate_ids = outputs[0]
    else:
        generate_ids = outputs
    generate_ids = generate_ids[:, inputs.input_ids.size(1):]
    return processor.decode(generate_ids[0], skip_special_tokens=True).strip()


# =========================
# 7. Main
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        default="/root/model/Fun-Audio-Chat/pretrained_models/Fun-Audio-Chat-8B")
    parser.add_argument("--adapter_path", type=str,
                        default="/root/model/Fun-Audio-Chat/training/saves/Fun-Audio-Chat-8B/audio_mcq_enhanced_on_binary")
    parser.add_argument("--fun_repo_path", type=str, default="/root/model/Fun-Audio-Chat")
    parser.add_argument("--input_jsonl", type=str, default="/root/data/Audio/datasets/Eval/dev.jsonl")
    parser.add_argument("--data_root", type=str, default="/root/data/Audio/datasets/Eval")
    parser.add_argument("--output_dir", type=str,
                        default="/root/model/Fun-Audio-Chat/eval_scripts/outputs_enhanced_on_binary_v2")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])

    args = parser.parse_args()
    ensure_dir(args.output_dir)

    detailed_jsonl = os.path.join(args.output_dir, "predictions.jsonl")
    submission_csv = os.path.join(args.output_dir, "submission.csv")
    summary_csv = os.path.join(args.output_dir, "summary_by_type.csv")
    errors_jsonl = os.path.join(args.output_dir, "errors.jsonl")

    print("=" * 80)
    print("MCQ SFT Eval V2 — Prompt aligned with training")
    print(f"Adapter: {args.adapter_path}")
    print(f"Data   : {args.input_jsonl}")
    print("=" * 80)

    data = load_jsonl(args.input_jsonl)
    if args.start > 0:
        data = data[args.start:]
    if args.limit > 0:
        data = data[:args.limit]
    print(f"Loaded samples: {len(data)}")

    model, processor = load_fun_audio_chat_model(
        model_path=args.model_path,
        fun_repo_path=args.fun_repo_path,
        adapter_path=args.adapter_path or None,
        dtype=args.dtype,
    )

    results = []
    error_cases = []

    for item in tqdm(data, desc="Inferencing"):
        question_id = get_question_id(item)
        question_type = get_question_type(item)
        try:
            question_text = get_question_text(item)
            choices = get_choices(item)
            audio_path = get_audio_path(item, args.data_root)

            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio not found: {audio_path}")
            if question_text == "":
                raise ValueError("Empty question.")
            if len(choices) == 0:
                raise ValueError("No choices.")

            instruction = build_fun_prompt(question_text, choices)
            raw_response = infer_one(model, processor, audio_path, instruction, max_new_tokens=args.max_new_tokens)
            prediction, parse_status = extract_answer_from_response(raw_response, choices)
            gold_answer = get_gold_answer(item, choices)

            if parse_status == "no_prediction":
                correct = False if gold_answer else None
                error_type = "no_prediction"
            elif parse_status == "parse_failure":
                correct = False if gold_answer else None
                error_type = "parse_failure"
            else:
                correct = is_correct_prediction(prediction, gold_answer) if gold_answer else None
                error_type = ""

            result = {
                "question": question_id, "question_text": question_text,
                "question_type": question_type,
                "source_dataset": item.get("source_dataset", item.get("dataset", "")),
                "audio_file": audio_path, "choices": choices,
                "gold_answer": gold_answer, "prompt": instruction,
                "raw_response": raw_response, "prediction": prediction,
                "parse_status": parse_status, "correct": correct,
                "error_type": error_type, "error_message": "",
            }
            results.append(result)
            if error_type:
                error_cases.append(result)

        except Exception as e:
            result = {
                "question": question_id, "question_text": get_question_text(item),
                "question_type": question_type,
                "source_dataset": item.get("source_dataset", item.get("dataset", "")),
                "audio_file": "", "choices": get_choices(item),
                "gold_answer": "", "prompt": "", "raw_response": "",
                "prediction": "", "parse_status": "", "correct": False,
                "error_type": "runtime_error", "error_message": repr(e),
            }
            results.append(result)
            error_cases.append(result)

        if len(results) % 50 == 0:
            write_jsonl(results, detailed_jsonl)
            write_jsonl(error_cases, errors_jsonl)

    write_jsonl(results, detailed_jsonl)
    write_jsonl(error_cases, errors_jsonl)
    write_submission_csv(results, submission_csv)

    stats = aggregate_by_question_type(results)
    write_summary_csv(stats, summary_csv)

    total = len(results)
    correct = sum(1 for r in results if r.get("correct") is True)
    wrong = sum(1 for r in results if r.get("correct") is False)
    acc = correct / total if total > 0 else 0.0

    print("\n" + "=" * 80)
    print("Inference finished.")
    print(f"Total samples : {total}")
    print(f"Correct       : {correct}")
    print(f"Wrong         : {wrong}")
    print(f"Accuracy      : {acc:.4f}")
    print("=" * 80)

    print("\nSummary by question_type:")
    for qt in sorted(stats.keys()):
        s = stats[qt]
        qt_acc = s["correct"] / s["total"] if s["total"] > 0 else 0.0
        print(f"{qt}: total={s['total']}, correct={s['correct']}, wrong={s['wrong']}, "
              f"accuracy={qt_acc:.4f}, parse_failures={s['parse_failures']}, "
              f"no_prediction={s['no_prediction_errors']}, runtime={s['runtime_errors']}")


if __name__ == "__main__":
    main()
