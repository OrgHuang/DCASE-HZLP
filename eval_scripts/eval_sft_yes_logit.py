#!/usr/bin/env python3
"""
Yes-Logit Inference for DCASE2026 ADQA — Batched Version.

Cross-sample batching: processes K samples in one forward pass by concatenating
all their option prompts and audio. Gives ~5x speedup over per-sample batching.

Pipeline:
1. For each sample, for each option: build Yes/No verification prompt
2. Batch K samples together → single forward pass for all K*N options
3. S_real[i] = log_odds("Yes" | real_audio), S_silence[i] = log_odds("Yes" | silence)
4. S_final[i] = S_real[i] - lambda * S_silence[i]
5. Lambda search + shuffle ensemble
"""

import os, re, csv, sys, json, argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import librosa


# ============================================================================
# 1. Data utilities
# ============================================================================

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


# ============================================================================
# 2. Field parsing
# ============================================================================

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
    for key in ["question", "question_text", "query", "prompt"]:
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
            return str(Path(data_root) / audio)
        if isinstance(audio, dict) and "path" in audio:
            p = str(audio["path"])
            if os.path.isabs(p):
                return p
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
    return [str(choices).strip()]


def get_choices(item: Dict[str, Any]) -> List[str]:
    for key in ["choices", "multi_choice", "options", "choice", "candidate_answers"]:
        if key in item:
            return parse_choices(item[key])
    return []


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


def normalize_text(x: Any) -> str:
    if x is None:
        return ""
    x = str(x).strip().lower()
    x = x.replace("<response>", "").replace("</response>", "")
    x = re.sub(r"^\s*\(?[a-d]\)?[\.\:\-\)]\s*", "", x)
    x = re.sub(r"\s+", " ", x)
    return x.strip().strip("\"'`").strip(" .,:;!?()[]{}")


def is_correct_prediction(prediction: str, gold_answer: str) -> bool:
    return normalize_text(prediction) == normalize_text(gold_answer)


# ============================================================================
# 3. Summary CSV
# ============================================================================

def aggregate_by_question_type(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    stats = {}
    for r in results:
        qt = r.get("question_type", "unknown")
        if qt not in stats:
            stats[qt] = {"total": 0, "correct": 0, "wrong": 0}
        stats[qt]["total"] += 1
        if r.get("correct") is True:
            stats[qt]["correct"] += 1
        elif r.get("correct") is False:
            stats[qt]["wrong"] += 1
    return stats


def write_summary_csv(stats, output_csv: str):
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["question_type", "total", "correct", "wrong", "accuracy"])
        w.writeheader()
        for qt in sorted(stats.keys()):
            s = stats[qt]
            total = s["total"]
            w.writerow({"question_type": qt, "total": total, "correct": s["correct"],
                        "wrong": s["wrong"], "accuracy": f"{s['correct']/total:.4f}" if total else "N/A"})


def write_submission_csv(results, output_csv: str):
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["question", "answer"])
        w.writeheader()
        for r in results:
            w.writerow({"question": r.get("question", ""), "answer": r.get("prediction", "")})


# ============================================================================
# 4. Model loading
# ============================================================================

AUDIO_TEMPLATE = "<|audio_bos|><|AUDIO|><|audio_eos|>"
DEFAULT_S2T_PROMPT = "You are asked to generate text tokens."
YES_NO_TEMPLATE = (
    "Question: {question}\n"
    "Candidate answer: {option}\n"
    "Is this candidate answer correctly supported by the audio?\n"
    "Answer with Yes or No."
)


def load_fun_audio_chat_model(model_path, fun_repo_path, adapter_path=None, dtype="bfloat16"):
    fun_repo_path = str(Path(fun_repo_path).resolve())
    if fun_repo_path not in sys.path:
        sys.path.insert(0, fun_repo_path)

    from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoProcessor
    from funaudiochat.register import register_funaudiochat

    register_funaudiochat()

    torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16

    print("Loading Fun-Audio-Chat processor...")
    processor = AutoProcessor.from_pretrained(model_path)
    print("Loading Fun-Audio-Chat model...")
    config = AutoConfig.from_pretrained(model_path)
    target_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path, config=config, torch_dtype=torch_dtype, low_cpu_mem_usage=True,
    )

    if adapter_path:
        print(f"Loading adapter: {adapter_path}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)

    if target_device.type == "cuda":
        model = model.to(target_device)

    model.eval()
    if hasattr(model, "sp_gen_kwargs"):
        model.sp_gen_kwargs.update({"text_greedy": True, "disable_speech": True})

    return model, processor


# ============================================================================
# 5. Batched Yes-Logit scoring (cross-sample batching)
# ============================================================================

@torch.inference_mode()
def score_batch(
    model,
    processor,
    samples: List[Dict],        # [{"question": str, "options": [str], "audio": np.ndarray}]
    yes_token_id: int,
    no_token_id: int,
) -> List[List[Dict]]:
    """
    Score all options for K samples in ONE forward pass.

    Returns: list of per-sample results, each a list of per-option score dicts.
    """
    texts = []
    audios = []
    option_counts = []  # track how many options per sample

    for s in samples:
        n = len(s["options"])
        option_counts.append(n)
        for opt in s["options"]:
            instruction = YES_NO_TEMPLATE.format(question=s["question"], option=opt)
            conv = [
                {"role": "system", "content": DEFAULT_S2T_PROMPT},
                {"role": "user", "content": AUDIO_TEMPLATE + "\n" + instruction},
            ]
            text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
            texts.append(text)
        audios.extend([s["audio"]] * n)

    # One big forward pass
    inputs = processor(
        text=texts, audio=audios,
        return_tensors="pt", return_token_type_ids=False,
        padding=True,
    ).to(model.device)

    # Fix feature_attention_mask mismatch (WhisperFeatureExtractor off-by-one)
    if "input_features" in inputs and "feature_attention_mask" in inputs:
        if inputs["input_features"].shape[-1] != inputs["feature_attention_mask"].shape[-1]:
            m = min(inputs["input_features"].shape[-1], inputs["feature_attention_mask"].shape[-1])
            inputs["input_features"] = inputs["input_features"][..., :m]
            inputs["feature_attention_mask"] = inputs["feature_attention_mask"][..., :m]

    outputs = model(**inputs, return_dict=True)
    logits = outputs.text_logits  # [total_options, seq_len, vocab]
    last_logits = logits[:, -1, :]

    yes_v = last_logits[:, yes_token_id]
    no_v = last_logits[:, no_token_id]
    log_odds = (yes_v - no_v).cpu().tolist()
    yes_probs = F.softmax(torch.stack([yes_v, no_v], dim=-1), dim=-1)[:, 0].cpu().tolist()
    yes_logits_list = yes_v.cpu().tolist()
    no_logits_list = no_v.cpu().tolist()

    del inputs, outputs, logits, last_logits, yes_v, no_v
    torch.cuda.empty_cache()

    # Split flat results back per-sample
    results = []
    offset = 0
    for count in option_counts:
        sample_results = []
        for j in range(count):
            idx = offset + j
            sample_results.append({
                "log_odds": log_odds[idx],
                "yes_logit": yes_logits_list[idx],
                "no_logit": no_logits_list[idx],
                "yes_prob": yes_probs[idx],
            })
        results.append(sample_results)
        offset += count

    return results


def debias_scores(real_results, silence_results, options, lambda_value):
    """Apply debiasing: S_final[i] = S_real[i] - lambda * S_silence[i]"""
    n = len(options)
    results = []
    for i in range(n):
        s_real = real_results[i]["log_odds"]
        s_silence = silence_results[i]["log_odds"]
        s_final = s_real - lambda_value * s_silence
        results.append({
            "option_text": options[i],
            "option_index": i,
            "S_real": s_real,
            "S_silence": s_silence,
            "S_final": s_final,
            "yes_logit_real": real_results[i]["yes_logit"],
            "yes_logit_silence": silence_results[i]["yes_logit"],
            "no_logit_real": real_results[i]["no_logit"],
            "no_logit_silence": silence_results[i]["no_logit"],
            "yes_prob_real": real_results[i]["yes_prob"],
            "yes_prob_silence": silence_results[i]["yes_prob"],
        })
    best_idx = max(range(n), key=lambda i: results[i]["S_final"])
    sorted_scores = sorted([r["S_final"] for r in results], reverse=True)
    margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) >= 2 else float("inf")
    return {"per_option": results, "predicted_index": best_idx,
            "predicted_text": options[best_idx], "margin": margin}


# ============================================================================
# 6. Main eval loop with cross-sample batching
# ============================================================================

def run_eval(
    model, processor, data, data_root,
    yes_token_id, no_token_id,
    lambda_value=0.0, num_shuffles=1, seed=42,
    eval_batch_size=8, max_audio_seconds=None,
):
    """
    Run Yes-logit evaluation with cross-sample batching.

    eval_batch_size: number of SAMPLES per GPU forward pass (not options).
    """
    results = []
    correct = 0
    total = 0
    max_samples_val = int(max_audio_seconds * 16000) if max_audio_seconds else None

    # Pre-load all audio into memory for speed
    print("Pre-loading audio files...")
    audio_cache = {}
    for item in tqdm(data, desc="Loading audio"):
        try:
            path = get_audio_path(item, data_root)
            if path not in audio_cache:
                wav = librosa.load(path, sr=16000, mono=True)[0]
                if max_samples_val and len(wav) > max_samples_val:
                    wav = wav[:max_samples_val]
                audio_cache[path] = wav
        except Exception:
            audio_cache[get_audio_path(item, data_root)] = None

    # Build sample descriptors
    samples = []
    for item in data:
        try:
            path = get_audio_path(item, data_root)
            real_wav = audio_cache.get(path)
            if real_wav is None:
                raise FileNotFoundError(f"Audio not found: {path}")
            samples.append({
                "item": item,
                "real_wav": real_wav,
                "silence_wav": np.zeros_like(real_wav),
                "question": get_question_text(item),
                "options": get_choices(item),
                "gold": get_gold_answer(item, get_choices(item)),
            })
        except Exception as e:
            samples.append({"item": item, "error": str(e)})

    # Process in micro-batches
    desc = f"Eval (λ={lambda_value}, shuf={num_shuffles})"
    for batch_start in tqdm(range(0, len(samples), eval_batch_size), desc=desc):
        batch_end = min(batch_start + eval_batch_size, len(samples))
        batch_samples = [s for s in samples[batch_start:batch_end] if "error" not in s]
        error_samples = [s for s in samples[batch_start:batch_end] if "error" in s]

        # Handle errors
        for es in error_samples:
            item = es["item"]
            choices = get_choices(item)
            results.append({
                "question": get_question_id(item),
                "question_text": get_question_text(item),
                "question_type": get_question_type(item),
                "source_dataset": item.get("source_dataset", item.get("dataset", "")),
                "audio_file": "",
                "choices": choices,
                "gold_answer": get_gold_answer(item, choices),
                "prediction": "", "correct": False, "margin": 0.0,
                "lambda": lambda_value, "num_shuffles": num_shuffles,
                "score_details": [], "error_type": "runtime_error", "error_message": es["error"],
            })
            total += 1

        if not batch_samples:
            continue

        # --- Real audio: all options for all batch samples in one pass ---
        real_inputs = [{"question": s["question"], "options": s["options"], "audio": s["real_wav"]}
                       for s in batch_samples]

        if num_shuffles > 1:
            # Shuffle ensemble: run multiple shuffles, each with real+silence
            all_debiased = [[] for _ in batch_samples]
            rng = np.random.RandomState(seed + batch_start)
            for shuffle_idx in range(num_shuffles):
                shuffle_inputs_real = []
                shuffle_inputs_silence = []
                shuffle_mappings = []  # per sample: perm list
                for s in batch_samples:
                    n = len(s["options"])
                    perm = list(range(n))
                    if n > 1:
                        rng.shuffle(perm)
                    shuffle_mappings.append(perm)
                    shuffled_opts = [s["options"][p] for p in perm]
                    shuffle_inputs_real.append({"question": s["question"], "options": shuffled_opts, "audio": s["real_wav"]})
                    shuffle_inputs_silence.append({"question": s["question"], "options": shuffled_opts, "audio": s["silence_wav"]})

                real_results = score_batch(model, processor, shuffle_inputs_real, yes_token_id, no_token_id)
                silence_results = score_batch(model, processor, shuffle_inputs_silence, yes_token_id, no_token_id)

                # Map back to original option order
                for bi, (perm, s) in enumerate(zip(shuffle_mappings, batch_samples)):
                    inv_perm = {shuffled_i: orig_i for orig_i, shuffled_i in enumerate(perm)}
                    mapped_real = [real_results[bi][inv_perm[orig_i]] for orig_i in range(len(s["options"]))]
                    mapped_silence = [silence_results[bi][inv_perm[orig_i]] for orig_i in range(len(s["options"]))]
                    d = debias_scores(mapped_real, mapped_silence, s["options"], lambda_value)
                    for opt_result in d["per_option"]:
                        opt_result["S_final_shuffle"] = opt_result["S_final"]
                    all_debiased[bi].append(d)

            # Average scores across shuffles
            for bi, s in enumerate(batch_samples):
                n_opts = len(s["options"])
                avg_scores = {s["options"][i]: [] for i in range(n_opts)}
                s_real_vals = {s["options"][i]: [] for i in range(n_opts)}
                s_silence_vals = {s["options"][i]: [] for i in range(n_opts)}
                for shuffle_result in all_debiased[bi]:
                    for opt in shuffle_result["per_option"]:
                        avg_scores[opt["option_text"]].append(opt["S_final"])
                        s_real_vals[opt["option_text"]].append(opt["S_real"])
                        s_silence_vals[opt["option_text"]].append(opt["S_silence"])

                final_per_option = []
                for i, opt_text in enumerate(s["options"]):
                    final_per_option.append({
                        "option_text": opt_text, "option_index": i,
                        "S_final_avg": np.mean(avg_scores[opt_text]),
                        "S_real_avg": np.mean(s_real_vals[opt_text]),
                        "S_silence_avg": np.mean(s_silence_vals[opt_text]),
                        "S_final_per_shuffle": avg_scores[opt_text],
                    })

                best_text = max(s["options"], key=lambda o: np.mean(avg_scores[o]))
                best_idx = s["options"].index(best_text)
                sorted_avg = sorted([np.mean(avg_scores[o]) for o in s["options"]], reverse=True)
                margin = sorted_avg[0] - sorted_avg[1] if len(sorted_avg) >= 2 else float("inf")

                is_corr = is_correct_prediction(best_text, s["gold"]) if s["gold"] else None
                if is_corr:
                    correct += 1
                total += 1
                item = s["item"]
                results.append({
                    "question": get_question_id(item),
                    "question_text": s["question"],
                    "question_type": get_question_type(item),
                    "source_dataset": item.get("source_dataset", item.get("dataset", "")),
                    "audio_file": get_audio_path(item, data_root),
                    "choices": s["options"],
                    "gold_answer": s["gold"],
                    "prediction": best_text, "correct": is_corr, "margin": margin,
                    "lambda": lambda_value, "num_shuffles": num_shuffles,
                    "score_details": final_per_option,
                    "error_type": "", "error_message": "",
                })
        else:
            # No shuffle ensemble: one pass each for real + silence
            silence_inputs = [{"question": s["question"], "options": s["options"], "audio": s["silence_wav"]}
                              for s in batch_samples]

            real_results = score_batch(model, processor, real_inputs, yes_token_id, no_token_id)
            silence_results = score_batch(model, processor, silence_inputs, yes_token_id, no_token_id)

            for bi, s in enumerate(batch_samples):
                d = debias_scores(real_results[bi], silence_results[bi], s["options"], lambda_value)
                is_corr = is_correct_prediction(d["predicted_text"], s["gold"]) if s["gold"] else None
                if is_corr:
                    correct += 1
                total += 1
                item = s["item"]
                results.append({
                    "question": get_question_id(item),
                    "question_text": s["question"],
                    "question_type": get_question_type(item),
                    "source_dataset": item.get("source_dataset", item.get("dataset", "")),
                    "audio_file": get_audio_path(item, data_root),
                    "choices": s["options"],
                    "gold_answer": s["gold"],
                    "prediction": d["predicted_text"], "correct": is_corr, "margin": d["margin"],
                    "lambda": lambda_value, "num_shuffles": num_shuffles,
                    "score_details": d["per_option"],
                    "error_type": "", "error_message": "",
                })

    acc = correct / total if total > 0 else 0.0
    return results, acc


# ============================================================================
# 7. Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Yes-Logit Batched Eval for Fun-Audio-Chat ADQA")
    parser.add_argument("--model_path", default="/root/model/Fun-Audio-Chat/pretrained_models/Fun-Audio-Chat-8B")
    parser.add_argument("--adapter_path", default="/root/model/Fun-Audio-Chat/training/saves/Fun-Audio-Chat-8B/audio_mcq_enhanced_sft")
    parser.add_argument("--fun_repo_path", default="/root/model/Fun-Audio-Chat")
    parser.add_argument("--input_jsonl", default="/root/data/Audio/datasets/Eval/dev.jsonl")
    parser.add_argument("--data_root", default="/root/data/Audio/datasets/Eval")
    parser.add_argument("--output_dir", default="/root/model/Fun-Audio-Chat/eval_scripts/outputs/yes_logit_v1")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--max_audio_seconds", type=float, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=8,
                        help="Number of samples per GPU forward pass")
    parser.add_argument("--lambda_values", default="0.0,0.3,0.5,0.7,1.0",
                        help="Lambda values to search")
    parser.add_argument("--lambda_search_subset", type=int, default=100,
                        help="Subset for lambda search (0=skip, use full data)")
    parser.add_argument("--num_shuffles", type=int, default=3)
    parser.add_argument("--skip_lambda_search", action="store_true")
    parser.add_argument("--best_lambda", type=float, default=0.0)

    args = parser.parse_args()
    ensure_dir(args.output_dir)

    # Load data
    data = load_jsonl(args.input_jsonl)
    if args.start > 0:
        data = data[args.start:]
    if args.limit > 0:
        data = data[:args.limit]
    print(f"Loaded {len(data)} samples from {args.input_jsonl}")

    # Load model
    model, processor = load_fun_audio_chat_model(
        args.model_path, args.fun_repo_path, args.adapter_path, args.dtype)

    # Get Yes/No token IDs
    yes_tokens = processor.tokenizer.encode("Yes", add_special_tokens=False)
    no_tokens = processor.tokenizer.encode("No", add_special_tokens=False)
    yes_token_id = yes_tokens[0]
    no_token_id = no_tokens[0]
    print(f"'Yes' token: {yes_token_id} -> '{processor.tokenizer.decode([yes_token_id])}'")
    print(f"'No'  token: {no_token_id}  -> '{processor.tokenizer.decode([no_token_id])}'")
    print(f"Eval batch size: {args.eval_batch_size} samples/forward")

    # Lambda search
    best_lambda = args.best_lambda
    lambda_results = []

    if not args.skip_lambda_search and args.lambda_search_subset > 0:
        lambda_candidates = [float(x.strip()) for x in args.lambda_values.split(",")]
        search_data = data
        if args.lambda_search_subset < len(data):
            rng = np.random.RandomState(42)
            indices = rng.choice(len(data), args.lambda_search_subset, replace=False)
            search_data = [data[i] for i in indices]
            print(f"\nLambda search: {len(search_data)}/{len(data)} samples, lambdas={lambda_candidates}")

        best_acc = -1.0
        for lam in lambda_candidates:
            print(f"\n--- Testing lambda={lam} ---")
            _, acc = run_eval(model, processor, search_data, args.data_root,
                              yes_token_id, no_token_id, lambda_value=lam, num_shuffles=1,
                              eval_batch_size=args.eval_batch_size)
            lambda_results.append({"lambda": lam, "accuracy": acc})
            print(f"Lambda {lam}: accuracy={acc:.4f}")
            if acc > best_acc:
                best_acc = acc
                best_lambda = lam

        print(f"\nBest lambda: {best_lambda} (acc={best_acc:.4f})")
        with open(os.path.join(args.output_dir, "lambda_search.json"), "w") as f:
            json.dump({"best_lambda": best_lambda, "results": lambda_results}, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Full eval: lambda={best_lambda}, shuffles={args.num_shuffles}, batch_size={args.eval_batch_size}")
    print(f"{'='*60}")

    results, final_acc = run_eval(
        model, processor, data, args.data_root,
        yes_token_id, no_token_id,
        lambda_value=best_lambda, num_shuffles=args.num_shuffles,
        eval_batch_size=args.eval_batch_size,
        max_audio_seconds=args.max_audio_seconds,
    )

    # Save outputs
    write_jsonl(results, os.path.join(args.output_dir, "predictions.jsonl"))

    score_records = []
    for r in results:
        score_records.append({
            "question": r["question"], "question_text": r["question_text"],
            "question_type": r["question_type"], "gold_answer": r["gold_answer"],
            "predicted_option": r["prediction"], "correct": r["correct"],
            "margin": r["margin"], "lambda": r["lambda"], "num_shuffles": r["num_shuffles"],
            "per_option": r["score_details"],
        })
    write_jsonl(score_records, os.path.join(args.output_dir, "score_details.jsonl"))
    write_submission_csv(results, os.path.join(args.output_dir, "submission.csv"))

    stats = aggregate_by_question_type(results)
    write_summary_csv(stats, os.path.join(args.output_dir, "summary_by_type.csv"))

    total = len(results)
    correct = sum(1 for r in results if r.get("correct") is True)
    wrong = sum(1 for r in results if r.get("correct") is False)
    errors = sum(1 for r in results if r.get("error_type") == "runtime_error")
    acc = correct / total if total > 0 else 0.0

    print("\n" + "=" * 80)
    print("Yes-Logit Evaluation Complete")
    print(f"Best lambda: {best_lambda}  |  Shuffles: {args.num_shuffles}  |  Batch: {args.eval_batch_size}")
    print(f"Total: {total}  |  Correct: {correct}  |  Wrong: {wrong}  |  Errors: {errors}")
    print(f"Accuracy: {acc:.4f}")
    print("=" * 80)

    for qt in sorted(stats.keys()):
        s = stats[qt]
        qt_acc = s["correct"] / s["total"] if s["total"] > 0 else 0.0
        print(f"  {qt}: total={s['total']}, correct={s['correct']}, wrong={s['wrong']}, acc={qt_acc:.4f}")


if __name__ == "__main__":
    main()
