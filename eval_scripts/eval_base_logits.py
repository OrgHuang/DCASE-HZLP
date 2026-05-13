#!/usr/bin/env python3
"""
Base 模型（无 SFT）多选题 Logits 评估脚本。

策略：将多选题拆解为每个选项的 Yes/No 判别问题，
直接提取模型 logits 层中 "Yes" token 的权重，
通过对比所有选项的 "Yes" logit 确定最终答案。
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np
import librosa
from tqdm import tqdm


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


def get_question_text(item: Dict[str, Any]) -> str:
    for key in ["question", "question_text", "query", "prompt"]:
        if key in item and item[key] is not None:
            text = str(item[key]).strip()
            if text:
                return text
    return ""


def get_audio_path(item: Dict[str, Any], data_root: str) -> str:
    p = str(item.get("audio_path", ""))
    if os.path.isabs(p):
        return p
    return str(Path(data_root) / p)


def get_choices(item: Dict[str, Any]) -> List[str]:
    raw = item.get("multi_choice", item.get("choices", []))
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    return []


def get_gold_answer(item: Dict[str, Any]) -> str:
    return str(item.get("answer", "")).strip()


def normalize_text(x: str) -> str:
    return " ".join(str(x).strip().split()).lower()


def is_correct(prediction: str, gold: str) -> bool:
    return normalize_text(prediction) == normalize_text(gold)


# =========================
# 模型加载
# =========================

def load_base_model(model_path: str, fun_repo_path: str, adapter_path: Optional[str] = None, dtype: str = "bfloat16"):
    fun_repo_path = str(Path(fun_repo_path).resolve())
    if fun_repo_path not in sys.path:
        sys.path.insert(0, fun_repo_path)

    from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoProcessor
    from funaudiochat.register import register_funaudiochat
    register_funaudiochat()

    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float16

    processor = AutoProcessor.from_pretrained(model_path)
    # 前向传播用 right padding 即可（非生成任务）
    processor.tokenizer.padding_side = "right"

    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    if adapter_path:
        print(f"Loading SFT adapter: {adapter_path}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model, processor


# =========================
# Token ID 初始化
# =========================

def init_yes_token_ids(processor) -> Tuple[List[int], List[str]]:
    """
    获取代表 "Yes" 的 token IDs。
    同时考虑带前导空格和不带空格的变体，取平均 logit。
    """
    tokenizer = processor.tokenizer
    variants = [
        " Yes",   # 最自然的续写形式（前导空格）
        "Yes",    # 句首形式
    ]
    token_ids = []
    labels = []
    for v in variants:
        ids = tokenizer.encode(v, add_special_tokens=False)
        if len(ids) == 1:
            token_ids.append(ids[0])
            labels.append(v)
            print(f'  "{v}" -> token_id={ids[0]}')
        else:
            print(f'  [SKIP] "{v}" -> multi-token {ids}, not suitable for single-token logit extraction')

    if not token_ids:
        raise ValueError("No suitable single-token 'Yes' found in vocabulary!")

    print(f"Will use {len(token_ids)} Yes token(s) and average their logits: {token_ids}")
    return token_ids, labels


# =========================
# 核心评估函数
# =========================

def build_prompt(question: str, option: str) -> str:
    """
    Base 模型适用的 Yes/No 判别 prompt。
    格式模仿预训练常见文本模式（问答对）。
    """
    return (
        f"Question: {question}\n"
        f"Proposed Answer: {option}\n"
        f"Is the proposed answer correct?"
    )


@torch.inference_mode()
def evaluate_single_mcq(
    model,
    processor,
    audio_path: str,
    question_text: str,
    choices: List[str],
    gold_answer: str,
    yes_token_ids: List[int],
) -> Dict[str, Any]:
    """
    对单个多选题进行 logits 评估。
    将每个选项构造为 Yes/No 判别 prompt，batch 前向传播，
    提取最后一个 token 对 "Yes" 的 logit。
    """
    device = model.device
    audio = [librosa.load(audio_path, sr=16000)[0]]

    # 为每个选项构造独立的 prompt
    option_texts = []
    for opt in choices:
        prompt = build_prompt(question_text, opt)
        option_texts.append(prompt)

    # 使用对话模板包装（让模型以 assistant 身份回答 Yes/No）
    # 注意：这里不使用 add_generation_prompt，因为我们不需要 generate，
    # 而是直接看最后一个 token 的 next-token logits。
    # 但为了和训练分布对齐，我们仍然用 apply_chat_template 加上 generation prompt，
    # 使最后一个 token 是 assistant 开始前的位置。
    from utils.constant import AUDIO_TEMPLATE

    full_texts = []
    for prompt in option_texts:
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": AUDIO_TEMPLATE + "\n" + prompt},
        ]
        text = processor.apply_chat_template(
            [conversation],
            add_generation_prompt=True,
            tokenize=False,
        )[0]
        full_texts.append(text)

    # batch 处理：同一个音频复制 N 份
    audios = audio * len(choices)
    inputs = processor(
        text=full_texts,
        audio=audios,
        return_tensors="pt",
        padding=True,
        return_token_type_ids=False,
    ).to(device)

    # 前向传播（不使用 generate）
    outputs = model(**inputs)
    logits = outputs.logits  # [batch, seq_len, vocab_size]

    # 找到每个样本实际最后一个非 pad token 的位置
    # attention_mask: [batch, seq_len]
    last_positions = inputs.attention_mask.sum(dim=1) - 1  # [batch]
    batch_size = logits.size(0)

    # 提取每个样本最后一个 token 对所有 Yes 变体的 logits
    yes_logits_list = []
    for i in range(batch_size):
        pos = last_positions[i].item()
        # 取所有 Yes token logits 的平均
        token_logits = [logits[i, pos, tid].item() for tid in yes_token_ids]
        avg_logit = float(np.mean(token_logits))
        yes_logits_list.append(avg_logit)

    # 选择 Yes logit 最大的选项
    best_idx = int(np.argmax(yes_logits_list))
    prediction = choices[best_idx]
    correct = is_correct(prediction, gold_answer)

    # 组装每个选项的详细 logit 信息
    option_details = []
    for idx, (opt, ylogit) in enumerate(zip(choices, yes_logits_list)):
        option_details.append({
            "option_index": idx,
            "option_text": opt,
            "yes_logit": round(ylogit, 4),
            "is_best": idx == best_idx,
        })

    return {
        "predicted_index": best_idx,
        "predicted_answer": prediction,
        "gold_answer": gold_answer,
        "correct": correct,
        "options": option_details,
    }


# =========================
# 主程序
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        default="/root/model/Fun-Audio-Chat/pretrained_models/Fun-Audio-Chat-8B")
    parser.add_argument("--fun_repo_path", type=str,
                        default="/root/model/Fun-Audio-Chat")
    parser.add_argument("--input_jsonl", type=str,
                        default="/root/data/Audio/datasets/Eval/dev.jsonl")
    parser.add_argument("--data_root", type=str,
                        default="/root/data/Audio/datasets/Eval")
    parser.add_argument("--output_dir", type=str,
                        default="/root/model/Fun-Audio-Chat/eval_scripts/outputs_base_logits")
    parser.add_argument("--adapter_path", type=str, default="",
                        help="Optional SFT adapter path. Leave empty for base model.")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    detailed_jsonl = os.path.join(args.output_dir, "predictions.jsonl")
    summary_path = os.path.join(args.output_dir, "summary.json")

    print("=" * 80)
    print("Base Model (No SFT) MCQ Logits Evaluation")
    print(f"Model : {args.model_path}")
    print(f"Data  : {args.input_jsonl}")
    print(f"Output: {args.output_dir}")
    print("=" * 80)

    data = load_jsonl(args.input_jsonl)
    if args.limit > 0:
        data = data[:args.limit]
    print(f"Loaded {len(data)} samples.")

    print("\nLoading model...")
    model, processor = load_base_model(
        args.model_path,
        args.fun_repo_path,
        adapter_path=args.adapter_path or None,
        dtype=args.dtype,
    )

    print("\nInitializing Yes token IDs...")
    yes_token_ids, yes_labels = init_yes_token_ids(processor)

    results = []
    correct_count = 0

    pbar = tqdm(data, desc="Evaluating")
    for item in pbar:
        qid = str(item.get("id", item.get("_line_id", 0)))
        qtext = get_question_text(item)
        choices = get_choices(item)
        gold = get_gold_answer(item)
        audio_path = get_audio_path(item, args.data_root)

        if len(choices) == 0:
            print(f"[WARN] {qid}: no choices, skip.")
            continue

        try:
            res = evaluate_single_mcq(
                model=model,
                processor=processor,
                audio_path=audio_path,
                question_text=qtext,
                choices=choices,
                gold_answer=gold,
                yes_token_ids=yes_token_ids,
            )
            correct_count += int(res["correct"])
            pbar.set_postfix({"acc": f"{correct_count}/{len(results)+1}"})

            record = {
                "question_id": qid,
                "question_text": qtext,
                "gold_answer": gold,
                "predicted_index": res["predicted_index"],
                "predicted_answer": res["predicted_answer"],
                "correct": res["correct"],
                "yes_logits": res["options"],
            }
            results.append(record)

        except Exception as e:
            print(f"[ERROR] {qid}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "question_id": qid,
                "question_text": qtext,
                "gold_answer": gold,
                "error": repr(e),
                "correct": False,
            })

        # 每 10 条保存一次
        if len(results) % 10 == 0:
            write_jsonl(results, detailed_jsonl)

    write_jsonl(results, detailed_jsonl)

    total = len([r for r in results if "error" not in r])
    accuracy = correct_count / total if total > 0 else 0.0

    summary = {
        "total_samples": total,
        "correct": correct_count,
        "wrong": total - correct_count,
        "accuracy": round(accuracy, 4),
        "yes_token_ids": yes_token_ids,
        "yes_labels": yes_labels,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("Evaluation Finished")
    print(f"Total : {total}")
    print(f"Correct: {correct_count}")
    print(f"Wrong  : {total - correct_count}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Detailed: {detailed_jsonl}")
    print(f"Summary : {summary_path}")
    print("=" * 80)

    # 打印前 3 个样本的详细 logit 作为示例
    print("\n--- Sample Yes-Logits (first 3) ---")
    for r in results[:3]:
        print(f"\nQ: {r['question_text'][:80]}...")
        print(f"Gold: {r['gold_answer']}")
        print(f"Pred: {r.get('predicted_answer', 'N/A')} (correct={r.get('correct', False)})")
        for opt in r.get("yes_logits", []):
            marker = " *" if opt["is_best"] else ""
            print(f"  [{opt['option_index']}] {opt['option_text'][:50]:50} yes_logit={opt['yes_logit']:.4f}{marker}")


if __name__ == "__main__":
    main()
