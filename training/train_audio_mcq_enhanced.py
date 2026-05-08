#!/usr/bin/env python
"""
Enhanced AudioMCQ SFT for Fun-Audio-Chat with multi-view data augmentation.

Inspired by StepAudio2's counterfactual audio training:
- positive_sft: original audio + question + answer (standard CE loss)
- positive_score: original audio + question + answer (sequence score only)
- silence_score: silent audio + question + answer (sequence score only)
- mismatch_score: mismatched audio + question + answer (sequence score only)
- permuted_score: original audio + shuffled choices + answer (sequence score only)

Margin losses force the model to assign higher likelihood to the correct answer
when given real audio vs. counterfactual audio.
"""

import argparse
import json
import math
import os
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torch.nn.functional as F
import librosa
import soundfile as sf
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    set_seed,
    BitsAndBytesConfig,
)

# Register FunAudioChat so AutoModel / AutoProcessor can find it.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from funaudiochat.register import register_funaudiochat

register_funaudiochat()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IGNORE_INDEX = -100
TOKEN_FPS = 25
AUDIO_PAD_TOKEN = "<|audio_pad|>"
AUDIO_BOS_TOKEN = "<|audio_bos|>"
AUDIO_EOS_TOKEN = "<|audio_eos|>"
AUDIO_TEMPLATE = "<|audio_bos|><|AUDIO|><|audio_eos|>"

VIEW_POSITIVE_SFT = "positive_sft"
VIEW_POSITIVE_SCORE = "positive_score"
VIEW_SILENCE_SCORE = "silence_score"
VIEW_MISMATCH_SCORE = "mismatch_score"
VIEW_PERMUTED_SCORE = "permuted_score"

DEFAULT_SYSTEM_PROMPT = "You are asked to generate text tokens."


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def get_audio_duration_seconds(audio_path: str) -> float:
    """Fast duration lookup without loading the full waveform."""
    try:
        info = sf.info(audio_path)
        if info.frames and info.samplerate:
            return float(info.frames) / float(info.samplerate)
    except Exception:
        pass
    try:
        return float(librosa.get_duration(path=audio_path))
    except Exception:
        return 0.0


def load_audio_waveform(audio_path: str, target_rate: int = 16000, max_length: Optional[int] = None):
    """Load audio and return a 1-D numpy array."""
    audio_np, sr = librosa.load(audio_path, sr=target_rate, mono=True)
    if max_length is not None and audio_np.shape[0] > max_length:
        audio_np = audio_np[:max_length]
    return audio_np


def format_question(question: str, choices: List[str]) -> str:
    lines = [f"{question} Choose the correct option from the following options:"]
    for idx, choice in enumerate(choices):
        lines.append(f"({chr(ord('A') + idx)}) {choice}")
    return "\n".join(lines)


def parse_question_and_choices(user_content: str):
    """Parse question text and choices from Fun-Audio-Chat user content."""
    # Strip the audio placeholder
    content = user_content.replace(AUDIO_TEMPLATE, "").strip()
    if content.startswith("\n"):
        content = content[1:]

    lines = content.split("\n")
    question_lines = []
    choices = []
    in_choices = False

    for line in lines:
        line_stripped = line.strip()
        if line_stripped.startswith("Question:"):
            question_lines.append(line_stripped[len("Question:"):].strip())
        elif line_stripped == "Choices:":
            in_choices = True
        elif in_choices and line_stripped.startswith("(") and ")" in line_stripped:
            choice_text = line_stripped.split(")", 1)[1].strip()
            choices.append(choice_text)
        elif line_stripped == "Answer:" or line_stripped.startswith("Answer:"):
            break
        elif not in_choices and line_stripped:
            question_lines.append(line_stripped)

    question = " ".join(question_lines)
    return question, choices


def build_answer_text(answer: str, choices: List[str], answer_format: str = "letter_and_text") -> str:
    """Build the final answer string matching the dataset format."""
    answer = str(answer).strip()
    answer_idx = None
    for i, c in enumerate(choices):
        if str(c).strip() == answer:
            answer_idx = i
            break

    if answer_idx is None:
        # Fallback: if answer itself looks like a letter
        match = re.match(r"^\(([A-D])\)", answer)
        if match:
            letter = match.group(1)
            text = answer.split(")", 1)[1].strip() if ")" in answer else answer
            if answer_format == "letter":
                return f"({letter})"
            elif answer_format == "text":
                return text
            return f"({letter}) {text}"
        return answer

    letter = chr(ord("A") + answer_idx)
    if answer_format == "letter":
        return f"({letter})"
    elif answer_format == "text":
        return answer
    return f"({letter}) {answer}"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class AudioMCQEnhancedDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        processor,
        max_samples: Optional[int] = None,
        max_audio_seconds: Optional[float] = None,
        seed: int = 42,
        use_counterfactual_audio: bool = False,
        use_silence_view: bool = True,
        use_mismatch_view: bool = True,
        use_permuted_view: bool = False,
    ):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_audio_samples = int(max_audio_seconds * 16000) if max_audio_seconds else None
        self.seed = seed
        self.use_counterfactual_audio = use_counterfactual_audio
        self.use_silence_view = use_counterfactual_audio and use_silence_view
        self.use_mismatch_view = use_counterfactual_audio and use_mismatch_view
        self.use_permuted_view = use_permuted_view
        self.group_size = getattr(processor, "audio_group_size", 5)

        self.rows = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                self.rows.append(json.loads(line))
                if max_samples and len(self.rows) >= max_samples:
                    break

        if not self.rows:
            raise RuntimeError(f"No training rows found in {data_path}")

        # Extract audio paths and pre-compute durations for fast indexing
        self.audio_paths = []
        self.audio_durations = []
        for row in self.rows:
            audio_json = json.loads(row["audio"][0])
            path = audio_json["path"]
            self.audio_paths.append(path)
            self.audio_durations.append(get_audio_duration_seconds(path))

        # Build mismatch lookup table
        self.mismatch_indices = self._build_mismatch_indices()

        print(
            f"[Dataset] Loaded {len(self.rows)} rows. "
            f"Counterfactual={use_counterfactual_audio}, "
            f"Silence={self.use_silence_view}, "
            f"Mismatch={self.use_mismatch_view}, "
            f"Permuted={self.use_permuted_view}"
        )

    def _build_mismatch_indices(self):
        if len(self.rows) < 2:
            return [None] * len(self.rows)

        by_source_and_type = defaultdict(list)
        by_question_type = defaultdict(list)
        all_indices = list(range(len(self.rows)))

        for idx, row in enumerate(self.rows):
            by_source_and_type[(row.get("source_dataset"), row.get("question_type"))].append(idx)
            by_question_type[row.get("question_type")].append(idx)

        mismatch_indices = []
        for idx, row in enumerate(self.rows):
            candidates = [
                by_source_and_type[(row.get("source_dataset"), row.get("question_type"))],
                by_question_type[row.get("question_type")],
                all_indices,
            ]
            mismatch_indices.append(self._pick_different_index(idx, candidates))
        return mismatch_indices

    @staticmethod
    def _pick_different_index(idx: int, candidate_groups: List[List[int]]):
        for group in candidate_groups:
            if not group:
                continue
            if len(group) == 1 and group[0] == idx:
                continue
            if idx in group:
                pos = group.index(idx)
                return group[(pos + 1) % len(group)]
            return group[0]
        return None

    def _make_audio_token_str(self, duration: float) -> str:
        num_pad_tokens = max(1, int(duration * TOKEN_FPS))
        return AUDIO_PAD_TOKEN * num_pad_tokens

    def _expand_audio_in_text(self, text: str, num_grouped_tokens: int) -> str:
        expanded = f"<|audio_bos|>{ '<|AUDIO|>' * num_grouped_tokens }<|audio_eos|>"
        return text.replace(AUDIO_TEMPLATE, expanded)

    def _tokenize_with_labels(self, prompt_only_messages, full_messages, answer_text):
        """
        Build input_ids and labels by separating prompt from answer.

        Uses prompt_only_messages (system+user only) with add_generation_prompt=True
        to obtain the exact prompt prefix, and full_messages with
        add_generation_prompt=False to obtain the full sequence.

        For Qwen3-style templates this yields full_text.startswith(prompt_text),
        so the label boundary is simply len(prompt_ids).

        Returns (input_ids, labels, score_labels).
        """
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_only_messages, add_generation_prompt=True, tokenize=False
        )
        full_text = self.tokenizer.apply_chat_template(
            full_messages, add_generation_prompt=False, tokenize=False
        )

        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]

        # Qwen3 template guarantees full_text starts with prompt_text.
        # Verify and fall back to answer subsequence search if not.
        if full_text.startswith(prompt_text):
            answer_start = len(prompt_ids)
        else:
            answer_ids = self.tokenizer(answer_text, add_special_tokens=False)["input_ids"]
            answer_start = None
            for i in range(len(full_ids) - len(answer_ids), -1, -1):
                if full_ids[i : i + len(answer_ids)] == answer_ids:
                    answer_start = i
                    break
            if answer_start is None:
                common_len = 0
                for a, b in zip(prompt_ids, full_ids):
                    if a == b:
                        common_len += 1
                    else:
                        break
                answer_start = common_len

        labels = [IGNORE_INDEX] * answer_start + full_ids[answer_start:]
        score_labels = [IGNORE_INDEX] * answer_start + full_ids[answer_start:]
        return full_ids, labels, score_labels

    def _build_view(
        self,
        row: Dict,
        audio_wav: np.ndarray,
        audio_token_str: str,
        user_content: str,
        answer_text: str,
        view_name: str,
        learn_target: bool = True,
    ):
        # Number of grouped audio tokens for expanding <|AUDIO|>
        num_pad_tokens = max(1, len(audio_token_str) // len(AUDIO_PAD_TOKEN))
        num_grouped = (num_pad_tokens + self.group_size - 1) // self.group_size

        expanded_user = self._expand_audio_in_text(user_content, num_grouped)
        system = row.get("system", DEFAULT_SYSTEM_PROMPT)

        # prompt_only: system + user  (no assistant message)
        prompt_only_messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": expanded_user},
        ]
        full_messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": expanded_user},
            {"role": "assistant", "content": answer_text},
        ]

        input_ids, labels, score_labels = self._tokenize_with_labels(
            prompt_only_messages, full_messages, answer_text
        )

        if not learn_target:
            labels = [IGNORE_INDEX] * len(input_ids)

        return {
            "view_name": view_name,
            "input_ids": input_ids,
            "labels": labels,
            "score_labels": score_labels,
            "audio_wav": audio_wav,
            "audio_token_str": audio_token_str,
        }

    def _build_permuted_choices(self, idx: int, choices: List[str]) -> List[str]:
        if len(choices) <= 1:
            return list(choices)
        rng = random.Random(self.seed + idx)
        for _ in range(5):
            permuted = list(choices)
            rng.shuffle(permuted)
            if permuted != list(choices):
                return permuted
        return list(choices[1:]) + [choices[0]]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        user_content = row["messages"][0]["content"]
        answer_text = row["messages"][1]["content"]
        duration = self.audio_durations[idx]
        audio_token_str = self._make_audio_token_str(duration)
        audio_wav = load_audio_waveform(self.audio_paths[idx], max_length=self.max_audio_samples)

        views = []

        # 1. Positive SFT view (always present)
        views.append(
            self._build_view(
                row, audio_wav, audio_token_str, user_content, answer_text,
                VIEW_POSITIVE_SFT, learn_target=True,
            )
        )

        # 2. Positive score view (baseline for margins)
        if self.use_counterfactual_audio or self.use_permuted_view:
            views.append(
                self._build_view(
                    row, audio_wav, audio_token_str, user_content, answer_text,
                    VIEW_POSITIVE_SCORE, learn_target=False,
                )
            )

        # 3. Silence score view
        if self.use_silence_view:
            silence_wav = np.zeros_like(audio_wav)
            silence_token_str = audio_token_str  # same length to keep placeholder count consistent
            views.append(
                self._build_view(
                    row, silence_wav, silence_token_str, user_content, answer_text,
                    VIEW_SILENCE_SCORE, learn_target=False,
                )
            )

        # 4. Mismatch score view
        if self.use_mismatch_view and self.mismatch_indices[idx] is not None:
            mismatch_idx = self.mismatch_indices[idx]
            mismatch_wav = load_audio_waveform(
                self.audio_paths[mismatch_idx], max_length=self.max_audio_samples
            )
            mismatch_duration = self.audio_durations[mismatch_idx]
            mismatch_token_str = self._make_audio_token_str(mismatch_duration)
            views.append(
                self._build_view(
                    row, mismatch_wav, mismatch_token_str, user_content, answer_text,
                    VIEW_MISMATCH_SCORE, learn_target=False,
                )
            )

        # 5. Permuted choices score view
        if self.use_permuted_view:
            question, choices = parse_question_and_choices(user_content)
            if len(choices) > 1:
                permuted_choices = self._build_permuted_choices(idx, choices)
                permuted_user = f"{AUDIO_TEMPLATE}\n{format_question(question, permuted_choices)}\nAnswer:"
                views.append(
                    self._build_view(
                        row, audio_wav, audio_token_str, permuted_user, answer_text,
                        VIEW_PERMUTED_SCORE, learn_target=False,
                    )
                )

        return {"views": views}


# ---------------------------------------------------------------------------
# Data Collator
# ---------------------------------------------------------------------------
@dataclass
class EnhancedDataCollator:
    processor: object

    def __call__(self, features: List[Dict]):
        flat_features = []
        view_indices = {
            VIEW_POSITIVE_SCORE: [],
            VIEW_SILENCE_SCORE: [],
            VIEW_MISMATCH_SCORE: [],
            VIEW_PERMUTED_SCORE: [],
        }

        for sample in features:
            sample_views = sample["views"]
            local_indices = {k: -1 for k in view_indices}
            for view in sample_views:
                flat_idx = len(flat_features)
                flat_features.append(view)
                if view["view_name"] in local_indices:
                    local_indices[view["view_name"]] = flat_idx

            for k in view_indices:
                view_indices[k].append(local_indices[k])

        pad_id = self.processor.tokenizer.pad_token_id
        max_len = max(len(x["input_ids"]) for x in flat_features)

        input_ids, labels, score_labels, attention_mask = [], [], [], []
        for feat in flat_features:
            ids = torch.tensor(feat["input_ids"], dtype=torch.long)
            labs = torch.tensor(feat["labels"], dtype=torch.long)
            score_labs = torch.tensor(feat["score_labels"], dtype=torch.long)
            pad_len = max_len - ids.numel()

            input_ids.append(F.pad(ids, (0, pad_len), value=pad_id))
            labels.append(F.pad(labs, (0, pad_len), value=IGNORE_INDEX))
            score_labels.append(F.pad(score_labs, (0, pad_len), value=IGNORE_INDEX))
            attention_mask.append(
                F.pad(torch.ones_like(ids, dtype=torch.long), (0, pad_len), value=0)
            )

        batch = {
            "input_ids": torch.stack(input_ids),
            "labels": torch.stack(labels),
            "score_labels": torch.stack(score_labels),
            "attention_mask": torch.stack(attention_mask),
            "positive_score_indices": torch.tensor(view_indices[VIEW_POSITIVE_SCORE], dtype=torch.long),
            "silence_score_indices": torch.tensor(view_indices[VIEW_SILENCE_SCORE], dtype=torch.long),
            "mismatch_score_indices": torch.tensor(view_indices[VIEW_MISMATCH_SCORE], dtype=torch.long),
            "permuted_score_indices": torch.tensor(view_indices[VIEW_PERMUTED_SCORE], dtype=torch.long),
        }

        # Continuous audio features (Mel spectrograms)
        audio_wavs = [feat["audio_wav"] for feat in flat_features]
        wav_inputs = self.processor.feature_extractor(
            audio_wavs,
            sampling_rate=getattr(self.processor, "audio_sampling_rate", 16000),
            return_attention_mask=True,
            padding=True,
            return_tensors="pt",
        )
        input_features = wav_inputs["input_features"]
        feature_attention_mask = wav_inputs["attention_mask"]

        # Fix length mismatch between input_features and feature_attention_mask
        # (WhisperFeatureExtractor can produce off-by-one differences on batched inputs)
        input_seq_len = input_features.shape[-1]
        mask_seq_len = feature_attention_mask.shape[-1]
        if input_seq_len != mask_seq_len:
            min_seq_len = min(input_seq_len, mask_seq_len)
            input_features = input_features[..., :min_seq_len]
            feature_attention_mask = feature_attention_mask[..., :min_seq_len]

        batch["input_features"] = input_features
        batch["feature_attention_mask"] = feature_attention_mask
        batch["feature_exist_mask"] = torch.ones(len(flat_features), dtype=torch.bool)

        # Discrete speech tokens
        audio_tokens = [feat["audio_token_str"] for feat in flat_features]
        speech_kwargs = {
            "return_attention_mask": True,
            "return_token_type_ids": False,
            "padding": True,
            "pad_to_multiple_of": getattr(self.processor, "audio_group_size", 5),
            "return_tensors": "pt",
        }
        speech_inputs = self.processor.speech_tokenizer(audio_tokens, **speech_kwargs)
        batch["speech_ids"] = speech_inputs["input_ids"]
        batch["speech_attention_mask"] = speech_inputs["attention_mask"]

        return batch


# ---------------------------------------------------------------------------
# Trainer with margin losses
# ---------------------------------------------------------------------------
class FunAudioChatEnhancedTrainer(Trainer):
    def __init__(
        self,
        *args,
        audio_margin_weight: float = 0.0,
        audio_margin: float = 0.2,
        permutation_loss_weight: float = 0.0,
        permutation_margin: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.audio_margin_weight = audio_margin_weight
        self.audio_margin = audio_margin
        self.permutation_loss_weight = permutation_loss_weight
        self.permutation_margin = permutation_margin

    @staticmethod
    def compute_ce_loss(logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=IGNORE_INDEX,
        )

    @staticmethod
    def compute_sequence_scores(logits, score_labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = score_labels[..., 1:].contiguous()
        token_mask = shift_labels.ne(IGNORE_INDEX)
        safe_labels = shift_labels.masked_fill(~token_mask, 0)
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_scores = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        token_scores = token_scores * token_mask
        denom = token_mask.sum(dim=-1).clamp_min(1)
        return token_scores.sum(dim=-1) / denom

    def margin_loss(self, scores, positive_indices, negative_indices, margin):
        valid = positive_indices.ge(0) & negative_indices.ge(0)
        if not valid.any():
            return None
        positive_scores = scores[positive_indices[valid]]
        negative_scores = scores[negative_indices[valid]]
        return F.relu(margin - positive_scores + negative_scores).mean()

    def consistency_loss(self, scores, reference_indices, target_indices, margin):
        valid = reference_indices.ge(0) & target_indices.ge(0)
        if not valid.any():
            return None
        reference_scores = scores[reference_indices[valid]]
        target_scores = scores[target_indices[valid]]
        return F.relu((reference_scores - target_scores).abs() - margin).mean()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        score_labels = inputs.pop("score_labels")
        positive_score_indices = inputs.pop("positive_score_indices")
        silence_score_indices = inputs.pop("silence_score_indices")
        mismatch_score_indices = inputs.pop("mismatch_score_indices")
        permuted_score_indices = inputs.pop("permuted_score_indices")

        # Determine which samples are positive SFT views (have real labels)
        is_positive = labels.ne(IGNORE_INDEX).any(dim=-1)

        # Forward WITHOUT labels so the model does not compute its own internal
        # loss.  Score-view labels are all IGNORE_INDEX which would make the
        # model's default CE loss NaN (mean over zero valid positions).
        outputs = model(**inputs, return_dict=True)
        logits = outputs.text_logits

        # CE loss only on positive SFT views
        ce_loss = (
            self.compute_ce_loss(logits[is_positive], labels[is_positive])
            if is_positive.any()
            else torch.tensor(0.0, device=logits.device)
        )
        loss = ce_loss

        # Margin / consistency auxiliary losses
        if (self.audio_margin_weight > 0.0 or self.permutation_loss_weight > 0.0) and score_labels.ne(IGNORE_INDEX).any():
            sequence_scores = self.compute_sequence_scores(logits, score_labels)

            if self.audio_margin_weight > 0.0:
                margin_terms = []
                silence_loss = self.margin_loss(
                    sequence_scores, positive_score_indices, silence_score_indices, self.audio_margin
                )
                if silence_loss is not None:
                    margin_terms.append(silence_loss)

                mismatch_loss = self.margin_loss(
                    sequence_scores, positive_score_indices, mismatch_score_indices, self.audio_margin
                )
                if mismatch_loss is not None:
                    margin_terms.append(mismatch_loss)

                if margin_terms:
                    loss = loss + self.audio_margin_weight * torch.stack(margin_terms).mean()

            if self.permutation_loss_weight > 0.0:
                perm_loss = self.consistency_loss(
                    sequence_scores, positive_score_indices, permuted_score_indices, self.permutation_margin
                )
                if perm_loss is not None:
                    loss = loss + self.permutation_loss_weight * perm_loss

        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Enhanced AudioMCQ SFT for Fun-Audio-Chat")
    parser.add_argument("--model_name_or_path", default="../pretrained_models/Fun-Audio-Chat-8B")
    parser.add_argument("--data_path", default="datasets/audio-mcq-strongac-gemini-cot/train.jsonl")
    parser.add_argument("--output_dir", default="saves/Fun-Audio-Chat-8B/audio_mcq_enhanced_sft")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_audio_seconds", type=float, default=30.0)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2.0e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target", default="all")
    parser.add_argument("--use_counterfactual_audio", action="store_true")
    parser.add_argument("--use_silence_view", action="store_true", default=True)
    parser.add_argument("--use_mismatch_view", action="store_true", default=True)
    parser.add_argument("--use_permuted_view", action="store_true")
    parser.add_argument("--audio_margin_weight", type=float, default=0.0)
    parser.add_argument("--audio_margin", type=float, default=0.2)
    parser.add_argument("--permutation_loss_weight", type=float, default=0.0)
    parser.add_argument("--permutation_margin", type=float, default=0.0)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--allow_cpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.fp16 and args.bf16:
        raise ValueError("Use either --fp16 or --bf16, not both.")
    set_seed(args.seed)

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is not available. Pass --allow_cpu only for tiny debugging runs.")

    model_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)

    # Load config & processor
    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    # 4-bit QLoRA quantization (matching original config)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=model_dtype,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name_or_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=model_dtype,
        quantization_config=bnb_config,
        device_map="auto",
    )

    # Disable speech decoder head for S2T task (audio understanding / MCQ)
    model.sp_gen_kwargs["disable_speech"] = True

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        # Gradient checkpointing requires use_cache=False
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        text_cfg = getattr(model.config, "text_config", None)
        if text_cfg is not None and hasattr(text_cfg, "use_cache"):
            text_cfg.use_cache = False

    # LoRA
    if args.lora_target == "all":
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    else:
        target_modules = [x.strip() for x in args.lora_target.split(",")]

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Dataset
    train_dataset = AudioMCQEnhancedDataset(
        data_path=args.data_path,
        processor=processor,
        max_samples=args.max_samples,
        max_audio_seconds=args.max_audio_seconds,
        seed=args.seed,
        use_counterfactual_audio=args.use_counterfactual_audio or args.audio_margin_weight > 0.0,
        use_silence_view=args.use_silence_view,
        use_mismatch_view=args.use_mismatch_view,
        use_permuted_view=args.use_permuted_view or args.permutation_loss_weight > 0.0,
    )

    # Training args (matching original fast config as closely as possible)
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
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=4,
    )

    effective_batch = (
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        * max(1, training_args.world_size)
    )
    print(
        f"Training: {len(train_dataset)} examples | "
        f"per_device_bs={training_args.per_device_train_batch_size} | "
        f"grad_acc={training_args.gradient_accumulation_steps} | "
        f"world_size={max(1, training_args.world_size)} | "
        f"effective_batch={effective_batch}"
    )

    trainer = FunAudioChatEnhancedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=EnhancedDataCollator(processor),
        audio_margin_weight=args.audio_margin_weight,
        audio_margin=args.audio_margin,
        permutation_loss_weight=args.permutation_loss_weight,
        permutation_margin=args.permutation_margin,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Training complete. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
