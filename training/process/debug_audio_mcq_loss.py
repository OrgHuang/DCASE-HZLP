#!/usr/bin/env python
"""Debug raw Fun-Audio-Chat SFT loss on AudioMCQ batches."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from transformers import Seq2SeqTrainingArguments

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LLAMAFACTORY_SRC = PROJECT_ROOT / "third_party" / "LLaMA-Factory" / "src"
for path in (PROJECT_ROOT, LLAMAFACTORY_SRC):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from llamafactory.data import SFTDataCollatorWith4DAttentionMask, get_dataset, get_template_and_fix_tokenizer
from llamafactory.hparams import FinetuningArguments
from llamafactory.hparams.data_args import DataArguments
from llamafactory.hparams.model_args import ModelArguments
from llamafactory.model.loader import load_model, load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "training" / "configs" / "audio_mcq_qlora_sft_smoke.yaml",
        help="Training yaml to debug.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for the debug forward pass.")
    parser.add_argument("--max-samples", type=int, default=4, help="Cap dataset size for quick debugging.")
    return parser.parse_args()


def load_args_from_yaml(
    config_path: Path,
) -> tuple[ModelArguments, DataArguments, Seq2SeqTrainingArguments, FinetuningArguments, torch.dtype]:
    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    training_root = config_path.resolve().parents[1]
    model_path = Path(cfg["model_name_or_path"])
    if not model_path.is_absolute():
        model_path = (training_root / model_path).resolve()

    model_args = ModelArguments(
        model_name_or_path=str(model_path),
        trust_remote_code=cfg.get("trust_remote_code", False),
        flash_attn=cfg.get("flash_attn", "auto"),
        quantization_bit=cfg.get("quantization_bit"),
        double_quantization=cfg.get("double_quantization", False),
        quantization_type=cfg.get("quantization_type", "nf4"),
        upcast_layernorm=cfg.get("upcast_layernorm", False),
        print_param_status=cfg.get("print_param_status", False),
    )

    data_args = DataArguments(
        dataset=cfg["dataset"],
        dataset_dir=str(PROJECT_ROOT / "training" / "data"),
        template=cfg["template"],
        cutoff_len=cfg["cutoff_len"],
        overwrite_cache=cfg.get("overwrite_cache", False),
        preprocessing_num_workers=1,
        max_samples=cfg.get("max_samples"),
        val_size=cfg.get("val_size", 0.0),
    )

    finetuning_args = FinetuningArguments(
        stage=cfg["stage"],
        finetuning_type=cfg["finetuning_type"],
        lora_rank=cfg.get("lora_rank", 8),
        lora_alpha=cfg.get("lora_alpha"),
        lora_dropout=cfg.get("lora_dropout", 0.0),
        lora_target=cfg.get("lora_target", "all"),
    )

    compute_dtype = torch.bfloat16 if cfg.get("bf16", False) else torch.float16 if cfg.get("fp16", False) else torch.float32

    training_args = Seq2SeqTrainingArguments(
        output_dir="/tmp/fac_loss_debug",
        do_train=False,
        do_eval=False,
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 1),
        per_device_eval_batch_size=cfg.get("per_device_eval_batch_size", 1),
        bf16=False,
        fp16=False,
        report_to=[],
        remove_unused_columns=False,
    )
    training_args.predict_with_generate = False
    return model_args, data_args, training_args, finetuning_args, compute_dtype


def compute_manual_ce(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int) -> dict[str, float]:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    valid_mask = shift_labels.ne(ignore_index)
    valid_count = int(valid_mask.sum().item())
    if valid_count == 0:
        return {"valid_shift_labels": 0, "manual_ce": math.nan}

    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_labels = shift_labels.view(-1)
    loss = F.cross_entropy(flat_logits, flat_labels, ignore_index=ignore_index, reduction="mean")
    return {"valid_shift_labels": valid_count, "manual_ce": float(loss.detach().cpu())}


def main() -> None:
    args = parse_args()
    model_args, data_args, training_args, finetuning_args, compute_dtype = load_args_from_yaml(args.config)
    if args.max_samples is not None:
        data_args.max_samples = args.max_samples

    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    processor = tokenizer_module["processor"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, "sft", tokenizer, processor)
    train_dataset = dataset_module["train_dataset"]

    model = load_model(tokenizer, model_args, finetuning_args, is_trainable=True, add_valuehead=False)
    model.eval()

    collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
        attn_implementation="sdpa",
        compute_dtype=compute_dtype,
    )

    features = [train_dataset[i] for i in range(min(args.batch_size, len(train_dataset)))]
    batch = collator(features)
    device = next(model.parameters()).device
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

    with torch.inference_mode():
        outputs = model(**batch, return_dict=True)

    labels = batch["labels"]
    valid_labels = int(labels.ne(-100).sum().item())
    manual_stats = compute_manual_ce(outputs.text_logits.float(), labels, ignore_index=-100)

    result = {
        "device": str(device),
        "batch_size": len(features),
        "input_shape": tuple(batch["input_ids"].shape),
        "labels_shape": tuple(labels.shape),
        "valid_labels": valid_labels,
        "output_loss": None if outputs.loss is None else float(outputs.loss.detach().cpu()),
        "output_text_loss": None if outputs.text_loss is None else float(outputs.text_loss.detach().cpu()),
        "output_speech_loss": None if outputs.speech_loss is None else float(outputs.speech_loss.detach().cpu()),
        "manual_stats": manual_stats,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
