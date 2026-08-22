#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dgm4_pipeline.grpo_reward import clipped_case_weights, count_cases, make_badcase_aware_reward
from dgm4_pipeline.io_utils import load_records, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SFT -> Badcase-Aware GRPO for DGM4/Qwen3-VL.")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data" / "generated" / "dgm4_preference_pool.jsonl")
    parser.add_argument("--weights-from", type=Path, default=PROJECT_ROOT / "data" / "generated" / "dgm4_sft_train.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--adapter", type=Path, default=PROJECT_ROOT / "outputs" / "sft_lora")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "sft_grpo_lora")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5.0e-6)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-completion-length", type=int, default=192)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--beta", type=float, default=0.01, help="KL weight against the frozen SFT reference adapter.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--report-to", default="tensorboard")
    parser.add_argument("--use-vllm", action="store_true")
    parser.add_argument("--vllm-mode", default="colocate")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.35)
    return parser.parse_args()


def strip_image_token(prompt: str) -> str:
    return prompt.replace("<image>\n", "", 1).replace("<image>", "", 1).lstrip()


def first_user_prompt(record: dict[str, Any]) -> str:
    conversations = record.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        raise ValueError(f"Record {record.get('id', '<unknown>')} has no conversations")
    first = conversations[0]
    if first.get("from") != "human":
        raise ValueError(f"Record {record.get('id', '<unknown>')} does not start with a human turn")
    return strip_image_token(str(first.get("value", "")))


def validate_train_pool(records: list[dict[str, Any]]) -> None:
    for record in records:
        meta = record.get("meta", {})
        if meta.get("dgm4_split") != "train" or meta.get("pool") != "preference_pool":
            raise ValueError("GRPO dataset must come from official train/preference_pool only.")


def build_dataset(path: Path, max_samples: int | None):
    from datasets import Dataset, Image as DatasetImage

    records = load_records(path)
    validate_train_pool(records)
    if max_samples is not None:
        records = records[:max_samples]
    examples = [
        {
            "id": record["id"],
            "prompt": [{"role": "user", "content": first_user_prompt(record)}],
            "image": record["images"][0],
            "target": record["meta"]["target"],
        }
        for record in records
    ]
    if not examples:
        raise ValueError("GRPO dataset is empty")
    return Dataset.from_list(examples).cast_column("image", DatasetImage(decode=True))


def supported_kwargs(cls: type, values: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(cls.__init__)
    parameters = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return values
    return {key: value for key, value in values.items() if key in parameters}


def report_to_value(raw: str) -> str | list[str]:
    if raw.lower() in {"none", "no", "false"}:
        return []
    if "," in raw:
        return [item.strip() for item in raw.split(",") if item.strip()]
    return raw


def load_policy_model(args: argparse.Namespace):
    from peft import PeftModel
    from transformers import Qwen3VLForConditionalGeneration

    load_kwargs: dict[str, Any] = {"dtype": args.dtype, "trust_remote_code": True}
    if args.attn_implementation:
        load_kwargs["attn_implementation"] = args.attn_implementation
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, **load_kwargs)
    model = PeftModel.from_pretrained(model, str(args.adapter), adapter_name="policy", is_trainable=True)
    try:
        model.load_adapter(str(args.adapter), adapter_name="ref", is_trainable=False)
    except Exception as exc:
        if args.beta > 0:
            raise RuntimeError("Failed to load the SFT adapter as the GRPO reference adapter.") from exc
    model.set_adapter("policy")
    model.config.use_cache = False
    return model


def main() -> None:
    args = parse_args()
    if args.num_generations < 2:
        raise ValueError("--num-generations must be at least 2 for GRPO")
    if not args.adapter.exists():
        raise FileNotFoundError(f"SFT adapter not found: {args.adapter}")

    from transformers import AutoProcessor
    from trl import GRPOConfig, GRPOTrainer

    train_dataset = build_dataset(args.dataset, args.max_samples)
    case_counts = count_cases(load_records(args.weights_from))
    case_weights = clipped_case_weights(case_counts)
    reward_func = make_badcase_aware_reward(case_weights=case_weights)

    config_values: dict[str, Any] = {
        "output_dir": str(args.output_dir),
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_generations": args.num_generations,
        "max_prompt_length": args.max_prompt_length,
        "max_completion_length": args.max_completion_length,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "bf16": args.dtype in {"auto", "bfloat16"},
        "fp16": args.dtype == "float16",
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "remove_unused_columns": False,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "report_to": report_to_value(args.report_to),
        "seed": args.seed,
        "trust_remote_code": True,
        "beta": args.beta,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "use_vllm": args.use_vllm,
        "vllm_mode": args.vllm_mode,
        "vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization,
    }
    training_args = GRPOConfig(**supported_kwargs(GRPOConfig, config_values))

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"

    model = load_policy_model(args)
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        reward_funcs=[reward_func],
        train_dataset=train_dataset,
        processing_class=processor,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    processor.save_pretrained(str(args.output_dir))
    write_json(
        args.output_dir / "grpo_reward_manifest.json",
        {
            "dataset": str(args.dataset),
            "weights_from": str(args.weights_from),
            "case_counts": dict(sorted(case_counts.items())),
            "case_weights": dict(sorted(case_weights.items())),
            "num_train_samples": len(train_dataset),
            "num_generations": args.num_generations,
            "beta": args.beta,
            "reward": "W_case * (0.5*R_verdict + 0.25*R_type + 0.25*R_evidence) - R_penalty",
        },
    )


if __name__ == "__main__":
    main()
