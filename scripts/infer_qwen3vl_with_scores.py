#!/usr/bin/env python3
"""Generate structured predictions and likelihood scores for DGM4 evaluation.

Training remains in LLaMA-Factory. This script loads the resulting PEFT adapter
with the Hugging Face runtime so AUC/EER and mAP receive continuous scores.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dgm4_pipeline.io_utils import load_records
from dgm4_pipeline.schema import TYPE_ORDER


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Qwen3-VL generation plus sequence-likelihood scoring on converted DGM4 data."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--adapter", type=Path, default=None, help="LLaMA-Factory LoRA output; omit for Base")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto")
    parser.add_argument("--attn-implementation", default=None, help="For example flash_attention_2 or sdpa")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--resume", action="store_true", help="Append to an existing output file and skip completed ids")
    return parser.parse_args()


def strip_image_token(prompt: str) -> str:
    return prompt.replace("<image>\n", "", 1).replace("<image>", "", 1).lstrip()


def chat_messages(image: Any, prompt: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def apply_template(processor: Any, messages: list[dict[str, Any]]) -> str:
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    try:
        return processor.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return processor.apply_chat_template(messages, **kwargs)


def prepare_inputs(processor: Any, image: Any, prompt: str, device: Any) -> Any:
    rendered = apply_template(processor, chat_messages(image, prompt))
    batch = processor(text=[rendered], images=[image], padding=True, return_tensors="pt")
    return batch.to(device)


class KnownProcessorWarningFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "Kwargs passed to `processor.__call__` have to be in `processor_kwargs`" not in record.getMessage()


def suppress_known_transformers_noise() -> None:
    warning_filter = KnownProcessorWarningFilter()
    logging.getLogger("transformers").addFilter(warning_filter)


def append_candidate_to_inputs(inputs: Any, candidate_ids: Any) -> tuple[dict[str, Any], Any]:
    import torch

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    sequence_length = input_ids.shape[1]
    candidate_length = candidate_ids.shape[1]

    full_inputs: dict[str, Any] = {}
    for key, value in dict(inputs).items():
        if key == "input_ids":
            full_inputs[key] = torch.cat((input_ids, candidate_ids), dim=1)
        elif key == "attention_mask":
            extra_attention = torch.ones_like(candidate_ids, dtype=attention_mask.dtype)
            full_inputs[key] = torch.cat((attention_mask, extra_attention), dim=1)
        elif torch.is_tensor(value) and value.ndim >= 2 and value.shape[0] == input_ids.shape[0] and value.shape[1] == sequence_length:
            # Qwen3-VL processors may return token-type tensors with the same
            # sequence length as input_ids. Candidate text tokens must extend
            # those fields too, otherwise RoPE indexing sees mismatched shapes.
            repeat_shape = [1, candidate_length] + [1] * (value.ndim - 2)
            full_inputs[key] = torch.cat((value, value[:, -1:].repeat(*repeat_shape)), dim=1)
        else:
            full_inputs[key] = value

    labels = torch.full_like(full_inputs["input_ids"], -100)
    labels[:, -candidate_length:] = candidate_ids
    return full_inputs, labels


def candidate_mean_logprob(model: Any, processor: Any, inputs: Any, candidate: str) -> float:
    import torch

    candidate_ids = processor.tokenizer(candidate, add_special_tokens=False, return_tensors="pt").input_ids
    candidate_ids = candidate_ids.to(inputs["input_ids"].device)
    if candidate_ids.shape[1] == 0:
        raise ValueError(f"Candidate tokenized to an empty sequence: {candidate!r}")

    full_inputs, labels = append_candidate_to_inputs(inputs, candidate_ids)
    with torch.inference_mode():
        output = model(**full_inputs, labels=labels, use_cache=False)
    return -float(output.loss.detach().float().item())


def normalized_candidate_score(
    model: Any,
    processor: Any,
    inputs: Any,
    negative: str,
    positive: str,
) -> float:
    negative_logprob = candidate_mean_logprob(model, processor, inputs, negative)
    positive_logprob = candidate_mean_logprob(model, processor, inputs, positive)
    maximum = max(negative_logprob, positive_logprob)
    negative_exp = math.exp(negative_logprob - maximum)
    positive_exp = math.exp(positive_logprob - maximum)
    return positive_exp / (negative_exp + positive_exp)


def classification_prompt(caption: str) -> str:
    return (
        f"News text: {caption}\n"
        "Decide whether the image-text pair is pristine or manipulated. "
        "Answer with exactly one word: pristine or manipulated."
    )


def type_prompt(caption: str, type_name: str) -> str:
    descriptions = {
        "face_swap": "a face identity swap",
        "face_attribute": "a manipulated facial attribute",
        "text_swap": "a swapped text entity or phrase",
        "text_attribute": "a manipulated textual attribute",
    }
    return (
        f"News text: {caption}\n"
        f"Does this image-text pair contain {descriptions[type_name]}? "
        "Answer with exactly one word: yes or no."
    )


def generate_response(model: Any, processor: Any, inputs: Any, max_new_tokens: int) -> str:
    import torch

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    continuation = generated[:, inputs["input_ids"].shape[1] :]
    return processor.batch_decode(continuation, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def load_runtime(args: argparse.Namespace) -> tuple[Any, Any]:
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    load_kwargs: dict[str, Any] = {
        "dtype": args.dtype,
        "device_map": args.device_map,
    }
    if args.attn_implementation:
        load_kwargs["attn_implementation"] = args.attn_implementation
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, **load_kwargs)
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(args.adapter))
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model)
    return model, processor


def main() -> None:
    args = parse_args()
    suppress_known_transformers_noise()
    rows = load_records(args.dataset)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("Dataset is empty")

    model, processor = load_runtime(args)
    device = model.device
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed_ids: set[str] = set()
    existing_count = 0
    if args.resume and args.output.exists():
        existing_rows = load_records(args.output)
        completed_ids = {str(row["id"]) for row in existing_rows if "id" in row}
        existing_count = len(existing_rows)
        print(
            f"Resuming from {args.output}: {existing_count} existing rows, {len(completed_ids)} unique ids",
            file=sys.stderr,
            flush=True,
        )

    mode = "a" if args.resume and args.output.exists() else "w"
    with args.output.open(mode, encoding="utf-8") as handle:
        written = existing_count
        for index, row in enumerate(rows, start=1):
            from PIL import Image

            if str(row["id"]) in completed_ids:
                print(f"[{index}/{len(rows)}] {row['id']} skipped", file=sys.stderr, flush=True)
                continue

            image_path = Path(row["images"][0])
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
                generation_prompt = strip_image_token(row["conversations"][0]["value"])
                generation_inputs = prepare_inputs(processor, image, generation_prompt, device)
                response = generate_response(model, processor, generation_inputs, args.max_new_tokens)

                caption = row["meta"]["normalized_text"]
                binary_inputs = prepare_inputs(processor, image, classification_prompt(caption), device)
                manipulated_score = normalized_candidate_score(
                    model, processor, binary_inputs, "pristine", "manipulated"
                )
                type_scores = {}
                for type_name in TYPE_ORDER:
                    scoring_inputs = prepare_inputs(processor, image, type_prompt(caption, type_name), device)
                    type_scores[type_name] = normalized_candidate_score(model, processor, scoring_inputs, "no", "yes")

            prediction = {
                "id": row["id"],
                "response": response,
                "manipulated_score": manipulated_score,
                "type_scores": type_scores,
                "score_method": "length_normalized_sequence_likelihood",
                "model": args.model,
                "adapter": str(args.adapter) if args.adapter else None,
            }
            handle.write(json.dumps(prediction, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            written += 1
            print(f"[{index}/{len(rows)}] {row['id']}", file=sys.stderr, flush=True)

    print(json.dumps({"written": written, "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
