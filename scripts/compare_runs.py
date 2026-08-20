#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dgm4_pipeline.io_utils import write_json


METRICS = (
    ("binary_auc", ("binary_classification", "auc"), "higher"),
    ("binary_accuracy", ("binary_classification", "accuracy"), "higher"),
    ("binary_eer", ("binary_classification", "eer"), "lower"),
    ("multilabel_map", ("multilabel_classification", "map"), "higher"),
    ("multilabel_of1", ("multilabel_classification", "of1"), "higher"),
    ("multilabel_cf1", ("multilabel_classification", "cf1"), "higher"),
    ("image_mean_iou", ("image_grounding", "mean_iou"), "higher"),
    ("image_iou_at_50", ("image_grounding", "iou_at_50"), "higher"),
    ("json_valid_rate", ("json_valid_rate",), "higher"),
    ("structured_verdict_accuracy", ("structured_output", "verdict_accuracy"), "higher"),
    ("structured_types_exact_match", ("structured_output", "types_exact_match"), "higher"),
    ("badcase_rate", ("badcase_analysis", "badcase_rate"), "lower"),
    (
        "evidence_hallucination_rate",
        ("badcase_analysis", "evidence_hallucination_rate"),
        "lower",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two evaluator outputs and quantify RL improvement.")
    parser.add_argument("--before", type=Path, required=True, help="Usually SFT validation metrics")
    parser.add_argument("--after", type=Path, required=True, help="Usually SFT-DPO validation metrics")
    parser.add_argument("--before-name", default="before")
    parser.add_argument("--after-name", default="after")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, default=None)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def nested_value(payload: dict[str, Any], path: tuple[str, ...]) -> float | None:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def text_f1(payload: dict[str, Any]) -> tuple[float | None, str]:
    text = payload.get("text_grounding", {})
    official = text.get("official_bert_token_level") if isinstance(text, dict) else None
    if isinstance(official, dict) and official.get("f1") is not None:
        return nested_value(payload, ("text_grounding", "official_bert_token_level", "f1")), "official_bert_token"
    return nested_value(payload, ("text_grounding", "word_level", "f1")), "word_level_fallback"


def compare_value(before: float, after: float, direction: str) -> dict[str, Any]:
    delta = after - before
    improved = delta > 0 if direction == "higher" else delta < 0
    if delta == 0:
        improved = False
    if direction == "higher":
        denominator = 1.0 - before
        relative_reduction = delta / denominator if denominator > 0 else None
    else:
        relative_reduction = (before - after) / before if before > 0 else None
    return {
        "before": before,
        "after": after,
        "absolute_delta": delta,
        "direction": direction,
        "improved": improved,
        "relative_error_reduction": relative_reduction,
    }


def markdown_report(result: dict[str, Any]) -> str:
    lines = [
        f"# {result['before_name']} vs {result['after_name']}",
        "",
        "| Metric | Before | After | Delta | Improved | Relative error reduction |",
        "|---|---:|---:|---:|:---:|---:|",
    ]
    for name, values in result["metrics"].items():
        reduction = values["relative_error_reduction"]
        reduction_text = "N/A" if reduction is None else f"{reduction * 100:.2f}%"
        lines.append(
            f"| {name} | {values['before']:.6f} | {values['after']:.6f} | "
            f"{values['absolute_delta']:+.6f} | {'yes' if values['improved'] else 'no'} | {reduction_text} |"
        )
    lines.append("")
    lines.append(f"Improved metrics: {result['summary']['improved_count']}/{result['summary']['compared_count']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    before = read_json(args.before)
    after = read_json(args.after)
    compared = {}
    missing = []
    for name, path, direction in METRICS:
        before_value = nested_value(before, path)
        after_value = nested_value(after, path)
        if before_value is None or after_value is None:
            missing.append(name)
            continue
        compared[name] = compare_value(before_value, after_value, direction)

    before_text_f1, before_mode = text_f1(before)
    after_text_f1, after_mode = text_f1(after)
    if before_text_f1 is not None and after_text_f1 is not None and before_mode == after_mode:
        compared["text_f1"] = {
            **compare_value(before_text_f1, after_text_f1, "higher"),
            "mode": before_mode,
        }
    else:
        missing.append("text_f1")

    for type_name in ("face_swap", "face_attribute", "text_swap", "text_attribute"):
        path = ("multilabel_classification", "per_class_f1", type_name)
        before_value = nested_value(before, path)
        after_value = nested_value(after, path)
        if before_value is not None and after_value is not None:
            compared[f"{type_name}_f1"] = compare_value(before_value, after_value, "higher")

    result = {
        "before_name": args.before_name,
        "after_name": args.after_name,
        "before_file": str(args.before),
        "after_file": str(args.after),
        "metrics": compared,
        "missing_or_incompatible": missing,
        "summary": {
            "compared_count": len(compared),
            "improved_count": sum(value["improved"] for value in compared.values()),
        },
        "interpretation": (
            "Use absolute_delta for paper/interview tables. Relative error reduction is computed against the "
            "remaining error for higher-is-better metrics and against the original value for lower-is-better metrics."
        ),
    }
    write_json(args.output, result)
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(markdown_report(result), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
