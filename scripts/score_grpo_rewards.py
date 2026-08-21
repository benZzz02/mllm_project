#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dgm4_pipeline.grpo_reward import clipped_case_weights, count_cases, score_response
from dgm4_pipeline.io_utils import load_records, write_json, write_jsonl


RESPONSE_KEYS = ("response", "predict", "prediction", "generated_text", "output")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score model outputs with the DGM4 badcase-aware GRPO reward.")
    parser.add_argument("--ground-truth", type=Path, required=True, help="Converted DGM4 JSONL with meta.target")
    parser.add_argument("--responses", type=Path, required=True, help="Prediction JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Per-sample reward JSONL")
    parser.add_argument("--summary-output", type=Path, default=None, help="Reward summary JSON")
    parser.add_argument(
        "--weights-from",
        type=Path,
        default=None,
        help="Dataset used to estimate case weights; defaults to --ground-truth",
    )
    parser.add_argument("--allow-positional", action="store_true")
    return parser.parse_args()


def row_id(row: dict[str, Any]) -> str | None:
    value = row.get("id")
    if value is None and isinstance(row.get("meta"), dict):
        value = row["meta"].get("id")
    return str(value) if value is not None else None


def align_records(ground_truth: list[dict[str, Any]], responses: list[dict[str, Any]], allow_positional: bool) -> list[dict[str, Any]]:
    response_ids = [row_id(row) for row in responses]
    if all(value is not None for value in response_ids):
        by_id = {str(value): row for value, row in zip(response_ids, responses)}
        if len(by_id) != len(responses):
            raise ValueError("Response ids are not unique")
        missing = [str(row["id"]) for row in ground_truth if str(row["id"]) not in by_id]
        if missing:
            raise ValueError(f"Missing {len(missing)} responses; first missing id: {missing[0]}")
        return [by_id[str(row["id"])] for row in ground_truth]
    if not allow_positional:
        raise ValueError("Responses lack ids. Use ids or pass --allow-positional.")
    if len(ground_truth) != len(responses):
        raise ValueError(f"Positional alignment requires equal counts: {len(ground_truth)} vs {len(responses)}")
    return responses


def response_text(row: dict[str, Any]) -> Any:
    for key in RESPONSE_KEYS:
        if key in row:
            return row[key]
    raise ValueError(f"Response row has none of the supported keys: {RESPONSE_KEYS}")


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    args = parse_args()
    ground_truth = load_records(args.ground_truth)
    responses = load_records(args.responses)
    aligned = align_records(ground_truth, responses, args.allow_positional)
    weight_records = load_records(args.weights_from or args.ground_truth)
    case_counts = count_cases(weight_records)
    case_weights = clipped_case_weights(case_counts)

    scored_rows = []
    rewards = []
    penalties = []
    valid_count = 0
    case_counter: Counter[str] = Counter()
    for truth_row, response_row in zip(ground_truth, aligned):
        target = truth_row["meta"]["target"]
        breakdown = score_response(target, response_text(response_row), case_weights)
        rewards.append(breakdown.reward)
        penalties.append(breakdown.penalty)
        valid_count += int(breakdown.valid)
        case_counter.update([breakdown.case])
        scored_rows.append(
            {
                "id": truth_row["id"],
                "reward": breakdown.reward,
                "breakdown": breakdown.to_dict(),
                "target": target,
            }
        )

    write_jsonl(args.output, scored_rows)
    summary = {
        "count": len(scored_rows),
        "mean_reward": average(rewards),
        "min_reward": min(rewards) if rewards else 0.0,
        "max_reward": max(rewards) if rewards else 0.0,
        "mean_penalty": average(penalties),
        "valid_rate": valid_count / len(scored_rows) if scored_rows else 0.0,
        "case_counts": dict(sorted(case_counter.items())),
        "weight_counts": dict(sorted(case_counts.items())),
        "case_weights": dict(sorted(case_weights.items())),
    }
    if args.summary_output:
        write_json(args.summary_output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
