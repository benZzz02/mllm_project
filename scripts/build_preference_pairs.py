#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dgm4_pipeline.io_utils import load_records, write_json, write_jsonl
from dgm4_pipeline.schema import canonical_answer, compare_prediction, parse_response


RESPONSE_KEYS = ("response", "predict", "prediction", "generated_text", "output")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build multimodal DPO/SimPO pairs from train-pool badcases.")
    parser.add_argument("--pool", type=Path, required=True, help="dgm4_preference_pool.jsonl")
    parser.add_argument("--predictions", type=Path, required=True, help="Model outputs as JSON/JSONL")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "generated" / "dgm4_badcase_preference.jsonl")
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--max-pairs", type=int, default=5000)
    parser.add_argument("--bbox-threshold", type=float, default=0.5)
    parser.add_argument("--text-f1-threshold", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-pristine-errors",
        action="store_true",
        help="Also mine false positives on pristine rows; default focuses RL on manipulated business badcases",
    )
    parser.add_argument(
        "--allow-positional",
        action="store_true",
        help="Join predictions to pool rows by order when predictions do not contain ids",
    )
    return parser.parse_args()


def response_from_prediction(row: dict[str, Any]) -> Any:
    for key in RESPONSE_KEYS:
        if key in row:
            return row[key]
    raise ValueError(f"Prediction row has none of the supported response keys: {RESPONSE_KEYS}")


def prediction_id(row: dict[str, Any]) -> str | None:
    value = row.get("id")
    if value is None and isinstance(row.get("meta"), dict):
        value = row["meta"].get("id")
    return str(value) if value is not None else None


def align_predictions(
    pool: list[dict[str, Any]], predictions: list[dict[str, Any]], allow_positional: bool
) -> list[dict[str, Any]]:
    ids = [prediction_id(row) for row in predictions]
    if all(value is not None for value in ids):
        by_id = {str(value): row for value, row in zip(ids, predictions)}
        if len(by_id) != len(predictions):
            raise ValueError("Prediction ids are not unique")
        missing = [row["id"] for row in pool if str(row["id"]) not in by_id]
        if missing:
            raise ValueError(f"Predictions are missing {len(missing)} pool ids; first missing id: {missing[0]}")
        return [by_id[str(row["id"])] for row in pool]
    if not allow_positional:
        raise ValueError("Predictions lack ids. Re-run with ids or explicitly pass --allow-positional.")
    if len(pool) != len(predictions):
        raise ValueError(f"Positional alignment requires equal counts: pool={len(pool)}, predictions={len(predictions)}")
    return predictions


def assert_train_pool(pool: list[dict[str, Any]]) -> None:
    invalid = []
    for row in pool:
        meta = row.get("meta", {})
        if meta.get("dgm4_split") != "train" or meta.get("pool") != "preference_pool":
            invalid.append(str(row.get("id", "unknown")))
    if invalid:
        raise ValueError(
            "Leakage guard rejected input: every preference example must come from official train/preference_pool. "
            f"Invalid rows: {invalid[:3]}"
        )


def main() -> None:
    args = parse_args()
    pool = load_records(args.pool)
    predictions = load_records(args.predictions)
    assert_train_pool(pool)
    aligned = align_predictions(pool, predictions, args.allow_positional)

    pairs = []
    error_counts: Counter[str] = Counter()
    valid_prediction_count = 0
    eligible_count = 0
    for source, prediction in zip(pool, aligned):
        target = source["meta"]["target"]
        if target["verdict"] == "pristine" and not args.include_pristine_errors:
            continue
        eligible_count += 1
        raw_response = response_from_prediction(prediction)
        parsed = parse_response(raw_response)
        valid_prediction_count += int(parsed.valid)
        errors = compare_prediction(
            target,
            parsed,
            bbox_threshold=args.bbox_threshold,
            text_f1_threshold=args.text_f1_threshold,
        )
        if not errors:
            continue
        error_counts.update(errors)
        rejected_value = (
            json.dumps(raw_response, ensure_ascii=False, separators=(",", ":"))
            if isinstance(raw_response, dict)
            else str(raw_response).strip()
        )
        if not rejected_value:
            rejected_value = "{}"
        pairs.append(
            {
                "id": source["id"],
                "images": source["images"],
                "conversations": [source["conversations"][0]],
                "chosen": {"from": "gpt", "value": canonical_answer(target)},
                "rejected": {"from": "gpt", "value": rejected_value},
                "meta": {
                    **source["meta"],
                    "error_tags": errors,
                    "mined_from": str(args.predictions),
                },
            }
        )

    badcase_count_before_cap = len(pairs)
    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    if args.max_pairs > 0:
        pairs = pairs[: args.max_pairs]
    write_jsonl(args.output, pairs)

    report_path = args.report or args.output.with_suffix(".report.json")
    report = {
        "pool_count": len(pool),
        "eligible_badcase_count": eligible_count,
        "prediction_count": len(predictions),
        "valid_prediction_rate": valid_prediction_count / eligible_count if eligible_count else 0.0,
        "badcase_count_before_cap": badcase_count_before_cap,
        "written_pair_count": len(pairs),
        "max_pairs": args.max_pairs,
        "error_tag_counts_before_cap": dict(sorted(error_counts.items())),
        "source_split": "train",
        "include_pristine_errors": args.include_pristine_errors,
        "leakage_guard": "passed",
    }
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
