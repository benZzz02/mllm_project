#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dgm4_pipeline.io_utils import load_records, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a prompt-only ShareGPT dataset for LLaMA-Factory PPO.")
    parser.add_argument("--source", type=Path, required=True, help="Usually dgm4_preference_pool.jsonl")
    parser.add_argument("--output", type=Path, required=True, help="Output prompt dataset JSONL")
    parser.add_argument("--dataset-info", type=Path, required=True, help="dataset_info.json to patch")
    parser.add_argument("--dataset-name", default="dgm4_ppo_train")
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def first_user_turn(record: dict[str, Any]) -> dict[str, str]:
    conversations = record.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        raise ValueError(f"Record {record.get('id', '<unknown>')} has no conversations")
    first = conversations[0]
    if first.get("from") != "human":
        raise ValueError(f"Record {record.get('id', '<unknown>')} does not start with a human turn")
    return {"from": "human", "value": str(first.get("value", ""))}


def patch_dataset_info(path: Path, dataset_name: str, output: Path) -> None:
    info = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    info[dataset_name] = {
        "file_name": output.name,
        "formatting": "sharegpt",
        "columns": {"messages": "conversations", "images": "images"},
    }
    write_json(path, info)


def main() -> None:
    args = parse_args()
    records = load_records(args.source)
    if args.max_samples is not None:
        records = records[: args.max_samples]

    prompts = []
    for record in records:
        meta = record.get("meta", {})
        if meta.get("dgm4_split") != "train" or meta.get("pool") != "preference_pool":
            raise ValueError(
                "Leakage guard rejected input: PPO prompts must come from official train/preference_pool."
            )
        prompts.append(
            {
                "id": record["id"],
                "images": record["images"],
                "conversations": [first_user_turn(record)],
                "meta": meta,
            }
        )

    count = write_jsonl(args.output, prompts)
    patch_dataset_info(args.dataset_info, args.dataset_name, args.output)
    print(
        json.dumps(
            {
                "source": str(args.source),
                "output": str(args.output),
                "dataset_info": str(args.dataset_info),
                "dataset_name": args.dataset_name,
                "num_samples": count,
                "leakage_guard": "passed",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
