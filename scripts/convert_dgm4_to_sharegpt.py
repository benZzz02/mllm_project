#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dgm4_pipeline.io_utils import load_records, write_json, write_jsonl
from dgm4_pipeline.schema import build_prompt, build_target, canonical_answer, normalize_caption


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert official DGM4 metadata to LLaMA-Factory ShareGPT JSONL.")
    parser.add_argument("--metadata-dir", type=Path, required=True, help="Directory containing train.json/val.json/test.json")
    parser.add_argument("--image-root", type=Path, required=True, help="Dataset root used to resolve annotation image paths")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "generated")
    parser.add_argument("--preference-pool-ratio", type=float, default=0.10)
    parser.add_argument("--normal-ratio", type=float, default=0.50, help="Target pristine ratio for SFT train only")
    parser.add_argument("--max-words", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples-per-split", type=int, default=None, help="Debug-only cap applied before conversion")
    parser.add_argument("--skip-image-check", action="store_true", help="Do not call Image.verify; dimensions are still read")
    return parser.parse_args()


def stable_fraction(seed: int, group_key: str) -> float:
    digest = hashlib.sha256(f"{seed}:{group_key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def sample_uid(split: str, index: int, annotation: dict[str, Any]) -> str:
    identity = {
        "split": split,
        "index": index,
        "id": annotation.get("id"),
        "image": annotation.get("image"),
        "text": annotation.get("text"),
        "fake_cls": annotation.get("fake_cls"),
        "fake_image_box": annotation.get("fake_image_box"),
        "fake_text_pos": annotation.get("fake_text_pos"),
    }
    payload = json.dumps(identity, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:20]


def resolve_image(image_root: Path, annotation_path: str) -> Path:
    relative = Path(annotation_path)
    candidates = [image_root / relative]
    if relative.parts and relative.parts[0].lower() == image_root.name.lower():
        candidates.append(image_root.joinpath(*relative.parts[1:]))
    candidates.append(image_root.parent / relative)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    attempted = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not resolve image {annotation_path}. Tried: {attempted}")


def image_size(path: Path, verify: bool) -> tuple[int, int]:
    with Image.open(path) as image:
        width, height = image.size
        if verify:
            image.verify()
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions for {path}: {width}x{height}")
    return width, height


def convert_record(
    split: str,
    pool: str,
    index: int,
    annotation: dict[str, Any],
    image_root: Path,
    max_words: int,
    verify_image: bool,
) -> dict[str, Any]:
    required = ("image", "text", "fake_cls", "fake_text_pos")
    missing = [key for key in required if key not in annotation]
    if missing:
        raise ValueError(f"DGM4 record {split}[{index}] is missing fields: {missing}")
    image_path = resolve_image(image_root, str(annotation["image"]))
    width, height = image_size(image_path, verify_image)
    caption = normalize_caption(str(annotation["text"]), max_words=max_words)
    target = build_target(annotation, width=width, height=height, max_words=max_words)
    uid = sample_uid(split, index, annotation)
    return {
        "id": uid,
        "images": [str(image_path)],
        "conversations": [
            {"from": "human", "value": build_prompt(caption)},
            {"from": "gpt", "value": canonical_answer(target)},
        ],
        "meta": {
            "id": uid,
            "news_id": str(annotation.get("id", "")),
            "dgm4_split": split,
            "pool": pool,
            "fake_cls": str(annotation["fake_cls"]),
            "fake_image_box": annotation.get("fake_image_box"),
            "fake_text_pos": annotation.get("fake_text_pos", []),
            "normalized_text": caption,
            "image_width": width,
            "image_height": height,
            "target": target,
        },
    }


def balanced_train(records: list[dict[str, Any]], normal_ratio: float, seed: int) -> list[dict[str, Any]]:
    if not 0.0 < normal_ratio < 1.0:
        raise ValueError("--normal-ratio must be between 0 and 1")
    pristine = [record for record in records if record["meta"]["target"]["verdict"] == "pristine"]
    manipulated = [record for record in records if record["meta"]["target"]["verdict"] == "manipulated"]
    if not pristine or not manipulated:
        raise ValueError("SFT train balancing requires both pristine and manipulated samples")
    max_total_from_pristine = int(len(pristine) / normal_ratio)
    max_total_from_manipulated = int(len(manipulated) / (1.0 - normal_ratio))
    target_total = min(max_total_from_pristine, max_total_from_manipulated)
    target_pristine = min(len(pristine), round(target_total * normal_ratio))
    target_manipulated = min(len(manipulated), target_total - target_pristine)
    rng = random.Random(seed)
    selected = rng.sample(pristine, target_pristine) + rng.sample(manipulated, target_manipulated)
    rng.shuffle(selected)
    return selected


def dataset_info() -> dict[str, Any]:
    common_columns = {"messages": "conversations", "images": "images"}
    return {
        "dgm4_sft_train": {
            "file_name": "dgm4_sft_train.jsonl",
            "formatting": "sharegpt",
            "columns": common_columns,
        },
        "dgm4_preference_pool": {
            "file_name": "dgm4_preference_pool.jsonl",
            "formatting": "sharegpt",
            "columns": common_columns,
        },
        "dgm4_val": {
            "file_name": "dgm4_val.jsonl",
            "formatting": "sharegpt",
            "columns": common_columns,
        },
        "dgm4_test": {
            "file_name": "dgm4_test.jsonl",
            "formatting": "sharegpt",
            "columns": common_columns,
        },
        "dgm4_badcase_preference": {
            "file_name": "dgm4_badcase_preference.jsonl",
            "formatting": "sharegpt",
            "ranking": True,
            "columns": {
                "messages": "conversations",
                "chosen": "chosen",
                "rejected": "rejected",
                "images": "images",
            },
        },
        "dgm4_base_badcase_preference": {
            "file_name": "dgm4_base_badcase_preference.jsonl",
            "formatting": "sharegpt",
            "ranking": True,
            "columns": {
                "messages": "conversations",
                "chosen": "chosen",
                "rejected": "rejected",
                "images": "images",
            },
        },
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    labels = Counter(record["meta"]["fake_cls"] for record in records)
    verdicts = Counter(record["meta"]["target"]["verdict"] for record in records)
    return {"count": len(records), "verdicts": dict(sorted(verdicts.items())), "labels": dict(sorted(labels.items()))}


def main() -> None:
    args = parse_args()
    if not 0.0 < args.preference_pool_ratio < 0.5:
        raise ValueError("--preference-pool-ratio must be between 0 and 0.5")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, list[dict[str, Any]]] = {}

    train_annotations = load_records(args.metadata_dir / "train.json")
    if args.max_samples_per_split:
        train_annotations = train_annotations[: args.max_samples_per_split]
    train_sft_raw: list[dict[str, Any]] = []
    preference_pool: list[dict[str, Any]] = []
    for index, annotation in enumerate(train_annotations):
        group_key = str(annotation.get("id", annotation.get("image", index)))
        is_preference = stable_fraction(args.seed, group_key) < args.preference_pool_ratio
        pool = "preference_pool" if is_preference else "sft_train"
        converted = convert_record(
            "train", pool, index, annotation, args.image_root, args.max_words, not args.skip_image_check
        )
        (preference_pool if is_preference else train_sft_raw).append(converted)

    outputs["dgm4_sft_train"] = balanced_train(train_sft_raw, args.normal_ratio, args.seed)
    outputs["dgm4_preference_pool"] = preference_pool

    for split in ("val", "test"):
        annotations = load_records(args.metadata_dir / f"{split}.json")
        if args.max_samples_per_split:
            annotations = annotations[: args.max_samples_per_split]
        outputs[f"dgm4_{split}"] = [
            convert_record(split, "evaluation", index, annotation, args.image_root, args.max_words, not args.skip_image_check)
            for index, annotation in enumerate(annotations)
        ]

    for name, records in outputs.items():
        write_jsonl(args.output_dir / f"{name}.jsonl", records)
    write_json(args.output_dir / "dataset_info.json", dataset_info())

    manifest = {
        "seed": args.seed,
        "max_words": args.max_words,
        "preference_pool_ratio": args.preference_pool_ratio,
        "normal_ratio": args.normal_ratio,
        "source": {
            "metadata_dir": str(args.metadata_dir.resolve()),
            "image_root": str(args.image_root.resolve()),
        },
        "outputs": {name: summarize(records) for name, records in outputs.items()},
        "leakage_guard": "preference data comes only from the official train split; val/test are evaluation-only",
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
