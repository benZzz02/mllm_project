#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dgm4_pipeline.io_utils import load_records, write_json, write_jsonl
from dgm4_pipeline.schema import (
    IMAGE_TYPES,
    TEXT_TYPES,
    TYPE_ORDER,
    bbox_iou,
    compare_prediction,
    parse_response,
)


RESPONSE_KEYS = ("response", "predict", "prediction", "generated_text", "output")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Qwen3-VL outputs with DGM4-compatible metrics.")
    parser.add_argument("--ground-truth", type=Path, required=True, help="Converted dgm4_val.jsonl or dgm4_test.jsonl")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Metrics JSON output")
    parser.add_argument("--badcases-output", type=Path, default=None)
    parser.add_argument("--allow-positional", action="store_true")
    parser.add_argument(
        "--bert-tokenizer",
        default=None,
        help="Optional official text-token metric tokenizer, normally bert-base-uncased",
    )
    parser.add_argument("--bbox-threshold", type=float, default=0.5)
    parser.add_argument("--text-f1-threshold", type=float, default=1.0)
    parser.add_argument("--binary-threshold", type=float, default=0.5)
    parser.add_argument("--type-threshold", type=float, default=0.5)
    return parser.parse_args()


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def f1_score(precision: float, recall: float) -> float:
    return safe_div(2 * precision * recall, precision + recall)


def auc_score(labels: list[int], scores: list[float]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    indexed = sorted(enumerate(scores), key=lambda item: item[1])
    ranks = [0.0] * len(scores)
    position = 0
    while position < len(indexed):
        end = position + 1
        while end < len(indexed) and indexed[end][1] == indexed[position][1]:
            end += 1
        average_rank = (position + 1 + end) / 2.0
        for rank_index in range(position, end):
            ranks[indexed[rank_index][0]] = average_rank
        position = end
    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label == 1)
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def eer_score(labels: list[int], scores: list[float]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    ranked = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    roc_points = [(0.0, 0.0)]
    true_positive = false_positive = 0
    index = 0
    while index < len(ranked):
        end = index + 1
        while end < len(ranked) and ranked[end][0] == ranked[index][0]:
            end += 1
        for _, label in ranked[index:end]:
            true_positive += int(label == 1)
            false_positive += int(label == 0)
        roc_points.append((false_positive / negatives, true_positive / positives))
        index = end

    for (fpr_a, tpr_a), (fpr_b, tpr_b) in zip(roc_points, roc_points[1:]):
        difference_a = (1.0 - tpr_a) - fpr_a
        difference_b = (1.0 - tpr_b) - fpr_b
        if difference_a == 0:
            return fpr_a
        if difference_a * difference_b <= 0:
            denominator = difference_a - difference_b
            fraction = difference_a / denominator if denominator else 0.0
            return fpr_a + fraction * (fpr_b - fpr_a)
    return 1.0


def average_precision(labels: list[int], scores: list[float]) -> float:
    positive_count = sum(labels)
    if positive_count == 0:
        return float("nan")
    ranked = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    true_positive = 0
    precision_sum = 0.0
    for rank, (_, label) in enumerate(ranked, start=1):
        if label:
            true_positive += 1
            precision_sum += true_positive / rank
    return precision_sum / positive_count


def response_from_prediction(row: dict[str, Any]) -> Any:
    for key in RESPONSE_KEYS:
        if key in row:
            return row[key]
    raise ValueError(f"Prediction row has none of the supported response keys: {RESPONSE_KEYS}")


def row_id(row: dict[str, Any]) -> str | None:
    value = row.get("id")
    if value is None and isinstance(row.get("meta"), dict):
        value = row["meta"].get("id")
    return str(value) if value is not None else None


def align_predictions(
    ground_truth: list[dict[str, Any]], predictions: list[dict[str, Any]], allow_positional: bool
) -> list[dict[str, Any]]:
    prediction_ids = [row_id(row) for row in predictions]
    if all(value is not None for value in prediction_ids):
        by_id = {str(value): row for value, row in zip(prediction_ids, predictions)}
        if len(by_id) != len(predictions):
            raise ValueError("Prediction ids are not unique")
        missing = [str(row["id"]) for row in ground_truth if str(row["id"]) not in by_id]
        if missing:
            raise ValueError(f"Missing {len(missing)} predictions; first missing id: {missing[0]}")
        return [by_id[str(row["id"])] for row in ground_truth]
    if not allow_positional:
        raise ValueError("Predictions lack ids. Use ids or explicitly pass --allow-positional.")
    if len(ground_truth) != len(predictions):
        raise ValueError(
            f"Positional alignment requires equal counts: ground_truth={len(ground_truth)}, predictions={len(predictions)}"
        )
    return predictions


def continuous_binary_score(prediction: dict[str, Any], parsed_verdict: str | None) -> tuple[float, bool]:
    for key in ("manipulated_score", "fake_score", "score"):
        if key in prediction:
            value = float(prediction[key])
            return min(1.0, max(0.0, value)), True
    return (1.0 if parsed_verdict == "manipulated" else 0.0 if parsed_verdict == "pristine" else 0.5), False


def type_score_vector(prediction: dict[str, Any], predicted_types: Iterable[str]) -> tuple[list[float], bool]:
    raw = prediction.get("type_scores")
    if isinstance(raw, dict):
        return [min(1.0, max(0.0, float(raw.get(name, 0.0)))) for name in TYPE_ORDER], True
    if isinstance(raw, list) and len(raw) == len(TYPE_ORDER):
        return [min(1.0, max(0.0, float(value))) for value in raw], True
    predicted = set(predicted_types)
    return [float(name in predicted) for name in TYPE_ORDER], False


def confusion_metrics(tp: int, tn: int, fp: int, fn: int) -> dict[str, float]:
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    return {
        "accuracy": safe_div(tp + tn, tp + tn + fp + fn),
        "precision": precision,
        "recall": recall,
        "f1": f1_score(precision, recall),
        "false_positive_rate": safe_div(fp, fp + tn),
        "false_negative_rate": safe_div(fn, fn + tp),
    }


def percent(value: float) -> float:
    return value * 100.0


def expand_word_positions(tokenizer: Any, text: str, positions: Iterable[int]) -> tuple[set[int], int]:
    encoding = tokenizer(text, max_length=128, truncation=True, add_special_tokens=True)
    word_ids = encoding.word_ids()
    content_word_ids = word_ids[1:-1]
    expected_words = set(positions)
    token_positions = {index for index, word_id in enumerate(content_word_ids) if word_id in expected_words}
    return token_positions, len(content_word_ids)


def token_counts(expected: set[int], predicted: set[int], token_count: int) -> tuple[int, int, int, int]:
    valid_positions = set(range(token_count))
    expected = expected.intersection(valid_positions)
    predicted = predicted.intersection(valid_positions)
    tp = len(expected.intersection(predicted))
    fp = len(predicted - expected)
    fn = len(expected - predicted)
    tn = token_count - tp - fp - fn
    return tp, tn, fp, fn


def main() -> None:
    args = parse_args()
    ground_truth = load_records(args.ground_truth)
    predictions = load_records(args.predictions)
    aligned = align_predictions(ground_truth, predictions, args.allow_positional)
    if not ground_truth:
        raise ValueError("Ground-truth file is empty")

    bert_tokenizer = None
    if args.bert_tokenizer:
        from transformers import AutoTokenizer

        bert_tokenizer = AutoTokenizer.from_pretrained(args.bert_tokenizer, use_fast=True)
        if not bert_tokenizer.is_fast:
            raise ValueError("Official token grounding requires a fast tokenizer with word_ids()")

    binary_labels: list[int] = []
    binary_predictions: list[int | None] = []
    binary_scores: list[float] = []
    binary_has_continuous_scores = True
    type_targets = [[] for _ in TYPE_ORDER]
    type_scores = [[] for _ in TYPE_ORDER]
    type_predictions = [[] for _ in TYPE_ORDER]
    type_has_continuous_scores = True
    image_ious: list[float] = []
    word_tp = word_tn = word_fp = word_fn = 0
    token_tp = token_tn = token_fp = token_fn = 0
    valid_count = 0
    structured_verdict_correct = 0
    structured_types_exact = 0
    hallucinated_evidence_count = 0
    badcases = []
    error_counts: Counter[str] = Counter()

    for truth_row, prediction_row in zip(ground_truth, aligned):
        target = truth_row["meta"]["target"]
        response = response_from_prediction(prediction_row)
        parsed = parse_response(response)
        valid_count += int(parsed.valid)

        label = int(target["verdict"] == "manipulated")
        score, has_continuous_score = continuous_binary_score(prediction_row, parsed.verdict)
        predicted_label = (
            int(score >= args.binary_threshold)
            if has_continuous_score
            else None if not parsed.valid else int(parsed.verdict == "manipulated")
        )
        binary_labels.append(label)
        binary_predictions.append(predicted_label)
        binary_scores.append(score)
        binary_has_continuous_scores &= has_continuous_score

        score_vector, has_type_scores = type_score_vector(prediction_row, parsed.types if parsed.valid else ())
        type_has_continuous_scores &= has_type_scores
        expected_types = set(target["types"])
        predicted_types = set(parsed.types) if parsed.valid else set()
        structured_verdict_correct += int(parsed.valid and parsed.verdict == target["verdict"])
        structured_types_exact += int(parsed.valid and predicted_types == expected_types)
        for index, type_name in enumerate(TYPE_ORDER):
            type_targets[index].append(int(type_name in expected_types))
            type_scores[index].append(score_vector[index])
            type_predictions[index].append(
                int(score_vector[index] >= args.type_threshold)
                if has_type_scores
                else int(type_name in predicted_types)
            )

        current_iou = bbox_iou(target["image_bbox"], parsed.image_bbox) if parsed.valid else 0.0
        image_ious.append(current_iou)

        caption = truth_row["meta"]["normalized_text"]
        word_count = len(caption.split())
        expected_word_positions = set(target["text_positions"])
        predicted_word_positions = set(parsed.text_positions) if parsed.valid else set()
        counts = token_counts(expected_word_positions, predicted_word_positions, word_count)
        word_tp += counts[0]
        word_tn += counts[1]
        word_fp += counts[2]
        word_fn += counts[3]

        if bert_tokenizer is not None:
            expected_token_positions, token_count = expand_word_positions(
                bert_tokenizer, caption, expected_word_positions
            )
            predicted_token_positions, _ = expand_word_positions(
                bert_tokenizer, caption, predicted_word_positions
            )
            counts = token_counts(expected_token_positions, predicted_token_positions, token_count)
            token_tp += counts[0]
            token_tn += counts[1]
            token_fp += counts[2]
            token_fn += counts[3]

        hallucinated = bool(
            parsed.valid
            and (
                (not IMAGE_TYPES.intersection(expected_types) and parsed.image_bbox is not None)
                or (not TEXT_TYPES.intersection(expected_types) and parsed.text_positions)
            )
        )
        hallucinated_evidence_count += int(hallucinated)

        errors = compare_prediction(
            target,
            parsed,
            bbox_threshold=args.bbox_threshold,
            text_f1_threshold=args.text_f1_threshold,
        )
        error_counts.update(errors)
        if errors:
            badcases.append(
                {
                    "id": truth_row["id"],
                    "errors": errors,
                    "target": target,
                    "response": response,
                    "meta": truth_row["meta"],
                }
            )

    tp = sum(pred == 1 and label == 1 for pred, label in zip(binary_predictions, binary_labels))
    tn = sum(pred == 0 and label == 0 for pred, label in zip(binary_predictions, binary_labels))
    fp = sum((pred == 1 or pred is None) and label == 0 for pred, label in zip(binary_predictions, binary_labels))
    fn = sum((pred == 0 or pred is None) and label == 1 for pred, label in zip(binary_predictions, binary_labels))
    binary = confusion_metrics(tp, tn, fp, fn)
    binary.update(
        {
            "auc": auc_score(binary_labels, binary_scores),
            "eer": eer_score(binary_labels, binary_scores),
            "score_mode": "continuous" if binary_has_continuous_scores else "discrete_or_mixed_fallback",
        }
    )

    per_class_ap = {}
    per_class_f1 = {}
    class_tp = []
    class_predicted = []
    class_expected = []
    for index, type_name in enumerate(TYPE_ORDER):
        targets = type_targets[index]
        predictions_for_type = type_predictions[index]
        scores_for_type = type_scores[index]
        tp_type = sum(pred == 1 and label == 1 for pred, label in zip(predictions_for_type, targets))
        fp_type = sum(pred == 1 and label == 0 for pred, label in zip(predictions_for_type, targets))
        fn_type = sum(pred == 0 and label == 1 for pred, label in zip(predictions_for_type, targets))
        precision_type = safe_div(tp_type, tp_type + fp_type)
        recall_type = safe_div(tp_type, tp_type + fn_type)
        per_class_f1[type_name] = f1_score(precision_type, recall_type)
        per_class_ap[type_name] = average_precision(targets, scores_for_type)
        class_tp.append(tp_type)
        class_predicted.append(tp_type + fp_type)
        class_expected.append(tp_type + fn_type)

    op = safe_div(sum(class_tp), sum(class_predicted))
    or_ = safe_div(sum(class_tp), sum(class_expected))
    cp = sum(safe_div(tp_value, predicted_value) for tp_value, predicted_value in zip(class_tp, class_predicted)) / len(TYPE_ORDER)
    cr = sum(safe_div(tp_value, expected_value) for tp_value, expected_value in zip(class_tp, class_expected)) / len(TYPE_ORDER)
    finite_ap = [value for value in per_class_ap.values() if not math.isnan(value)]
    multilabel = {
        "map": sum(finite_ap) / len(finite_ap) if finite_ap else float("nan"),
        "op": op,
        "or": or_,
        "of1": f1_score(op, or_),
        "cp": cp,
        "cr": cr,
        "cf1": f1_score(cp, cr),
        "per_class_ap": per_class_ap,
        "per_class_f1": per_class_f1,
        "score_mode": "continuous" if type_has_continuous_scores else "discrete_or_mixed_fallback",
    }

    word_metrics = confusion_metrics(word_tp, word_tn, word_fp, word_fn)
    text_grounding: dict[str, Any] = {"word_level": word_metrics}
    if bert_tokenizer is not None:
        text_grounding["official_bert_token_level"] = {
            **confusion_metrics(token_tp, token_tn, token_fp, token_fn),
            "tokenizer": args.bert_tokenizer,
            "max_length": 128,
        }
    official_text_metrics = text_grounding.get("official_bert_token_level", word_metrics)
    official_text_source = "bert_subword" if bert_tokenizer is not None else "word_level_fallback"

    official_metrics_percent = {
        "AUC_cls": percent(binary["auc"]),
        "ACC_cls": percent(binary["accuracy"]),
        "EER_cls": percent(binary["eer"]),
        "MAP": percent(multilabel["map"]),
        "OP": percent(multilabel["op"]),
        "OR": percent(multilabel["or"]),
        "OF1": percent(multilabel["of1"]),
        "CP": percent(multilabel["cp"]),
        "CR": percent(multilabel["cr"]),
        "CF1": percent(multilabel["cf1"]),
        "F1_FS": percent(multilabel["per_class_f1"]["face_swap"]),
        "F1_FA": percent(multilabel["per_class_f1"]["face_attribute"]),
        "F1_TS": percent(multilabel["per_class_f1"]["text_swap"]),
        "F1_TA": percent(multilabel["per_class_f1"]["text_attribute"]),
        "IOU_score": percent(sum(image_ious) / len(image_ious)),
        "IOU_ACC_50": percent(sum(value > 0.5 for value in image_ious) / len(image_ious)),
        "IOU_ACC_75": percent(sum(value > 0.75 for value in image_ious) / len(image_ious)),
        "IOU_ACC_95": percent(sum(value > 0.95 for value in image_ious) / len(image_ious)),
        "ACC_tok": percent(official_text_metrics["accuracy"]),
        "Precision_tok": percent(official_text_metrics["precision"]),
        "Recall_tok": percent(official_text_metrics["recall"]),
        "F1_tok": percent(official_text_metrics["f1"]),
    }

    metrics = {
        "num_samples": len(ground_truth),
        "json_valid_rate": valid_count / len(ground_truth),
        "official_metrics_percent": official_metrics_percent,
        "structured_output": {
            "verdict_accuracy": structured_verdict_correct / len(ground_truth),
            "types_exact_match": structured_types_exact / len(ground_truth),
        },
        "binary_classification": binary,
        "multilabel_classification": multilabel,
        "image_grounding": {
            "mean_iou": sum(image_ious) / len(image_ious),
            "iou_at_50": sum(value > 0.5 for value in image_ious) / len(image_ious),
            "iou_at_75": sum(value > 0.75 for value in image_ious) / len(image_ious),
            "iou_at_95": sum(value > 0.95 for value in image_ious) / len(image_ious),
        },
        "text_grounding": text_grounding,
        "badcase_analysis": {
            "badcase_count": len(badcases),
            "badcase_rate": len(badcases) / len(ground_truth),
            "error_tag_counts": dict(sorted(error_counts.items())),
            "evidence_hallucination_rate": hallucinated_evidence_count / len(ground_truth),
        },
        "notes": {
            "official_alignment": "AUC/EER require continuous manipulated scores; mAP requires continuous per-type scores.",
            "thresholds": {
                "binary": args.binary_threshold,
                "type": args.type_threshold,
            },
            "structured_output": "Generated JSON verdict/type quality is reported separately from score-based official classification metrics.",
            "fallback": "Without sidecar scores, AUC/mAP are computed from discrete generated decisions and must be labeled as fallback.",
            "text": "Pass --bert-tokenizer bert-base-uncased to reproduce HAMMER's word-to-BERT-subword token metric.",
            "official_metrics_percent": {
                "names": "Uses the same metric names and percent scale as official DGM4/HAMMER test.py.",
                "text_source": official_text_source,
            },
        },
    }
    write_json(args.output, metrics)
    if args.badcases_output:
        write_jsonl(args.badcases_output, badcases)
    print(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
