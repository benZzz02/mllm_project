from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass
from statistics import median
from typing import Any, Callable, Iterable

from .schema import IMAGE_TYPES, TEXT_TYPES, TYPE_ORDER, bbox_iou, parse_response, set_f1


CASE_NORMAL = "normal"
CASE_IMAGE_TEXT_MIXED = "image_text_mixed"
CASE_IMAGE_MIXED = "image_mixed"
CASE_TEXT_MIXED = "text_mixed"


@dataclass(frozen=True)
class RewardWeights:
    verdict: float = 0.50
    manipulation_type: float = 0.25
    evidence: float = 0.25


@dataclass(frozen=True)
class RewardConfig:
    min_case_weight: float = 0.6
    max_case_weight: float = 2.0
    max_penalty: float = 0.5
    all_types_penalty: float = 0.2
    full_bbox_penalty: float = 0.2
    full_text_penalty: float = 0.2
    false_alarm_penalty: float = 0.2
    full_bbox_area_ratio: float = 0.8
    full_text_ratio: float = 0.8
    max_words: int = 50
    weights: RewardWeights = RewardWeights()


@dataclass(frozen=True)
class RewardBreakdown:
    reward: float
    case: str
    case_weight: float
    task_reward: float
    verdict_reward: float
    type_reward: float
    evidence_reward: float
    penalty: float
    valid: bool
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def case_name(target: dict[str, Any]) -> str:
    types = tuple(target.get("types", ()))
    if target.get("verdict") != "manipulated" or not types:
        return CASE_NORMAL

    type_set = set(types)
    has_image = bool(IMAGE_TYPES.intersection(type_set))
    has_text = bool(TEXT_TYPES.intersection(type_set))
    if has_image and has_text:
        return CASE_IMAGE_TEXT_MIXED
    if len(type_set) == 1:
        return next(type_name for type_name in TYPE_ORDER if type_name in type_set)
    if has_image:
        return CASE_IMAGE_MIXED
    if has_text:
        return CASE_TEXT_MIXED
    return CASE_NORMAL


def count_cases(records: Iterable[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        target = record.get("target")
        if target is None and isinstance(record.get("meta"), dict):
            target = record["meta"].get("target")
        if not isinstance(target, dict):
            raise ValueError("Each record must contain target or meta.target")
        counts[case_name(target)] += 1
    return counts


def clipped_case_weights(
    counts: Counter[str] | dict[str, int],
    config: RewardConfig = RewardConfig(),
) -> dict[str, float]:
    positive_counts = [int(value) for value in counts.values() if int(value) > 0]
    if not positive_counts:
        raise ValueError("Cannot compute case weights from empty counts")
    reference = float(median(positive_counts))
    weights = {}
    for case, count in counts.items():
        if count <= 0:
            continue
        raw_weight = math.sqrt(reference / float(count))
        weights[str(case)] = min(config.max_case_weight, max(config.min_case_weight, raw_weight))
    return weights


def bbox_area_ratio(bbox: tuple[float, float, float, float] | None, scale: float = 1000.0) -> float:
    if bbox is None:
        return 0.0
    x1, y1, x2, y2 = bbox
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return area / (scale * scale)


def type_reward(expected_types: Iterable[str], predicted_types: Iterable[str]) -> float:
    return set_f1(tuple(TYPE_ORDER.index(item) for item in expected_types), tuple(TYPE_ORDER.index(item) for item in predicted_types))


def evidence_reward(target: dict[str, Any], parsed: Any) -> float:
    expected_types = set(target["types"])
    has_image = bool(IMAGE_TYPES.intersection(expected_types))
    has_text = bool(TEXT_TYPES.intersection(expected_types))
    if target["verdict"] == "pristine":
        return 1.0 if parsed.image_bbox is None and not parsed.text_positions else 0.0

    scores = []
    if has_image:
        scores.append(bbox_iou(target.get("image_bbox"), parsed.image_bbox))
    if has_text:
        scores.append(set_f1(target.get("text_positions", []), parsed.text_positions))
    return sum(scores) / len(scores) if scores else 0.0


def hacking_penalty(target: dict[str, Any], parsed: Any, config: RewardConfig = RewardConfig()) -> float:
    if not parsed.valid:
        return config.max_penalty

    penalty = 0.0
    if target["verdict"] == "pristine" and parsed.verdict == "manipulated":
        penalty += config.false_alarm_penalty
    if set(parsed.types) == set(TYPE_ORDER):
        penalty += config.all_types_penalty
    if bbox_area_ratio(parsed.image_bbox) >= config.full_bbox_area_ratio:
        penalty += config.full_bbox_penalty
    if parsed.text_positions:
        ratio = len(parsed.text_positions) / max(1, config.max_words)
        if ratio >= config.full_text_ratio:
            penalty += config.full_text_penalty
    return min(config.max_penalty, penalty)


def score_response(
    target: dict[str, Any],
    response: Any,
    case_weights: dict[str, float] | None = None,
    config: RewardConfig = RewardConfig(),
) -> RewardBreakdown:
    current_case = case_name(target)
    current_weight = (case_weights or {}).get(current_case, 1.0)
    parsed = parse_response(response, max_words=config.max_words)
    if not parsed.valid:
        return RewardBreakdown(
            reward=-config.max_penalty,
            case=current_case,
            case_weight=current_weight,
            task_reward=0.0,
            verdict_reward=0.0,
            type_reward=0.0,
            evidence_reward=0.0,
            penalty=config.max_penalty,
            valid=False,
            error=parsed.error,
        )

    expected_types = tuple(target["types"])
    verdict_score = 1.0
    if parsed.verdict != target["verdict"]:
        verdict_score = -1.0 if target["verdict"] == "manipulated" else -0.5
    type_score = 1.0 if target["verdict"] == "pristine" and not parsed.types else type_reward(expected_types, parsed.types)
    evidence_score = evidence_reward(target, parsed)
    task_score = (
        config.weights.verdict * verdict_score
        + config.weights.manipulation_type * type_score
        + config.weights.evidence * evidence_score
    )
    penalty = hacking_penalty(target, parsed, config)
    return RewardBreakdown(
        reward=current_weight * task_score - penalty,
        case=current_case,
        case_weight=current_weight,
        task_reward=task_score,
        verdict_reward=verdict_score,
        type_reward=type_score,
        evidence_reward=evidence_score,
        penalty=penalty,
        valid=True,
        error=None,
    )


def completion_text(completion: Any) -> Any:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return completion.get("content", completion.get("response", completion.get("text", "")))
    if isinstance(completion, list) and completion:
        return completion_text(completion[0])
    return str(completion)


def _targets_from_kwargs(kwargs: dict[str, Any], count: int) -> list[dict[str, Any]]:
    raw_targets = kwargs.get("target")
    if raw_targets is None and "meta" in kwargs:
        raw_targets = [meta["target"] for meta in kwargs["meta"]]
    if raw_targets is None:
        raise ValueError("GRPO reward requires a target column or meta.target column")
    if isinstance(raw_targets, dict):
        raw_targets = [raw_targets] * count
    if len(raw_targets) != count:
        raise ValueError(f"target count does not match completions: {len(raw_targets)} vs {count}")
    return list(raw_targets)


def make_badcase_aware_reward(
    case_weights: dict[str, float] | None = None,
    config: RewardConfig = RewardConfig(),
) -> Callable[..., list[float]]:
    def reward_func(completions: list[Any], **kwargs: Any) -> list[float]:
        targets = _targets_from_kwargs(kwargs, len(completions))
        rewards = []
        for target, completion in zip(targets, completions):
            breakdown = score_response(target, completion_text(completion), case_weights, config)
            rewards.append(float(breakdown.reward))
        return rewards

    return reward_func
