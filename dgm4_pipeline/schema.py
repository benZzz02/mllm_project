from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any


TYPE_ORDER = ("face_swap", "face_attribute", "text_swap", "text_attribute")
IMAGE_TYPES = frozenset(("face_swap", "face_attribute"))
TEXT_TYPES = frozenset(("text_swap", "text_attribute"))

_TYPE_ALIASES = {
    "fs": "face_swap",
    "face_swap": "face_swap",
    "face swap": "face_swap",
    "fa": "face_attribute",
    "face_attribute": "face_attribute",
    "face attribute": "face_attribute",
    "ts": "text_swap",
    "text_swap": "text_swap",
    "text swap": "text_swap",
    "ta": "text_attribute",
    "text_attribute": "text_attribute",
    "text attribute": "text_attribute",
}

_PRISTINE_ALIASES = {"pristine", "real", "normal", "orig", "original", "authentic"}
_MANIPULATED_ALIASES = {"manipulated", "fake", "tampered", "forged", "fraud"}


@dataclass(frozen=True)
class ParsedResponse:
    valid: bool
    verdict: str | None
    types: tuple[str, ...]
    image_bbox: tuple[float, float, float, float] | None
    text_positions: tuple[int, ...]
    error: str | None = None


def normalize_caption(text: str, max_words: int = 50) -> str:
    caption = re.sub(r"([,.'!?\"()*#:;~])", "", str(text).lower())
    caption = caption.replace("-", " ").replace("/", " ").replace("<person>", "person")
    caption = re.sub(r"\s{2,}", " ", caption).strip()
    return " ".join(caption.split(" ")[:max_words])


def label_to_types(label: str) -> tuple[str, ...]:
    if label == "orig":
        return ()
    raw_types = str(label).split("&")
    normalized = []
    for raw_type in raw_types:
        key = raw_type.strip().lower()
        if key not in _TYPE_ALIASES:
            raise ValueError(f"Unknown DGM4 manipulation label: {raw_type}")
        normalized.append(_TYPE_ALIASES[key])
    return tuple(item for item in TYPE_ORDER if item in normalized)


def normalize_bbox(
    bbox: list[float] | tuple[float, ...] | None,
    width: int,
    height: int,
    scale: int = 1000,
) -> list[int] | None:
    if not bbox:
        return None
    if len(bbox) != 4 or width <= 0 or height <= 0:
        raise ValueError("A bbox requires four coordinates and a valid image size")
    x1, y1, x2, y2 = (float(value) for value in bbox)
    normalized = [x1 / width * scale, y1 / height * scale, x2 / width * scale, y2 / height * scale]
    return [int(round(min(scale, max(0.0, value)))) for value in normalized]


def build_target(annotation: dict[str, Any], width: int, height: int, max_words: int = 50) -> dict[str, Any]:
    types = label_to_types(str(annotation["fake_cls"]))
    positions = sorted({int(pos) for pos in annotation.get("fake_text_pos", []) if 0 <= int(pos) < max_words})
    bbox = normalize_bbox(annotation.get("fake_image_box"), width, height)
    if not IMAGE_TYPES.intersection(types):
        bbox = None
    if not TEXT_TYPES.intersection(types):
        positions = []
    return {
        "verdict": "pristine" if not types else "manipulated",
        "types": list(types),
        "image_bbox": bbox,
        "text_positions": positions,
    }


def build_prompt(caption: str) -> str:
    return (
        "<image>\n"
        f"News text: {caption}\n"
        "Determine whether this image-text pair is pristine or manipulated. "
        "Return exactly one JSON object with keys verdict, types, image_bbox, and text_positions. "
        "verdict must be pristine or manipulated. types must be selected from "
        "[face_swap, face_attribute, text_swap, text_attribute]. "
        "image_bbox must be null or [x1,y1,x2,y2] normalized to 0-1000. "
        "text_positions must contain zero-based word indices in the normalized News text. "
        "Do not include markdown or extra explanation."
    )


def canonical_answer(target: dict[str, Any]) -> str:
    ordered = {
        "verdict": target["verdict"],
        "types": list(target["types"]),
        "image_bbox": target["image_bbox"],
        "text_positions": list(target["text_positions"]),
    }
    return json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))


def _extract_json(text: str) -> dict[str, Any]:
    stripped = str(text).strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(stripped[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("The response JSON must be an object")
    return value


def _normalize_verdict(value: Any) -> str:
    verdict = str(value).strip().lower()
    if verdict in _PRISTINE_ALIASES:
        return "pristine"
    if verdict in _MANIPULATED_ALIASES:
        return "manipulated"
    raise ValueError(f"Unknown verdict: {value}")


def _normalize_types(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values = re.split(r"[,&|+]", value) if value.strip() else []
    elif isinstance(value, list):
        values = value
    else:
        raise ValueError("types must be a list or string")
    normalized = []
    for item in values:
        key = str(item).strip().lower()
        if key not in _TYPE_ALIASES:
            raise ValueError(f"Unknown manipulation type: {item}")
        normalized.append(_TYPE_ALIASES[key])
    return tuple(item for item in TYPE_ORDER if item in normalized)


def _normalize_response_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if value in (None, [], "null", "None"):
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("image_bbox must be null or a four-value list")
    values = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in values):
        raise ValueError("image_bbox contains a non-finite value")
    x1, y1, x2, y2 = values
    if x2 < x1 or y2 < y1:
        raise ValueError("image_bbox must follow x1 <= x2 and y1 <= y2")
    return tuple(min(1000.0, max(0.0, item)) for item in values)


def _normalize_positions(value: Any, max_words: int) -> tuple[int, ...]:
    if value in (None, []):
        return ()
    if not isinstance(value, list):
        raise ValueError("text_positions must be a list")
    positions = {int(item) for item in value}
    if any(item < 0 or item >= max_words for item in positions):
        raise ValueError(f"text_positions must be within [0, {max_words})")
    return tuple(sorted(positions))


def parse_response(response: Any, max_words: int = 50) -> ParsedResponse:
    if isinstance(response, dict):
        payload = response
    else:
        try:
            payload = _extract_json(str(response))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            return ParsedResponse(False, None, (), None, (), f"invalid_json: {exc}")
    try:
        verdict = _normalize_verdict(payload.get("verdict", payload.get("label")))
        types = _normalize_types(payload.get("types", payload.get("badcase_types", [])))
        bbox = _normalize_response_bbox(payload.get("image_bbox", payload.get("evidence_bbox")))
        positions = _normalize_positions(
            payload.get("text_positions", payload.get("evidence_text_pos", [])), max_words
        )
        if verdict == "pristine" and types:
            raise ValueError("pristine responses cannot contain manipulation types")
        if verdict == "manipulated" and not types:
            raise ValueError("manipulated responses require at least one type")
        if not IMAGE_TYPES.intersection(types) and bbox is not None:
            raise ValueError("image_bbox is present without an image manipulation type")
        if not TEXT_TYPES.intersection(types) and positions:
            raise ValueError("text_positions are present without a text manipulation type")
        return ParsedResponse(True, verdict, types, bbox, positions)
    except (TypeError, ValueError) as exc:
        return ParsedResponse(False, None, (), None, (), f"invalid_schema: {exc}")


def bbox_iou(
    first: tuple[float, float, float, float] | list[float] | None,
    second: tuple[float, float, float, float] | list[float] | None,
) -> float:
    if first is None and second is None:
        return 1.0
    if first is None or second is None:
        return 0.0
    ax1, ay1, ax2, ay2 = (float(item) for item in first)
    bx1, by1, bx2, by2 = (float(item) for item in second)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    if union <= 0:
        return 1.0 if area_a == 0 and area_b == 0 else 0.0
    return intersection / union


def set_f1(expected: list[int] | tuple[int, ...], predicted: list[int] | tuple[int, ...]) -> float:
    expected_set = set(expected)
    predicted_set = set(predicted)
    if not expected_set and not predicted_set:
        return 1.0
    true_positive = len(expected_set.intersection(predicted_set))
    precision = true_positive / len(predicted_set) if predicted_set else 0.0
    recall = true_positive / len(expected_set) if expected_set else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def compare_prediction(
    target: dict[str, Any],
    parsed: ParsedResponse,
    bbox_threshold: float = 0.5,
    text_f1_threshold: float = 1.0,
) -> list[str]:
    if not parsed.valid:
        return ["invalid_json_or_schema"]
    errors = []
    if parsed.verdict != target["verdict"]:
        errors.append("wrong_verdict")
    if set(parsed.types) != set(target["types"]):
        errors.append("wrong_types")
    if bbox_iou(target["image_bbox"], parsed.image_bbox) < bbox_threshold:
        errors.append("image_grounding")
    if set_f1(target["text_positions"], parsed.text_positions) < text_f1_threshold:
        errors.append("text_grounding")
    if target["image_bbox"] is None and parsed.image_bbox is not None:
        errors.append("hallucinated_image_evidence")
    if not target["text_positions"] and parsed.text_positions:
        errors.append("hallucinated_text_evidence")
    return sorted(set(errors))
