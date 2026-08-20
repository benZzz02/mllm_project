from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def load_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        first = handle.read(1)
        handle.seek(0)
        if first == "[":
            records = json.load(handle)
        else:
            records = [json.loads(line) for line in handle if line.strip()]
    if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
        raise ValueError(f"Expected a JSON array or JSONL objects in {source}")
    return records


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> int:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
