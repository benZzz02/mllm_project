#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Patch top-level keys in a LLaMA-Factory YAML file.")
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--set",
        dest="sets",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Top-level YAML override. Repeat for multiple values.",
    )
    return parser.parse_args()


def parse_overrides(items: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE, got: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Empty key in override: {item}")
        overrides[key] = value.strip()
    return overrides


def patch_yaml(template: Path, overrides: dict[str, str]) -> str:
    lines = template.read_text(encoding="utf-8").splitlines()
    remaining = dict(overrides)
    patched = []
    for line in lines:
        stripped = line.lstrip()
        if stripped and not line.startswith((" ", "\t", "#")) and ":" in stripped:
            key = stripped.split(":", 1)[0].strip()
            if key in remaining:
                patched.append(f"{key}: {remaining.pop(key)}")
                continue
        patched.append(line)

    if remaining:
        patched.append("")
        patched.append("### runtime overrides")
        for key, value in remaining.items():
            patched.append(f"{key}: {value}")
    return "\n".join(patched) + "\n"


def main() -> None:
    args = parse_args()
    overrides = parse_overrides(args.sets)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(patch_yaml(args.template, overrides), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
