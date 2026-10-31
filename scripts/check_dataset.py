#!/usr/bin/env python3
"""Quick, non-destructive validation of the published pilot."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


DEFAULT_ROOT = Path("/srv/gibbon-activations/gpt_oss_pilot_3games")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(os.environ.get("GIBBON_DATA_ROOT", DEFAULT_ROOT)),
    )
    args = parser.parse_args()

    root = args.root
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"Cannot read pilot manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text())
    print(f"Dataset: {manifest['name']}")
    print(f"Root: {root}")
    print(f"Games: {manifest['totals']['games']}")
    print(f"Tokens: {manifest['totals']['tokens']:,}")
    print(f"Activation rows: {manifest['totals']['activation_vectors']:,}")

    errors: list[str] = []
    for game in manifest["games"]:
        activation = root / game["activation_file"]
        log = root / "logs" / f"{game['game_id']}.json"
        prompts = list((root / "prompts").glob(f"prompt_{game['game_id']}_*.txt"))
        actual_size = activation.stat().st_size if activation.is_file() else None
        ok = actual_size == game["bytes"] and log.is_file() and len(prompts) == 16
        print(
            f"{game['game_id']}: {'OK' if ok else 'ERROR'}; "
            f"activation={actual_size!r} bytes; prompts={len(prompts)}"
        )
        if not ok:
            errors.append(game["game_id"])

    if errors:
        raise SystemExit("Validation failed for: " + ", ".join(errors))


if __name__ == "__main__":
    main()

