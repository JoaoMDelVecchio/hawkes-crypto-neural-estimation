#!/usr/bin/env python
"""Cria índice cronológico de janelas sem cruzar fronteiras de split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def split_for(day: str) -> str:
    if day <= "2024-01-22":
        return "calibration_train"
    if day <= "2024-01-25":
        return "validation"
    return "test_closed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/processed/um_futures"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/window_index.jsonl"))
    args = parser.parse_args()
    rows: list[dict[str, object]] = []
    for path in sorted(args.root.glob("*/*.parquet")):
        day = path.stem
        symbol = path.parent.name
        events = pd.read_parquet(path, columns=["timestamp", "event_type", "inter_arrival"])
        length = 1000
        for start in range(0, len(events) - length + 1, length):
            window = events.iloc[start : start + length]
            rows.append(
                {
                    "symbol": symbol,
                    "date": day,
                    "split": split_for(day),
                    "length": length,
                    "start_event": start,
                    "end_event": start + length,
                    "start_timestamp": str(window["timestamp"].iloc[0]),
                    "end_timestamp": str(window["timestamp"].iloc[-1]),
                    "mle_status": "pending",
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"windows": len(rows), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
