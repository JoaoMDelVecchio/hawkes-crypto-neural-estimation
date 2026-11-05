"""Utilitários para janelas empíricas com separação cronológica rígida."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

PARAMETER_NAMES = ("mu", "alpha_self", "alpha_cross", "beta")


def read_window_index(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def window_events(
    record: dict[str, Any], data_root: Path, *, columns: list[str] | None = None
) -> pd.DataFrame:
    path = data_root / str(record["symbol"]) / f"{record['date']}.parquet"
    frame = pd.read_parquet(path, columns=columns)
    start = int(record.get("start_event", 0))
    end = int(record.get("end_event", start + int(record["length"])))
    window = frame.iloc[start:end].reset_index(drop=True)
    if len(window) != int(record["length"]):
        raise ValueError(f"Janela incompleta: {record}")
    return window


def labeled_arrays(
    records: list[dict[str, Any]],
    data_root: Path,
    *,
    split: str,
    length: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32], list[dict[str, Any]]]:
    selected = [
        record
        for record in records
        if record.get("split") == split
        and int(record.get("length", 0)) == length
        and record.get("converged")
        and "parameters" in record
    ]
    if not selected:
        raise ValueError(f"Nenhuma janela válida para split={split}, length={length}")
    inputs = np.empty((len(selected), length, 2), dtype=np.float32)
    targets = np.empty((len(selected), 4), dtype=np.float32)
    for index, record in enumerate(selected):
        events = window_events(
            record, data_root, columns=["inter_arrival", "event_type"]
        )
        inputs[index] = events[["inter_arrival", "event_type"]].to_numpy(np.float32)
        parameters = record["parameters"]
        targets[index] = [float(parameters[name]) for name in PARAMETER_NAMES]
    return inputs, targets, selected
