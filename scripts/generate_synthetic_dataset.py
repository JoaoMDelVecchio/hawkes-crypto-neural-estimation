#!/usr/bin/env python
"""Gera shards sintéticos independentes para treino, validação e teste."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hawkes_crypto.config import load_experiment_config
from hawkes_crypto.synthetic import generate_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/local_smoke.yaml"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    sizes = {
        "train": config.data.train_samples,
        "validation": config.data.validation_samples,
        "test": config.data.test_samples,
    }
    summary: dict[str, int] = {}
    for length in config.data.sequence_lengths:
        for split, size in sizes.items():
            records = generate_split(
                config,
                split=split,
                sequence_length=length,
                size=size,
                overwrite=args.overwrite,
            )
            summary[f"{length}/{split}"] = sum(int(item["samples"]) for item in records)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
