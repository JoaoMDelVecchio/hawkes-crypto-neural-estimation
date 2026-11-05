#!/usr/bin/env python
"""Consolida jobs independentes sem reabrir ou recalcular o teste."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hawkes_crypto.config import load_experiment_config
from hawkes_crypto.eval.metrics import aggregate_seed_metrics
from hawkes_crypto.training import execution_metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/lee_local_optimized.yaml")
    )
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    summary: dict[str, object] = {
        "execution": execution_metadata(config),
        "lengths": {},
    }
    missing = []
    for length in config.data.sequence_lengths:
        length_result: dict[str, object] = {"models": {}}
        for model in ("lstm_lee", "transformer"):
            runs = []
            for seed in config.training.seeds:
                path = (
                    config.artifacts_root
                    / f"length_{length}"
                    / model
                    / f"seed_{seed}"
                    / "result.json"
                )
                if not path.exists():
                    missing.append(str(path))
                    continue
                runs.append(json.loads(path.read_text()))
            if runs:
                metrics = [run["evaluation"] for run in runs]
                aggregate = aggregate_seed_metrics(metrics)
                aggregate["runs"] = runs
                length_result["models"][model] = aggregate
        mle_path = config.artifacts_root / f"length_{length}" / "mle" / "summary.json"
        if mle_path.exists():
            length_result["mle"] = json.loads(mle_path.read_text())
        elif config.mle.enabled:
            missing.append(str(mle_path))
        summary["lengths"][str(length)] = length_result
    summary["missing"] = missing
    output = config.artifacts_root / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"output": str(output), "missing": len(missing)}, indent=2))


if __name__ == "__main__":
    main()
