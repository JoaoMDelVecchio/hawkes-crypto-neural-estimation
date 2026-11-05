#!/usr/bin/env python
"""Agenda jobs independentes entre CPU e MPS, com retomada por job."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from hawkes_crypto.config import load_experiment_config

SUPPORTED_SEQUENCE_LENGTH = 1000


def run_queue(
    jobs: list[tuple[str, int, int, str, bool, int]],
    *,
    config_path: Path,
    logs_root: Path,
    dry_run: bool,
) -> None:
    for model, length, seed, device, compiled, batch_size in jobs:
        log_path = logs_root / f"{model}_length{length}_seed{seed}_{device}.log"
        command = [
            sys.executable,
            "scripts/run_training_job.py",
            "--config",
            str(config_path),
            "--model",
            model,
            "--length",
            str(length),
            "--seed",
            str(seed),
            "--device",
            device,
            "--batch-size",
            str(batch_size),
        ]
        if device == "cpu":
            command.extend(["--cpu-threads", "6"])
        if compiled:
            command.append("--compile")
        print(json.dumps({"job": command, "log": str(log_path)}), flush=True)
        if dry_run:
            continue
        logs_root.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/lee_local_optimized.yaml")
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path("configs/local_execution_profile.json"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    profile = json.loads(args.profile.read_text())
    queues: dict[str, list[tuple[str, int, int, str, bool, int]]] = {
        "cpu": [],
        "mps": [],
    }
    for model in ("lstm_lee", "transformer"):
        for length in config.data.sequence_lengths:
            if length != SUPPORTED_SEQUENCE_LENGTH:
                print(
                    json.dumps(
                        {
                            "skipped_length": length,
                            "reason": "only 1000-event experiments are supported",
                        }
                    ),
                    flush=True,
                )
                continue
            for seed in config.training.seeds:
                device = str(profile["devices"][model][str(length)])
                compiled = bool(profile["compile"][model][str(length)])
                batch_size = int(profile["physical_batch_size"][model][str(length)])
                queues[device].append(
                    (model, length, seed, device, compiled, batch_size)
                )
    logs = config.artifacts_root / "logs"
    active = [(device, jobs) for device, jobs in queues.items() if jobs]
    with ThreadPoolExecutor(max_workers=len(active)) as executor:
        futures = [
            executor.submit(
                run_queue,
                jobs,
                config_path=args.config,
                logs_root=logs,
                dry_run=args.dry_run,
            )
            for _, jobs in active
        ]
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()
