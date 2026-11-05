#!/usr/bin/env python
"""Executa uma combinação modelo/comprimento/seed de forma retomável."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hawkes_crypto.config import load_experiment_config
from hawkes_crypto.training import evaluate_model, train_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/lee_local_optimized.yaml")
    )
    parser.add_argument("--model", choices=["lstm_lee", "transformer"], required=True)
    parser.add_argument("--length", type=int, choices=[1000], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--cpu-threads", type=int)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    config = load_experiment_config(args.config)
    run_dir = (
        config.artifacts_root
        / f"length_{args.length}"
        / args.model
        / f"seed_{args.seed}"
    )
    result_path = run_dir / "result.json"
    if result_path.exists():
        print(json.dumps({"status": "already_complete", "result": str(result_path)}))
        return
    model, training, checkpoint = train_model(
        args.model,
        config,
        sequence_length=args.length,
        seed=args.seed,
        device_name=args.device,
        batch_size_override=args.batch_size,
        cpu_threads=args.cpu_threads,
        resume=not args.no_resume,
        compile_model=args.compile,
    )
    batch_size = args.batch_size or (
        int(config.transformer["batch_size"])
        if args.model == "transformer"
        else config.training.batch_size
    )
    evaluation = evaluate_model(
        model,
        config,
        sequence_length=args.length,
        batch_size=batch_size,
        compile_model=args.compile,
    )
    payload = {
        "model": args.model,
        "length": args.length,
        "seed": args.seed,
        "checkpoint": str(checkpoint),
        "training": training,
        "evaluation": evaluation,
    }
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "result": str(result_path)}))


if __name__ == "__main__":
    main()
