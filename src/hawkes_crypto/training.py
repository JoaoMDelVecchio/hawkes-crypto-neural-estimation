"""Treinamento eficiente, determinístico e retomável dos estimadores neurais."""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from hawkes_crypto.config import ExperimentConfig
from hawkes_crypto.eval.metrics import (
    group_indices_by_branching_ratio,
    latency_metrics,
    parameter_metrics,
)
from hawkes_crypto.models.attention_encoder import HawkesTransformer
from hawkes_crypto.models.lstm_baseline import LeeLSTM
from hawkes_crypto.synthetic import load_split


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def selected_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS solicitado, mas indisponível")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA solicitado, mas indisponível")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(name: str, config: ExperimentConfig) -> nn.Module:
    if name == "lstm_lee":
        return LeeLSTM()
    if name == "transformer":
        return HawkesTransformer(
            d_model=int(config.transformer["d_model"]),
            nhead=int(config.transformer["nhead"]),
            num_layers=int(config.transformer["num_layers"]),
            dim_feedforward=int(config.transformer["dim_feedforward"]),
            dropout=float(config.transformer["dropout"]),
        )
    raise ValueError(f"Modelo desconhecido: {name}")


def _loader(
    inputs: np.ndarray,
    targets: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataset = TensorDataset(torch.from_numpy(inputs), torch.from_numpy(targets))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )


def _move_optimizer(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _checkpoint_payload(
    *,
    name: str,
    sequence_length: int,
    seed: int,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    best_loss: float,
    stale_epochs: int,
    history: list[dict[str, float | int]],
) -> dict[str, Any]:
    return {
        "model": name,
        "sequence_length": sequence_length,
        "seed": seed,
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_validation_mse": best_loss,
        "stale_epochs": stale_epochs,
        "history": history,
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
    }


def _restore_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, float, int, list[dict[str, float | int]]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    _move_optimizer(optimizer, device)
    random.setstate(payload["python_random_state"])
    np.random.set_state(payload["numpy_random_state"])
    torch.set_rng_state(payload["torch_random_state"].cpu())
    return (
        int(payload["epoch"]) + 1,
        float(payload["best_validation_mse"]),
        int(payload["stale_epochs"]),
        list(payload["history"]),
    )


def _loss_over_loader(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> float:
    squared_error = torch.zeros((), device=device)
    values = 0
    model.eval()
    with torch.inference_mode():
        for batch_x, batch_y in loader:
            expected = batch_y.to(device)
            predicted = model(batch_x.to(device))
            squared_error += torch.sum((predicted - expected) ** 2)
            values += expected.numel()
    synchronize(device)
    return float(squared_error.cpu()) / values


def train_model(
    name: str,
    config: ExperimentConfig,
    *,
    sequence_length: int,
    seed: int,
    device_name: str = "auto",
    batch_size_override: int | None = None,
    cpu_threads: int | None = None,
    resume: bool = True,
    compile_model: bool = False,
) -> tuple[nn.Module, dict[str, object], Path]:
    set_deterministic_seed(seed)
    if cpu_threads is not None:
        torch.set_num_threads(cpu_threads)
    device = selected_device(device_name)
    model = build_model(name, config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    effective_batch_size = (
        int(config.transformer.get("batch_size", config.training.batch_size))
        if name == "transformer"
        else config.training.batch_size
    )
    batch_size = batch_size_override or effective_batch_size
    if effective_batch_size % batch_size != 0:
        raise ValueError("Batch físico deve dividir exatamente o batch efetivo")
    accumulation_steps = effective_batch_size // batch_size
    run_dir = config.artifacts_root / f"length_{sequence_length}" / name / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"
    complete_path = run_dir / "complete.json"

    train_x, train_y = load_split(config.synthetic_root, sequence_length, "train")
    validation_x, validation_y = load_split(
        config.synthetic_root, sequence_length, "validation"
    )
    validation_loader = _loader(
        validation_x,
        validation_y,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )

    history: list[dict[str, float | int]] = []
    best_loss = float("inf")
    stale_epochs = 0
    start_epoch = 1
    if resume and last_path.exists():
        start_epoch, best_loss, stale_epochs, history = _restore_checkpoint(
            last_path, model, optimizer, device
        )
    execution_model = (
        cast(nn.Module, torch.compile(model)) if compile_model else model
    )
    started = time.perf_counter()
    for epoch in range(start_epoch, config.training.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        squared_error = torch.zeros((), device=device)
        values = 0
        train_loader = _loader(
            train_x,
            train_y,
            batch_size=batch_size,
            shuffle=True,
            seed=seed + epoch,
        )
        optimizer.zero_grad(set_to_none=True)
        pending_steps = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            prediction = execution_model(batch_x)
            loss = nn.functional.mse_loss(prediction, batch_y)
            (loss / accumulation_steps).backward()
            pending_steps += 1
            if pending_steps == accumulation_steps:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                pending_steps = 0
            squared_error += torch.sum((prediction.detach() - batch_y) ** 2)
            values += batch_y.numel()
        if pending_steps:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        synchronize(device)
        train_loss = float(squared_error.cpu()) / values
        validation_loss = _loss_over_loader(execution_model, validation_loader, device)
        epoch_seconds = time.perf_counter() - epoch_started
        history.append(
            {
                "epoch": epoch,
                "train_mse": train_loss,
                "validation_mse": validation_loss,
                "seconds": epoch_seconds,
                "samples_per_second": len(train_x) / epoch_seconds,
            }
        )
        improved = validation_loss < best_loss
        if improved:
            best_loss = validation_loss
            stale_epochs = 0
        else:
            stale_epochs += 1
        checkpoint = _checkpoint_payload(
            name=name,
            sequence_length=sequence_length,
            seed=seed,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            best_loss=best_loss,
            stale_epochs=stale_epochs,
            history=history,
        )
        _atomic_torch_save(checkpoint, last_path)
        if improved:
            _atomic_torch_save(checkpoint, best_path)
        print(
            json.dumps(
                {
                    "model": name,
                    "length": sequence_length,
                    "seed": seed,
                    "epoch": epoch,
                    "train_mse": train_loss,
                    "validation_mse": validation_loss,
                    "seconds": epoch_seconds,
                    "stale": stale_epochs,
                }
            ),
            flush=True,
        )
        if stale_epochs >= config.training.patience:
            break

    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["state_dict"])
    metadata: dict[str, object] = {
        "model": name,
        "sequence_length": sequence_length,
        "seed": seed,
        "device": str(device),
        "batch_size": batch_size,
        "effective_batch_size": effective_batch_size,
        "gradient_accumulation_steps": accumulation_steps,
        "compiled": compile_model,
        "train_seconds_this_process": time.perf_counter() - started,
        "best_validation_mse": float(best["best_validation_mse"]),
        "best_epoch": int(best["epoch"]),
        "epochs_completed": len(history),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "history": history,
    }
    complete_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return model, metadata, best_path


def evaluate_model(
    model: nn.Module,
    config: ExperimentConfig,
    *,
    sequence_length: int,
    batch_size: int | None = None,
    latency_repetitions: int = 20,
    compile_model: bool = False,
) -> dict[str, object]:
    device = next(model.parameters()).device
    inputs, expected = load_split(config.synthetic_root, sequence_length, "test")
    evaluation_batch = batch_size or config.training.batch_size
    loader = _loader(
        inputs, expected, batch_size=evaluation_batch, shuffle=False, seed=config.seed
    )
    predictions: list[np.ndarray] = []
    model.eval()
    execution_model = torch.compile(model) if compile_model else model
    with torch.inference_mode():
        for batch_x, _ in loader:
            predictions.append(execution_model(batch_x.to(device)).cpu().numpy())
        sample = torch.from_numpy(inputs[:1]).to(device)
        _ = execution_model(sample)
        synchronize(device)
        timings = []
        for _ in range(latency_repetitions):
            started = time.perf_counter()
            _ = execution_model(sample)
            synchronize(device)
            timings.append((time.perf_counter() - started) * 1000)
    all_predictions = np.concatenate(predictions)
    result = parameter_metrics(all_predictions, expected)
    result["by_branching_ratio"] = {
        name: parameter_metrics(all_predictions[indices], expected[indices])
        for name, indices in group_indices_by_branching_ratio(expected).items()
        if len(indices)
    }
    result["evaluation_batch_size"] = evaluation_batch
    result["compiled"] = compile_model
    result["latency_batch_1"] = latency_metrics(timings)
    return result


def execution_metadata(config: ExperimentConfig) -> dict[str, object]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
    return {
        "config": asdict(config),
        "git_commit": commit,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
    }
