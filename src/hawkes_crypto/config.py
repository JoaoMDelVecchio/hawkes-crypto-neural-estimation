"""Configuração validada dos experimentos finais."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def _positive(value: Any, name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} deve ser positivo")
    return parsed


def _range(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} deve conter [mínimo, máximo]")
    lower, upper = map(float, value)
    if not 0 <= lower < upper:
        raise ValueError(f"Intervalo inválido em {name}")
    return lower, upper


@dataclass(frozen=True)
class DataConfig:
    sequence_lengths: tuple[int, ...]
    train_samples: int
    validation_samples: int
    test_samples: int
    shard_size: int


@dataclass(frozen=True)
class SamplingConfig:
    mu: tuple[float, float]
    beta: tuple[float, float]
    self_fraction: tuple[float, float]
    g_strata: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    patience: int
    seeds: tuple[int, ...]


@dataclass(frozen=True)
class MLEConfig:
    enabled: bool
    max_paths: int
    n_starts: int
    maxiter: int
    workers: int


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    seed: int
    synthetic_root: Path
    artifacts_root: Path
    data: DataConfig
    sampling: SamplingConfig
    training: TrainingConfig
    transformer: dict[str, int | float]
    mle: MLEConfig
    source_path: Path


def load_experiment_config(path: Path) -> ExperimentConfig:
    """Carrega YAML e falha cedo para configurações cientificamente ambíguas."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Configuração deve ser um mapeamento YAML")
    data = raw["data"]
    sampling = raw["sampling"]
    training = raw["training"]
    mle = raw["mle"]
    lengths = tuple(_positive(item, "sequence_length") for item in data["sequence_lengths"])
    seeds = tuple(int(item) for item in training["seeds"])
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("training.seeds deve conter seeds distintas")
    g_strata = tuple(_range(item, "sampling.g_strata") for item in sampling["g_strata"])
    if any(lower <= 0 or upper >= 1 for lower, upper in g_strata):
        raise ValueError("Todos os estratos de g devem estar estritamente entre 0 e 1")
    transformer = dict(raw["models"]["transformer"])
    if int(transformer["d_model"]) % int(transformer["nhead"]) != 0:
        raise ValueError("d_model deve ser divisível por nhead")
    return ExperimentConfig(
        name=str(raw["name"]),
        seed=int(raw["seed"]),
        synthetic_root=Path(raw["paths"]["synthetic_root"]),
        artifacts_root=Path(raw["paths"]["artifacts_root"]),
        data=DataConfig(
            sequence_lengths=lengths,
            train_samples=_positive(data["train_samples"], "train_samples"),
            validation_samples=_positive(data["validation_samples"], "validation_samples"),
            test_samples=_positive(data["test_samples"], "test_samples"),
            shard_size=_positive(data["shard_size"], "shard_size"),
        ),
        sampling=SamplingConfig(
            mu=_range(sampling["mu"], "sampling.mu"),
            beta=_range(sampling["beta"], "sampling.beta"),
            self_fraction=_range(sampling["self_fraction"], "sampling.self_fraction"),
            g_strata=g_strata,
        ),
        training=TrainingConfig(
            epochs=_positive(training["epochs"], "epochs"),
            batch_size=_positive(training["batch_size"], "batch_size"),
            learning_rate=float(training["learning_rate"]),
            patience=_positive(training["patience"], "patience"),
            seeds=seeds,
        ),
        transformer=transformer,
        mle=MLEConfig(
            enabled=bool(mle["enabled"]),
            max_paths=int(mle["max_paths"]),
            n_starts=_positive(mle["n_starts"], "mle.n_starts"),
            maxiter=_positive(mle["maxiter"], "mle.maxiter"),
            workers=_positive(mle["workers"], "mle.workers"),
        ),
        source_path=path.resolve(),
    )
