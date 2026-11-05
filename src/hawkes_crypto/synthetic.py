"""Datasets sintéticos persistidos para os experimentos de recuperação."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from hawkes_crypto.config import ExperimentConfig
from hawkes_crypto.simulate.hawkes_sim import HawkesParameters, simulate_path

SPLIT_OFFSETS = {"train": 0, "validation": 1_000_000, "test": 2_000_000}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_parameters(
    config: ExperimentConfig, rng: np.random.Generator, index: int
) -> HawkesParameters:
    """Amostra de forma balanceada entre regiões pré-registradas de g."""
    lower_g, upper_g = config.sampling.g_strata[index % len(config.sampling.g_strata)]
    mu = rng.uniform(*config.sampling.mu)
    beta = rng.uniform(*config.sampling.beta)
    branching_ratio = rng.uniform(lower_g, upper_g)
    self_fraction = rng.uniform(*config.sampling.self_fraction)
    total_alpha = branching_ratio * beta
    return HawkesParameters(
        mu=float(mu),
        alpha_self=float(total_alpha * self_fraction),
        alpha_cross=float(total_alpha * (1.0 - self_fraction)),
        beta=float(beta),
    )


def generate_split(
    config: ExperimentConfig,
    *,
    split: str,
    sequence_length: int,
    size: int,
    overwrite: bool = False,
) -> list[dict[str, object]]:
    if split not in SPLIT_OFFSETS:
        raise ValueError(f"Split desconhecido: {split}")
    output = config.synthetic_root / f"length_{sequence_length}" / split
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.jsonl"
    if manifest_path.exists() and not overwrite:
        return [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line
        ]

    rng = np.random.default_rng(config.seed + SPLIT_OFFSETS[split] + sequence_length)
    records: list[dict[str, object]] = []
    for shard_start in range(0, size, config.data.shard_size):
        shard_size = min(config.data.shard_size, size - shard_start)
        inputs = np.empty((shard_size, sequence_length, 2), dtype=np.float32)
        targets = np.empty((shard_size, 4), dtype=np.float32)
        sample_seeds = np.empty(shard_size, dtype=np.uint64)
        for local_index in range(shard_size):
            global_index = shard_start + local_index
            parameters = sample_parameters(config, rng, global_index)
            sample_seed = int(rng.integers(0, 2**32, dtype=np.uint64))
            path = simulate_path(parameters, sequence_length, seed=sample_seed)
            inputs[local_index, :, 0] = path.inter_arrivals
            inputs[local_index, :, 1] = path.event_types
            targets[local_index] = [
                parameters.mu,
                parameters.alpha_self,
                parameters.alpha_cross,
                parameters.beta,
            ]
            sample_seeds[local_index] = sample_seed
        shard_path = output / f"shard_{shard_start:06d}.npz"
        np.savez(shard_path, inputs=inputs, targets=targets, seeds=sample_seeds)
        records.append(
            {
                "path": str(shard_path.resolve()),
                "split": split,
                "sequence_length": sequence_length,
                "start": shard_start,
                "samples": shard_size,
                "sha256": _sha256(shard_path),
            }
        )
    manifest_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    return records


def manifest_records(root: Path, sequence_length: int, split: str) -> list[dict[str, object]]:
    path = root / f"length_{sequence_length}" / split / "manifest.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Manifesto ausente: {path}")
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def iter_arrays(
    root: Path, sequence_length: int, split: str
) -> Iterator[tuple[NDArray[np.float32], NDArray[np.float32]]]:
    for record in manifest_records(root, sequence_length, split):
        with np.load(Path(str(record["path"]))) as shard:
            yield shard["inputs"], shard["targets"]


def load_split(
    root: Path, sequence_length: int, split: str
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    arrays = list(iter_arrays(root, sequence_length, split))
    return (
        np.concatenate([inputs for inputs, _ in arrays]),
        np.concatenate([targets for _, targets in arrays]),
    )
