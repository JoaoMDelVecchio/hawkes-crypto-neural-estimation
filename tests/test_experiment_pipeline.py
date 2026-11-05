import importlib.util
from pathlib import Path

import numpy as np
import pytest

from hawkes_crypto.config import load_experiment_config
from hawkes_crypto.eval.metrics import aggregate_seed_metrics, parameter_metrics
from hawkes_crypto.synthetic import generate_split, load_split


def _load_script(name: str):
    path = Path("scripts") / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Não foi possível importar {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generate_figures = _load_script("generate_final_figures").generate_figures
render_fragment = _load_script("generate_final_report_fragment").render_fragment
load_completed_records = _load_script("run_mle_incremental").load_completed_records


def test_final_config_freezes_required_protocol() -> None:
    config = load_experiment_config(Path("configs/lee_synthetic_final.yaml"))

    assert config.data.sequence_lengths == (1000,)
    assert config.data.train_samples == 75_000
    assert config.data.validation_samples == 10_000
    assert config.data.test_samples == 15_000
    assert config.training.epochs == 300
    assert len(config.training.seeds) == 5


def test_manifests_have_independent_splits(tmp_path) -> None:
    source = Path("configs/local_smoke.yaml").read_text()
    source = source.replace(
        "synthetic_root: data/synthetic/local_smoke",
        f"synthetic_root: {tmp_path / 'synthetic'}",
    )
    source = source.replace("sequence_lengths: [1000]", "sequence_lengths: [4]")
    source = source.replace("train_samples: 48", "train_samples: 6")
    source = source.replace("validation_samples: 16", "validation_samples: 3")
    source = source.replace("test_samples: 16", "test_samples: 3")
    source = source.replace("shard_size: 16", "shard_size: 3")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(source)
    config = load_experiment_config(config_path)
    generate_split(config, split="train", sequence_length=4, size=6)
    generate_split(config, split="test", sequence_length=4, size=3)
    train_x, train_y = load_split(config.synthetic_root, 4, "train")
    test_x, test_y = load_split(config.synthetic_root, 4, "test")

    assert train_x.shape == (6, 4, 2)
    assert train_y.shape == (6, 4)
    assert test_x.shape == (3, 4, 2)
    assert not np.array_equal(train_x[:3], test_x)
    assert np.all((train_y[:, 1] + train_y[:, 2]) / train_y[:, 3] < 1)
    assert np.all((test_y[:, 1] + test_y[:, 2]) / test_y[:, 3] < 1)


def test_parameter_metrics_report_per_parameter_and_validity() -> None:
    targets = np.asarray([[0.3, 0.4, 0.7, 1.5], [0.2, 0.1, 0.2, 0.8]])
    predictions = targets + 0.1

    metrics = parameter_metrics(predictions, targets)

    assert metrics["macro_mse"] == pytest.approx(0.01)
    assert metrics["per_parameter"]["mu"]["bias"] == pytest.approx(0.1)
    assert metrics["validity_rate"] == 1.0
    aggregate = aggregate_seed_metrics([metrics])
    assert aggregate["macro_mse_mean"] == pytest.approx(0.01)
    assert aggregate["macro_mae_mean"] == pytest.approx(0.1)
    assert aggregate["branching_ratio_mse_mean"] >= 0.0


def _final_summary_fixture() -> dict[str, object]:
    per_parameter = {
        name: {"mse_mean": 0.01, "mse_std": 0.001, "mae_mean": 0.08}
        for name in ("mu", "alpha_self", "alpha_cross", "beta")
    }
    regimes = {
        name: {
            "macro_mse_mean": 0.01,
            "macro_mse_std": 0.001,
            "macro_mae_mean": 0.08,
        }
        for name in ("low", "intermediate", "near_critical")
    }
    neural = {
        "macro_mse_mean": 0.01,
        "macro_mse_approx_95ci": [0.009, 0.011],
        "macro_mae_mean": 0.08,
        "branching_ratio_mse_mean": 0.002,
        "validity_rate_mean": 1.0,
        "per_parameter": per_parameter,
        "by_branching_ratio": regimes,
        "latency_batch_1_median_ms_mean": 2.0,
    }
    mle_parameter = {
        name: {"mse": 0.02, "mae": 0.1}
        for name in ("mu", "alpha_self", "alpha_cross", "beta")
    }
    mle_regimes = {
        name: {"macro_mse": 0.02}
        for name in ("low", "intermediate", "near_critical")
    }
    mle = {
        "macro_mse": 0.02,
        "macro_mae": 0.1,
        "branching_ratio": {"mse": 0.003},
        "validity_rate": 1.0,
        "convergence_rate": 0.99,
        "median_iterations": 20.0,
        "seconds_per_path_wall_clock": 0.5,
        "per_parameter": mle_parameter,
        "by_branching_ratio": mle_regimes,
    }
    return {
        "missing": [],
        "lengths": {
            "1000": {
                "models": {"lstm_lee": neural, "transformer": neural},
                "mle": mle,
            }
        },
    }


def test_final_report_and_figures_are_1000_event_only(tmp_path) -> None:
    summary = _final_summary_fixture()

    fragment = render_fragment(summary)
    outputs = generate_figures(summary, tmp_path)

    assert "1.000 eventos" in fragment
    assert "2.000 eventos" not in fragment
    assert "LSTM" in fragment
    assert "Transformer" in fragment
    assert "MLE" in fragment
    assert len(outputs) == 3
    assert all(path.exists() and path.stat().st_size > 0 for path in outputs)


def test_incremental_mle_rejects_corrupt_records(tmp_path) -> None:
    records = tmp_path / "records.jsonl"
    records.write_text('{"index": 0, "parameters": [1, 1, 1, 3]}\nnot-json\n')

    with pytest.raises(ValueError, match="Registro MLE inválido"):
        load_completed_records(records)
