import numpy as np
import pandas as pd
import pytest

from hawkes_crypto.empirical import labeled_arrays


def test_labeled_arrays_select_exactly_one_split(tmp_path) -> None:
    data_root = tmp_path / "processed"
    symbol_root = data_root / "BTCUSDT"
    symbol_root.mkdir(parents=True)
    frame = pd.DataFrame(
        {
            "inter_arrival": [0.1, 0.2, 0.3, 0.4],
            "event_type": [1, 2, 1, 2],
        }
    )
    frame.to_parquet(symbol_root / "2024-01-01.parquet")
    records = [
        {
            "symbol": "BTCUSDT",
            "date": "2024-01-01",
            "split": split,
            "length": 2,
            "start_event": start,
            "end_event": start + 2,
            "converged": True,
            "parameters": {
                "mu": 0.3,
                "alpha_self": 0.2,
                "alpha_cross": 0.1,
                "beta": 0.8,
            },
        }
        for split, start in (("calibration_train", 0), ("test_closed", 2))
    ]

    inputs, targets, selected = labeled_arrays(
        records, data_root, split="calibration_train", length=2
    )

    assert len(selected) == 1
    assert selected[0]["split"] == "calibration_train"
    assert inputs[0, :, 0].tolist() == pytest.approx([0.1, 0.2])
    assert targets.shape == (1, 4)
    with pytest.raises(ValueError, match="Nenhuma janela"):
        labeled_arrays(records, data_root, split="validation", length=2)
    assert np.all(inputs[..., 1] == [[1, 2]])
