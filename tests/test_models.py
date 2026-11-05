import torch

from hawkes_crypto.models.attention_encoder import HawkesTransformer
from hawkes_crypto.models.lstm_baseline import LeeLSTM


def _assert_valid_parameters(output: torch.Tensor) -> None:
    assert output.shape == (3, 4)
    assert torch.all(output > 0)
    branching = (output[:, 1] + output[:, 2]) / output[:, 3]
    assert torch.all(branching < 1)


def test_lstm_output_respects_hawkes_constraints() -> None:
    inputs = torch.zeros(3, 20, 2)
    inputs[..., 0] = torch.rand(3, 20)
    inputs[..., 1] = torch.randint(1, 3, (3, 20)).float()
    _assert_valid_parameters(LeeLSTM()(inputs))


def test_transformer_output_respects_hawkes_constraints() -> None:
    inputs = torch.zeros(3, 20, 2)
    inputs[..., 0] = torch.rand(3, 20)
    inputs[..., 1] = torch.randint(1, 3, (3, 20)).float()
    model = HawkesTransformer(d_model=16, nhead=4, num_layers=1)
    _assert_valid_parameters(model(inputs))
