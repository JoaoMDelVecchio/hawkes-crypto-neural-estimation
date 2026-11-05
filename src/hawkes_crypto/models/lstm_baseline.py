"""LSTM de referência descrita por Lee (2023)."""

from __future__ import annotations

from torch import Tensor, nn

from hawkes_crypto.models.common import StableParameterHead, prepare_sequence


class LeeLSTM(nn.Module):
    """Duas camadas LSTM de 12 unidades e uma saída paramétrica estável."""

    def __init__(self, hidden_size: int = 12) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=2,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
        )
        self.head = StableParameterHead(hidden_size)

    def forward(self, inputs: Tensor) -> Tensor:
        sequence = prepare_sequence(inputs)
        _, (hidden, _) = self.lstm(sequence)
        return self.head(hidden[-1])
