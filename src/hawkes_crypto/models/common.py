"""Transformações compartilhadas pelos estimadores neurais."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def prepare_sequence(inputs: Tensor) -> Tensor:
    """Normaliza a representação sem usar estatísticas do conjunto de teste."""
    if inputs.ndim != 3 or inputs.shape[-1] != 2:
        raise ValueError("Entrada esperada: [batch, sequence, 2]")
    inter_arrival = torch.log1p(inputs[..., 0].clamp_min(0.0))
    event_type = torch.where(inputs[..., 1] == 1, 1.0, -1.0)
    return torch.stack((inter_arrival, event_type), dim=-1)


class StableParameterHead(nn.Module):
    """Mapeia quatro logits para parâmetros positivos e subcríticos."""

    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_size, 4)
        self.softplus = nn.Softplus()

    def forward(self, hidden: Tensor) -> Tensor:
        positive = self.softplus(self.linear(hidden))
        mu, alpha_self, alpha_cross, margin = positive.unbind(dim=-1)
        beta = alpha_self + alpha_cross + margin
        return torch.stack((mu, alpha_self, alpha_cross, beta), dim=-1)
