"""Encoder Transformer para estimação dos quatro parâmetros."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from hawkes_crypto.models.common import StableParameterHead, prepare_sequence


class SinusoidalEncoding(nn.Module):
    encoding: Tensor

    def __init__(self, d_model: int, max_length: int = 4096) -> None:
        super().__init__()
        positions = torch.arange(max_length).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10_000.0) / d_model)
        )
        encoding = torch.zeros(max_length, d_model)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies)
        self.register_buffer("encoding", encoding, persistent=False)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.shape[1] > self.encoding.shape[0]:
            raise ValueError("Sequência excede max_length do encoding")
        return inputs + self.encoding[: inputs.shape[1]]


class HawkesTransformer(nn.Module):
    """Baseline de atenção; ablações devem manter o protocolo congelado."""

    def __init__(
        self,
        *,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        max_length: int = 4096,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(2, d_model)
        self.position = SinusoidalEncoding(d_model, max_length)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = StableParameterHead(d_model)

    def forward(
        self, inputs: Tensor, padding_mask: Tensor | None = None
    ) -> Tensor:
        sequence = self.position(self.input_projection(prepare_sequence(inputs)))
        encoded = self.norm(
            self.encoder(sequence, src_key_padding_mask=padding_mask)
        )
        if padding_mask is None:
            pooled = encoded.mean(dim=1)
        else:
            valid = (~padding_mask).unsqueeze(-1)
            pooled = (encoded * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        return self.head(pooled)
