import math
from typing import Optional as Op

import torch
from .util import ModelInterface, nn_dataclass
from torch import nn


@nn_dataclass
class Transformer(ModelInterface):
    """Casual, decoder-only transformer without embeddings or final softmax"""

    N: int = 6  # number of decoder layers
    d_model: int = 512  # dimension all embeddings
    d_ff: int = 2048  # dimension of feedforward layer
    h: int = 8  # number of heads
    max_len: int = 512  # length of input sequence
    dropout: float = 0.1  # dropout rate

    def __post_init__(self):
        self.pos_encoding = PosEncoding(self)
        self.pos_encoding_dropout = nn.Dropout(self.dropout)
        self.decoder_layers = nn.ModuleList([DecoderLayer(self) for _ in range(self.N)])
        self.feed_forward = FeedForward(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """input/output size:
        (..., seq_len, d_model)
        """
        assert x.shape[-1] == self.d_model, f"Input embedding size mismatch {x.shape}"
        if self.training:
            assert (
                x.shape[-2] <= self.max_len
            ), f"Input sequence length exceeds max_len {x.shape} during training"

        output: torch.Tensor = torch.empty_like(x)
        output[..., : self.max_len, :] = self._forward(x[..., : self.max_len, :])

        for i in range(0, x.shape[-2] - self.max_len, 1):
            start = i + 1
            end = start + self.max_len
            output[..., end - 1, :] = self._forward(x[..., start:end, :])[..., -1, :]

        return output

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos_encoding(x)
        x = self.pos_encoding_dropout(x)
        x = self.decode(x)
        x = self.feed_forward(x)
        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.decoder_layers:
            x = layer(x)
        return x

    def get_attn(self, layer: int):
        """Visualization function to get attention weights"""
        return self.decoder_layers[layer].attn


class PosEncoding(nn.Module):
    """Positional encoding for input embeddings"""

    def __init__(self, config: Transformer):
        super().__init__()
        pe = torch.empty(config.max_len, config.d_model)
        position = torch.arange(0, config.max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, config.d_model, 2) * -(math.log(10000.0) / config.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """input/output size:
        (..., seq_len, d_model)
        """
        return x + self.pe[: x.shape[-2], :].requires_grad_(False)


class DecoderLayer(nn.Module):
    """Single decoder layer with multi-head attention and feedforward layer"""

    def __init__(self, config: Transformer):
        super().__init__()
        self.attn = MultiHeadAttention(config)
        self.act0 = nn.LeakyReLU()
        self.dropout0 = nn.Dropout(config.dropout)
        self.norm0 = nn.LayerNorm(config.d_model)

        self.feed_forward = FeedForward(config)
        self.act1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(config.dropout)
        self.norm1 = nn.LayerNorm(config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """input/output size:
        (..., seq_len, d_model)
        """
        x_orig = x
        x = self.attn(x)
        x = self.act0(x)
        x = self.dropout0(x)
        x = x + x_orig
        x = self.norm0(x)

        x_orig = x
        x = self.feed_forward(x)
        x = self.act1(x)
        x = self.dropout1(x)
        x = x + x_orig
        x = self.norm1(x)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head attention layer with casual masked and scaled dot-product attention"""

    def __init__(self, config: Transformer):
        super().__init__()
        self.h = config.h

        class FC(nn.Module):
            """Single fully connected residual layer with dropout"""

            def __init__(self, config: Transformer):
                super().__init__()
                self.fc = nn.Linear(config.d_model, config.d_model)
                self.dropout = nn.Dropout(config.dropout)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.dropout(x + self.fc(x))

        self.Q = FC(config)
        self.K = FC(config)
        self.V = FC(config)
        self.fc = FC(config)

        self.d_h = config.d_model // config.h

        self.attn_dropout = nn.Dropout(config.dropout)
        self.attn: Op[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """input/output size:
        (..., seq_len, d_model)
        """
        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)

        # Split Q, K, and V to be fed into each head (..., seq_len, d_model) to (..., h, seq_len, d_h)
        q, k, v = (
            arr.view(*x.shape[:-1], self.h, self.d_h).transpose(-2, -3)
            for arr in (q, k, v)
        )

        v_prime = self.attend(q, k, v)

        # Combine last 2 dimensions back to (..., seq_len, d_model)
        v_prime = v_prime.transpose(-2, -3).reshape_as(x)

        v_prime = self.fc(v_prime)

        return v_prime

    def attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """input/output size:
        (..., seq_len, d_h)
        """
        # (..., seq_len, d_h) @ (..., seq_len, d_h)T -> (..., seq_len, seq_len)
        attn = q @ k.transpose(-2, -1)
        attn = attn / math.sqrt(self.d_h)

        attn = self.attn_dropout(attn)
        attn = self.mask(attn)
        attn = nn.functional.softmax(attn, dim=-1)

        # (..., seq_len, seq_len(weights)) @ (..., seq_len, d_h) -> (..., seq_len, d_h
        v_prime = attn @ v
        self.attn = attn
        return v_prime

    def mask(self, x: torch.Tensor) -> torch.Tensor:
        """input/output size:
        (..., seq_len, seq_len(weights))
        """
        mask = torch.triu(torch.ones_like(x, dtype=torch.bool), diagonal=1)
        return x.masked_fill(mask, -1e9)


class FeedForward(nn.Module):
    """Feedforward layer with single hidden layer and a LeakyReLU activation
    Used in decoder layers and final output layer
    """

    def __init__(self, config: Transformer):
        super().__init__()
        self.fc0 = nn.Linear(config.d_model, config.d_ff)
        self.batch_norm0 = nn.BatchNorm1d(config.d_ff, affine=False, track_running_stats=False)
        self.act0 = nn.LeakyReLU()
        self.dropout0 = nn.Dropout(config.dropout)
        self.fc1 = nn.Linear(config.d_ff, config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """input/output size:
        (..., seq_len, d_model)
        """
        x_orig = x
        x = self.fc0(x)
        if sum(x.shape[:-1]) > 1:
            x = self.batch_norm0(x)
        x = self.act0(x)
        x = self.dropout0(x)
        x = self.fc1(x)
        x = x + x_orig
        return x
