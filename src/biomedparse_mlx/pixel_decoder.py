"""Transformer/FPN decoder adapted from BioMedParse (Meta copyright; see THIRD_PARTY_NOTICES.md)."""

from __future__ import annotations

from typing import cast

import mlx.core as mx
import mlx.nn as nn

from biomedparse_mlx.layers import EncoderLayer, position_encoding
from biomedparse_mlx.mlx_types import relu
from biomedparse_mlx.ontology import ModelConfig
from biomedparse_mlx.resize import ResizeMode, resize


class NormalizedConv(nn.Conv2d):
    """Bias-free NHWC convolution followed by GroupNorm (32 groups) and optional ReLU."""

    def __init__(self, inputs: int, outputs: int, kernel: int, *, activation: bool) -> None:
        """Build a padded convolution with 32-group normalization and optional ReLU."""
        super().__init__(inputs, outputs, kernel, padding=kernel // 2, bias=False)
        self.norm = nn.GroupNorm(32, outputs, pytorch_compatible=True)
        self._activation = activation

    def __call__(self, values: mx.array) -> mx.array:
        """Convolve and normalize NHWC features, then apply optional ReLU."""
        values = self.norm(super().__call__(values))
        return relu(values) if self._activation else values


class Encoder(nn.Module):
    """Stack of post-norm transformer encoder layers sharing one positional encoding."""

    def __init__(self, config: ModelConfig) -> None:
        """Construct the configured stack of transformer encoder layers."""
        super().__init__()
        self.layers = [
            EncoderLayer(config.hidden_dim, config.heads, config.ffn_dim)
            for _ in range(config.encoder_layers)
        ]

    def __call__(self, values: mx.array, position: mx.array) -> mx.array:
        """Apply each encoder layer with the same positional encoding."""
        for layer in self.layers:
            values = layer(values, position)
        return values


class EncoderOnly(nn.Module):
    """Run the transformer encoder over a flattened NHWC map with sine position encoding."""

    def __init__(self, config: ModelConfig) -> None:
        """Wrap the configured transformer encoder for spatial feature maps."""
        super().__init__()
        self.encoder = Encoder(config)

    def __call__(self, values: mx.array) -> mx.array:
        """Flatten NHWC positions for the encoder and restore the spatial shape."""
        batch, height, width, channels = values.shape
        position = position_encoding(height, width, channels)
        return mx.reshape(
            self.encoder(mx.reshape(values, (batch, height * width, channels)), position),
            (batch, height, width, channels),
        )


class PixelDecoder(nn.Module):
    """Transformer and FPN decoder for four NHWC backbone maps, finest first."""

    layer_4: NormalizedConv

    def __init__(self, config: ModelConfig) -> None:
        """Build the transformer, lateral FPN projections and mask-feature output."""
        super().__init__()
        width = config.hidden_dim
        self.input_proj = nn.Conv2d(config.channels[-1], width, 1)
        self.transformer = EncoderOnly(config)
        for i, inputs in enumerate(config.channels):
            if i != 3:
                setattr(
                    self, f"adapter_{i + 1}", NormalizedConv(inputs, width, 1, activation=False)
                )
            setattr(self, f"layer_{i + 1}", NormalizedConv(width, width, 3, activation=True))
        self.mask_features = nn.Conv2d(width, width, 3, padding=1)

    def __call__(self, features: tuple[mx.array, ...]) -> tuple[mx.array, tuple[mx.array, ...]]:
        """Fuse the four backbone maps (finest first) into FPN outputs.

        Returns the finest-resolution mask features (stride 4) and the three
        coarser scales (strides 32, 16, 8) used by the decoder, coarsest first.
        """
        values = self.layer_4(self.transformer(self.input_proj(features[-1])))
        scales = [values]
        for i in (2, 1, 0):
            adapter = cast(NormalizedConv, getattr(self, f"adapter_{i + 1}"))
            layer = cast(NormalizedConv, getattr(self, f"layer_{i + 1}"))
            lateral = adapter(features[i])
            values = layer(
                lateral + resize(values, (lateral.shape[1], lateral.shape[2]), ResizeMode.NEAREST),
            )
            if i != 0:
                scales.append(values)
        return self.mask_features(values), tuple(scales)
