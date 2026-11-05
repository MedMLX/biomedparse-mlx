"""Native FocalNet adaptation. Copyright (c) 2022 Microsoft, MIT; see THIRD_PARTY_NOTICES.md."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from biomedparse_mlx.depthwise import depthwise_conv
from biomedparse_mlx.mlx_types import compile_array_function, gelu
from biomedparse_mlx.ontology import ModelConfig


class Mlp(nn.Module):
    """Apply a four-times-width GELU feed-forward projection to the last axis."""

    def __init__(self, width: int) -> None:
        """Initialize the expansion and projection layers for the feature width."""
        super().__init__()
        self.fc1 = nn.Linear(width, 4 * width)
        self.fc2 = nn.Linear(4 * width, width)

    def __call__(self, values: mx.array) -> mx.array:
        """Project the last axis through GELU and back to the input width."""
        return self.fc2(gelu(self.fc1(values)))


class FocalModulation(nn.Module):
    """Combine gated local depthwise contexts and global context on NHWC features.

    The local filters have increasing kernel sizes. Global context is averaged
    over height and then width, retaining the source reduction order.
    """

    def __init__(self, width: int, levels: int, window: int) -> None:
        """Build per-level depthwise filters, gates and context projections."""
        super().__init__()
        self.f = nn.Linear(width, 2 * width + levels + 1)
        self.h = nn.Conv2d(width, width, 1)
        self.proj = nn.Linear(width, width)
        self.focal_layers = [
            [
                nn.Conv2d(
                    width,
                    width,
                    window + 2 * i,
                    padding=(window + 2 * i) // 2,
                    groups=width,
                    bias=False,
                )
            ]
            for i in range(levels)
        ]
        self._width = width

    def __call__(self, values: mx.array) -> mx.array:
        """Gate local and global NHWC contexts to modulate each query."""
        projected = self.f(values)
        query = projected[..., : self._width]
        context = projected[..., self._width : 2 * self._width]
        gates = projected[..., 2 * self._width :]
        total = mx.zeros_like(context)
        for i, layer in enumerate(self.focal_layers):
            conv = layer[0]
            # MLX's depthwise kernel stops at 7x7; larger filters take the exact replacement.
            filtered = (
                depthwise_conv(context, conv.weight) if conv.weight.shape[1] > 7 else conv(context)
            )
            context = gelu(filtered)
            total = total + context * gates[..., i : i + 1]
        # Preserve the source's two reductions, rather than changing summation order.
        global_context = gelu(context.mean(axis=1, keepdims=True).mean(axis=2, keepdims=True))
        total = total + global_context * gates[..., -1:]
        return self.proj(query * self.h(total / gates.shape[-1]))


class FocalBlock(nn.Module):
    """Apply modulation and MLP residuals with post-operation normalization and learned scales."""

    def __init__(self, width: int, config: ModelConfig) -> None:
        """Build the modulation and feed-forward residual branches."""
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.modulation = FocalModulation(width, config.focal_levels, config.focal_window)
        self.mlp = Mlp(width)
        self.gamma_1 = mx.full((width,), 1e-4)
        self.gamma_2 = mx.full((width,), 1e-4)
        # Fusing the block's elementwise work is exact; the state stays a live input.
        self._compiled = compile_array_function(self._apply, inputs=self.state)

    def __call__(self, values: mx.array) -> mx.array:
        """Apply scaled, normalized modulation and MLP residuals."""
        return self._compiled(values)

    def _apply(self, values: mx.array) -> mx.array:
        values = values + self.gamma_1 * self.norm1(self.modulation(values))
        return values + self.gamma_2 * self.norm2(self.mlp(values))


class PatchEmbed(nn.Module):
    """Pad the bottom and right, then project NHWC input with stride four or two."""

    def __init__(self, inputs: int, outputs: int, *, stem: bool) -> None:
        """Choose the stem or downsampling convolution and output normalization."""
        super().__init__()
        self.proj = nn.Conv2d(
            inputs, outputs, 7 if stem else 3, stride=4 if stem else 2, padding=2 if stem else 1
        )
        self.norm = nn.LayerNorm(outputs)
        self._patch = 4 if stem else 2

    def __call__(self, values: mx.array) -> mx.array:
        """Pad the spatial axes and project NHWC features to patches."""
        height, width = values.shape[1:3]
        values = mx.pad(
            values, [(0, 0), (0, -height % self._patch), (0, -width % self._patch), (0, 0)]
        )
        return self.norm(self.proj(values))


class FocalStage(nn.Module):
    """Run focal blocks; the parent applies the optional downsample between stages."""

    def __init__(self, width: int, depth: int, config: ModelConfig, *, final: bool) -> None:
        """Build the requested focal blocks and optional downsampling layer."""
        super().__init__()
        self.blocks = [FocalBlock(width, config) for _ in range(depth)]
        self.downsample = None if final else PatchEmbed(width, 2 * width, stem=False)

    def __call__(self, values: mx.array) -> mx.array:
        """Apply the focal blocks without applying the parent-owned downsample."""
        for block in self.blocks:
            values = block(values)
        return values


class FocalNet(nn.Module):
    """Produce four normalized NHWC feature maps at progressively coarser scales."""

    def __init__(self, config: ModelConfig) -> None:
        """Build the patch stem, four focal stages and output normalizations."""
        super().__init__()
        self.patch_embed = PatchEmbed(3, config.embed_dim, stem=True)
        self.layers = [
            FocalStage(ch, depth, config, final=i == 3)
            for i, (ch, depth) in enumerate(zip(config.channels, config.depths, strict=True))
        ]
        for i, width in enumerate(config.channels):
            setattr(self, f"norm{i}", nn.LayerNorm(width))

    def __call__(self, images: mx.array) -> tuple[mx.array, ...]:
        """Return the four stage outputs, each normalized, finest first.

        ``images`` is NHWC with 3 channels; the stages have strides 4, 8, 16 and 32
        (inputs not divisible by the patch size are zero-padded at the bottom/right).
        """
        values = self.patch_embed(images)
        features: list[mx.array] = []
        for i, stage in enumerate(self.layers):
            values = stage(values)
            features.append(getattr(self, f"norm{i}")(values))
            if stage.downsample is not None:
                values = stage.downsample(values)
        return tuple(features)
