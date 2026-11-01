"""Native NHWC layers adapted from BioMedParse v2.

Microsoft/Meta notices are in THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from biomedparse_mlx.mlx_types import relu


class Attention(nn.Module):
    """Multi-head attention on ``(batch, length, width)`` inputs.

    The Torch packed QKV layout is retained. A bool ``mask`` marks disallowed
    positions (True blocks attention), as in Torch; ``"causal"`` applies a causal
    mask.
    """

    def __init__(self, width: int, heads: int) -> None:
        """Allocate packed query, key and value projections and the output projection."""
        super().__init__()
        self.in_proj_weight = mx.zeros((3 * width, width))
        self.in_proj_bias = mx.zeros((3 * width,))
        self.out_proj = nn.Linear(width, width)
        self._heads = heads

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        """Apply scaled multi-head attention; true mask entries block positions."""
        width = query.shape[-1]
        projected = [
            x @ self.in_proj_weight[i * width : (i + 1) * width].T
            + self.in_proj_bias[i * width : (i + 1) * width]
            for i, x in enumerate((query, key, value))
        ]
        q, k, v = [
            mx.transpose(
                mx.reshape(x, (x.shape[0], x.shape[1], self._heads, width // self._heads)),
                (0, 2, 1, 3),
            )
            for x in projected
        ]
        if isinstance(mask, mx.array):
            mask = mx.where(mask, -float("inf"), 0.0)
        out = mx.fast.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=(width // self._heads) ** -0.5,
            mask=mask,
        )
        return self.out_proj(mx.reshape(mx.transpose(out, (0, 2, 1, 3)), query.shape))


class SelfAttention(nn.Module):
    """Post-norm residual self-attention; ``position`` is added to queries and keys only."""

    def __init__(self, width: int, heads: int) -> None:
        """Create self-attention and residual normalization at the given width."""
        super().__init__()
        self.self_attn = Attention(width, heads)
        self.norm = nn.LayerNorm(width)

    def __call__(self, values: mx.array, position: mx.array) -> mx.array:
        """Add positions to queries and keys and normalize the residual output."""
        query = values + position
        return self.norm(values + self.self_attn(query, query, values))


class CrossAttention(nn.Module):
    """Post-norm residual cross-attention from ``values`` to ``memory``.

    ``position`` is added to the memory keys (not its values) and ``query_position``,
    when given, to the queries. ``mask`` is a bool array where True blocks attention.
    """

    def __init__(self, width: int, heads: int) -> None:
        """Create cross-attention and residual normalization at the given width."""
        super().__init__()
        self.multihead_attn = Attention(width, heads)
        self.norm = nn.LayerNorm(width)

    def __call__(
        self,
        values: mx.array,
        memory: mx.array,
        position: mx.array,
        query_position: mx.array | None = None,
        mask: mx.array | None = None,
    ) -> mx.array:
        """Attend queries to positioned memory keys and normalize the residual."""
        query = values if query_position is None else values + query_position
        return self.norm(values + self.multihead_attn(query, memory + position, memory, mask))


class FFN(nn.Module):
    """Post-norm residual ReLU feed-forward block."""

    def __init__(self, width: int, hidden: int) -> None:
        """Create the hidden ReLU projection and residual normalization."""
        super().__init__()
        self.linear1 = nn.Linear(width, hidden)
        self.linear2 = nn.Linear(hidden, width)
        self.norm = nn.LayerNorm(width)

    def __call__(self, values: mx.array) -> mx.array:
        """Apply the feed-forward projection and post-normalize its residual."""
        return self.norm(values + self.linear2(relu(self.linear1(values))))


class EncoderLayer(nn.Module):
    """Post-norm transformer encoder layer; ``position`` is added to queries and keys only."""

    def __init__(self, width: int, heads: int, hidden: int) -> None:
        """Create post-normalized self-attention and feed-forward branches."""
        super().__init__()
        self.self_attn = Attention(width, heads)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.linear1 = nn.Linear(width, hidden)
        self.linear2 = nn.Linear(hidden, width)

    def __call__(self, values: mx.array, position: mx.array) -> mx.array:
        """Apply positioned self-attention followed by a feed-forward residual."""
        query = values + position
        values = self.norm1(values + self.self_attn(query, query, values))
        return self.norm2(values + self.linear2(relu(self.linear1(values))))


def position_encoding(height: int, width: int, channels: int) -> mx.array:
    """Unmasked DETR sine coordinates with upstream normalization epsilon.

    Returns ``(1, height * width, channels)``: the first half of the channels
    encodes the row, the second half the column, with positions flattened row-major.
    """
    count = channels // 2
    dim = 10000.0 ** (2 * (mx.arange(count) // 2).astype(mx.float32) / count)
    y = mx.arange(1, height + 1, dtype=mx.float32) / (height + 1e-6) * (2 * math.pi)
    x = mx.arange(1, width + 1, dtype=mx.float32) / (width + 1e-6) * (2 * math.pi)
    yy, xx = y[:, None] / dim, x[:, None] / dim
    py = mx.reshape(mx.stack([mx.sin(yy[:, 0::2]), mx.cos(yy[:, 1::2])], axis=-1), (height, count))
    px = mx.reshape(mx.stack([mx.sin(xx[:, 0::2]), mx.cos(xx[:, 1::2])], axis=-1), (width, count))
    return mx.reshape(
        mx.concatenate(
            [
                mx.broadcast_to(py[:, None, :], (height, width, count)),
                mx.broadcast_to(px[None, :, :], (height, width, count)),
            ],
            axis=-1,
        ),
        (1, height * width, channels),
    )
