"""Torch-compatible half-pixel resampling, executed as batched MLX operations."""

from __future__ import annotations

import math
from enum import StrEnum

import mlx.core as mx
import numpy as np
from numpy.typing import NDArray

from biomedparse_mlx.mlx_types import from_numpy


class ResizeMode(StrEnum):
    """Interpolation modes for :func:`resize`, named as in Torch's ``F.interpolate``."""

    NEAREST = "nearest"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"


def _cubic(distance: NDArray[np.float64 | np.float32], coefficient: float) -> NDArray[np.float32]:
    """Keys cubic convolution kernel with parameter ``coefficient`` (the usual ``a``)."""
    x = np.abs(distance)
    inner = ((coefficient + 2) * x - (coefficient + 3)) * x * x + 1
    outer = ((coefficient * x - 5 * coefficient) * x + 8 * coefficient) * x - 4 * coefficient
    return np.where(x < 1, inner, np.where(x < 2, outer, 0)).astype(np.float32)


def _weights(
    source: int,
    target: int,
    mode: ResizeMode,
    antialias: bool,
) -> tuple[mx.array, mx.array]:
    """Compute source indices and interpolation weights for one resize axis."""
    scale = np.float32(source / target)
    center = (np.arange(target, dtype=np.float32) + np.float32(0.5)) * scale
    if mode is ResizeMode.NEAREST:
        index = np.floor(np.arange(target, dtype=np.float32) * scale).astype(np.int32)
        return from_numpy(index[:, None]), mx.ones((target, 1))
    if mode is ResizeMode.BILINEAR:
        position = np.maximum(center - np.float32(0.5), 0)
        left = np.floor(position).astype(np.int32)
        fraction = position - left
        index = left[:, None] + np.arange(2, dtype=np.int32)
        weight = np.stack((1 - fraction, fraction), axis=-1).astype(np.float32)
    elif antialias:
        # Antialiased bicubic widens the kernel by the downscale factor and uses
        # Keys a=-0.5, as Torch's antialias path does.
        support = 2 * max(float(scale), 1)
        left = np.maximum((center - support + 0.5).astype(np.int32), 0)
        right = np.minimum((center + support + 0.5).astype(np.int32), source)
        index = left[:, None] + np.arange(2 * math.ceil(support) + 1, dtype=np.int32)
        weight = _cubic((index - center[:, None] + 0.5) / max(float(scale), 1), -0.5)
        weight = np.where(index < right[:, None], weight, 0)
        weight = weight / weight.sum(axis=-1, keepdims=True)
    else:
        # Plain bicubic uses Keys a=-0.75, Torch's default for F.interpolate.
        position = center - np.float32(0.5)
        index = np.floor(position).astype(np.int32)[:, None] + np.arange(-1, 3, dtype=np.int32)
        weight = _cubic(position[:, None] - index, -0.75)
    return from_numpy(np.clip(index, 0, source - 1)), from_numpy(weight.astype(np.float32))


def resize(
    values: mx.array,
    size: tuple[int, int],
    mode: ResizeMode,
    *,
    antialias: bool = False,
) -> mx.array:
    """Resize an NHWC batch to ``size = (height, width)`` with Torch-compatible sampling.

    Half-pixel sampling positions; ``NEAREST`` is Torch's legacy ``nearest`` (floor
    of the scaled index). Host work is limited to coordinate coefficients.
    ``antialias`` is implemented only for bicubic.

    Raises:
        ValueError: ``values`` is not 4D, ``size`` is not positive, or ``antialias``
            is requested for another mode.
    """
    if values.ndim != 4 or min(size) < 1:
        raise ValueError("resize requires NHWC input and positive output dimensions")
    if antialias and mode is not ResizeMode.BICUBIC:
        raise ValueError("Only bicubic antialias is implemented")
    # Torch's antialias kernel applies width before height.
    for axis, target in ((2, size[1]), (1, size[0])):
        if values.shape[axis] == target:
            continue
        index, weights = _weights(values.shape[axis], target, mode, antialias)
        samples = mx.take(values, index, axis=axis)
        shape = [1] * samples.ndim
        shape[axis : axis + 2] = list(weights.shape)
        values = (samples * mx.reshape(weights, shape)).sum(axis=axis + 1)
    return values
