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


def nearest_indices(source: int, target: int) -> NDArray[np.intp]:
    """Source indices of Torch's legacy ``nearest`` resize along one axis.

    ATen copies equal sizes, halves exact 2x upsampling with a shift, and
    otherwise floors ``index * scale`` in float32 with ``scale = source / target``
    also in float32. Float64 arithmetic picks the neighboring source index for
    some sizes (for example 512 to 218), so the float32 steps are kept.
    """
    if target == source:
        return np.arange(target)
    if target == 2 * source:
        return np.arange(target) >> 1
    scale = np.float32(source) / np.float32(target)
    index = np.floor(np.arange(target, dtype=np.float32) * scale).astype(np.intp)
    return np.minimum(index, source - 1)


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
        index = nearest_indices(source, target).astype(np.int32)
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


def _uint8_bicubic_weights(
    source: int, target: int
) -> tuple[NDArray[np.intp], NDArray[np.int32], int]:
    """Fixed-point taps of Torch's CPU uint8 bicubic resize for one axis.

    Keys a=-0.75 at half-pixel positions in float64, with out-of-range taps
    clamped to the border and merged into a four-wide window. Weights are rounded
    to the largest fixed-point precision whose biggest weight still fits int16,
    as ATen's separable uint8 kernel does. ATen floors the position with
    ``floorf`` (after rounding it to float32) and clamps the fraction to [0, 1],
    which matters where a position lies just below an integer. Returns source
    indices ``(target, 4)``, integer weights ``(target, 4)`` and that precision
    in bits.
    """
    position = (np.arange(target, dtype=np.float64) + 0.5) * (source / target) - 0.5
    floor = np.minimum(np.floor(position.astype(np.float32)).astype(np.intp), source - 1)
    fraction = np.clip(position - floor, 0.0, 1.0)
    taps = floor[:, None] + np.arange(-1, 3)
    x = np.abs(fraction[:, None] - np.arange(-1, 3))
    a = -0.75
    weight = np.where(
        x < 1,
        ((a + 2) * x - (a + 3)) * x * x + 1,
        np.where(x < 2, ((a * x - 5 * a) * x + 8 * a) * x - 4 * a, 0.0),
    )
    clamped = np.clip(taps, 0, source - 1)
    start = clamped[:, :1]
    merged = np.zeros_like(weight)
    # ATen tracks the largest weight while it merges border taps, so partial sums
    # count toward the precision choice as well as the final weights.
    largest = 0.0
    rows = np.arange(target)
    for tap in range(4):
        slot = clamped[:, tap] - start[:, 0]
        merged[rows, slot] += weight[:, tap]
        largest = max(largest, float(merged[rows, slot].max()))
    precision = 0
    while precision < 22 and int(0.5 + largest * (1 << (precision + 1))) < (1 << 15):
        precision += 1
    fixed = np.trunc(merged * (1 << precision) + np.copysign(0.5, merged)).astype(np.int32)
    index = np.minimum(start + np.arange(4), source - 1)
    return index, fixed, precision


def resize_uint8_bicubic(values: NDArray[np.uint8], size: tuple[int, int]) -> NDArray[np.uint8]:
    """Resize a uint8 NHWC batch as ``F.interpolate(..., mode="bicubic")`` does on CPU uint8.

    Upstream BiomedParse resizes its uint8 volumes this way before casting to
    int. Unlike the float kernel, ATen resizes width then height in separate
    fixed-point passes, rounding and clamping to 0..255 after each one, so
    results differ from a rounded float bicubic resize near strong edges. Host
    preprocessing, before the model runs on Metal.

    Raises:
        ValueError: ``values`` is not a 4D uint8 array or ``size`` is not positive.
    """
    if values.ndim != 4 or values.dtype != np.uint8 or min(size) < 1:
        raise ValueError("resize_uint8_bicubic requires uint8 NHWC input and a positive size")
    result = values
    for axis, target in ((2, size[1]), (1, size[0])):
        if result.shape[axis] == target:
            continue
        index, weight, precision = _uint8_bicubic_weights(result.shape[axis], target)
        moved = np.moveaxis(result, axis, -1).astype(np.int32)
        total = (moved[..., index] * weight).sum(axis=-1)
        rounded = (total + (1 << (precision - 1))) >> precision
        result = np.moveaxis(np.clip(rounded, 0, 255).astype(np.uint8), -1, axis)
    return result


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
