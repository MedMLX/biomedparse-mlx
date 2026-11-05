"""Depthwise convolution for FocalNet filters too large for MLX's depthwise kernel.

MLX 0.32.4 runs depthwise filters up to 7x7 on a dedicated kernel. Larger ones
fall back to its grouped implicit GEMM with one channel per group: each
threadgroup gathers one channel's taps from NHWC memory and fills only the
first column of every 8x8 tile. This kernel issues the same 8x8 simdgroup
multiply-accumulates over the same zero-padded 8-tap chunks, in the same
order, so it returns bit-identical values. It is faster because each
threadgroup stages one input patch for 8 channels and 8 output rows in
threadgroup memory before the multiply-accumulates.
"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, cast

import mlx.core as mx

if TYPE_CHECKING:
    from medmlx_core.typing import MetalKernel

# One threadgroup covers 8 output rows x 8 columns x 8 channels; one simdgroup per channel.
_ROWS = 8
_COLUMNS = 8
_CHANNELS = 8

# MLX's implicit GEMM reads taps in (row, column) order, 16 per K step, as two
# 8-tap MMA chunks; taps past K*K are zero. A holds 8 output pixels by 8 taps
# and B holds one channel's weights in column 0, zero elsewhere.
_SOURCE = f"""
constexpr int K = KERNEL, KK = KERNEL * KERNEL, pad = KERNEL / 2;
constexpr int chunks = ((KK + 15) / 16) * 2;
constexpr int TY = {_ROWS}, TX = {_COLUMNS}, TC = {_CHANNELS};
constexpr int PH = TY + K - 1, PW = TX + K - 1, PC = TC + 1;
threadgroup float patch[PH * PW * PC];
const int N = int(sizes[0]), H = int(sizes[1]), W = int(sizes[2]), C = int(sizes[3]);
const int YB = (H + TY - 1) / TY, XB = (W + TX - 1) / TX, CB = (C + TC - 1) / TC;
int group = int(threadgroup_position_in_grid.x);
const int cb = group % CB; group /= CB;
const int xb = group % XB; group /= XB;
const int yb = group % YB;
const int n = group / YB;
const int y0 = yb * TY, x0 = xb * TX, c0 = cb * TC;
for (int i = int(thread_position_in_threadgroup.x); i < PH * PW * TC; i += TC * 32) {{
  const int local = i % TC, pixel = i / TC, px = pixel % PW, py = pixel / PW;
  const int iy = y0 + py - pad, ix = x0 + px - pad, c = c0 + local;
  float value = 0.0f;
  if (iy >= 0 && iy < H && ix >= 0 && ix < W && c < C) {{
    value = values[((n * H + iy) * W + ix) * C + c];
  }}
  patch[pixel * PC + local] = value;
}}
threadgroup_barrier(mem_flags::mem_threadgroup);
const int local = int(simdgroup_index_in_threadgroup), c = c0 + local;
if (c >= C) {{
  return;
}}
const int lane = int(thread_index_in_simdgroup);
const int quad = lane / 4;
const int fm = (quad & 4) + ((lane / 2) % 4), fn = (quad & 2) * 2 + (lane % 2) * 2;
for (int r = 0; r < TY && y0 + r < H; ++r) {{
  simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  _Pragma("unroll") for (int chunk = 0; chunk < chunks; ++chunk) {{
    simdgroup_float8x8 a, b;
    const int tap = chunk * 8 + fm;
    for (int e = 0; e < 2; ++e) {{
      const int t = chunk * 8 + fn + e;
      a.thread_elements()[e] = t < KK ? patch[((r + t / K) * PW + fm + t % K) * PC + local] : 0.0f;
      b.thread_elements()[e] = fn + e == 0 && tap < KK ? weight[c * KK + tap] : 0.0f;
    }}
    simdgroup_multiply_accumulate(acc, a, b, acc);
  }}
  if (fn == 0 && x0 + fm < W) {{
    out[((n * H + y0 + r) * W + x0 + fm) * C + c] = acc.thread_elements()[0];
  }}
}}
"""


@cache
def _kernel() -> MetalKernel:
    return cast(
        "MetalKernel",
        mx.fast.metal_kernel(
            name="biomedparse_depthwise_implicit_gemm",
            input_names=["values", "weight", "sizes"],
            output_names=["out"],
            source=_SOURCE,
        ),
    )


def depthwise_conv(values: mx.array, weight: mx.array) -> mx.array:
    """Apply a same-padded, stride-1, bias-free depthwise convolution.

    Args:
        values: Float32 NHWC features.
        weight: Float32 ``(C, K, K, 1)`` filter with odd ``K``, as ``nn.Conv2d`` stores it.

    Returns:
        The NHWC result, bit-identical to ``mx.conv2d(values, weight,
        padding=K // 2, groups=C)`` for filters larger than 7x7 on MLX 0.32.4.

    Raises:
        ValueError: Shapes or dtypes are outside that contract, or the element
            count exceeds the kernel's 32-bit indexing.
    """
    batch, height, width, channels = values.shape
    size = weight.shape[1]
    if (
        values.dtype != mx.float32
        or weight.dtype != mx.float32
        or weight.shape != (channels, size, size, 1)
        or size % 2 == 0
        or values.size >= 2**31
    ):
        raise ValueError(
            "Depthwise convolution requires float32 NHWC values and odd square filters"
        )
    groups = batch * -(-height // _ROWS) * -(-width // _COLUMNS) * -(-channels // _CHANNELS)
    sizes = mx.array([batch, height, width, channels], dtype=mx.uint32)
    return _kernel()(
        inputs=[values, weight, sizes],
        template=[("KERNEL", size)],
        grid=(groups * _CHANNELS * 32, 1, 1),
        threadgroup=(_CHANNELS * 32, 1, 1),
        output_shapes=[values.shape],
        output_dtypes=[mx.float32],
    )[0]
