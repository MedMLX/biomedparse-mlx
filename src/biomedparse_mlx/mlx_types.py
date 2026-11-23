"""Typed entry points to MLX functions that the MLX stubs annotate incompletely.

The signatures come from ``medmlx_core.typing``; the runtime functions and the
parameter tree are unchanged.
"""

from __future__ import annotations

from typing import cast

import mlx.core as mx
import mlx.nn as nn
import mlx.utils as utils
from medmlx_core.typing import MlxRuntime, NeuralRuntime, UtilsRuntime

_mx = cast(MlxRuntime, mx)
_nn = cast(NeuralRuntime, nn)

from_numpy = _mx.array
evaluate = _mx.eval
compile_array_function = _mx.compile
gelu = _nn.gelu
relu = _nn.relu
flatten_parameters = cast(UtilsRuntime, utils).tree_flatten
