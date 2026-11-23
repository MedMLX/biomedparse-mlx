"""Narrow adapters for MLX 0.32's incompletely annotated Python entrypoints.

These signatures follow mlx.nn.layers.activations and mlx.utils.tree_flatten.
The runtime functions and parameter tree remain unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, cast

import mlx.core as mx
import mlx.nn as nn
import mlx.utils as utils

from biomedparse_mlx.dependency_types import callable_member

if TYPE_CHECKING:
    from medmlx_core.typing import ArrayEvaluator, ArrayFactory

type ParameterTree = mx.array | dict[str, ParameterTree] | list[ParameterTree]


class ArrayFunction(Protocol):
    """Typed signature for a one-array MLX function."""

    def __call__(self, values: mx.array, /) -> mx.array:
        """Map one MLX array to another."""
        ...


class ArrayCompiler(Protocol):
    """Typed signature for ``mx.compile`` over a one-array function and captured state."""

    def __call__(
        self, fun: ArrayFunction, inputs: ParameterTree | nn.Module | None = None
    ) -> ArrayFunction:
        """Compile ``fun``, reading ``inputs`` as live state on every call.

        ``inputs`` may be a module's ``state``: an ``nn.Module`` is itself the
        dict tree that ``mx.compile`` captures.
        """
        ...


# MLX accepts NumPy's DLPack arrays, and eval accepts nested array sequences.
from_numpy = cast("ArrayFactory", mx.array)
evaluate = cast("ArrayEvaluator", callable_member(mx, "eval"))
compile_array_function = cast(ArrayCompiler, callable_member(mx, "compile"))
gelu = nn.gelu
relu = nn.relu
flatten_parameters = cast(
    Callable[[dict[str, ParameterTree]], list[tuple[str, mx.array]]],
    callable_member(utils, "tree_flatten"),
)
