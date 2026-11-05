"""Repository-used MLX 0.32.4 neural-network APIs, from the installed source.

Covers mlx/nn/layers/base.py, activations.py, convolution.py, embedding.py,
linear.py and normalization.py. Only the members this repository uses are
declared. Module is a partial interface: the runtime class subclasses dict and
its state property returns the module itself.
"""

from typing import Self

from mlx.core import array

type ParameterTree = array | dict[str, ParameterTree] | list[ParameterTree]
type Shape = int | tuple[int, ...]

class Module:
    def __init__(self) -> None: ...
    @property
    def state(self) -> Self: ...
    def parameters(self) -> dict[str, ParameterTree]: ...
    def load_weights(
        self, file_or_weights: str | list[tuple[str, array]], strict: bool = True
    ) -> Self: ...
    def save_weights(self, file: str) -> None: ...
    def eval(self) -> Self: ...

class Conv2d(Module):
    weight: array
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Shape,
        stride: Shape = 1,
        padding: Shape = 0,
        dilation: Shape = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None: ...
    def __call__(self, x: array) -> array: ...

class Embedding(Module):
    weight: array
    def __init__(self, num_embeddings: int, dims: int) -> None: ...
    def __call__(self, x: array) -> array: ...

class Linear(Module):
    def __init__(self, input_dims: int, output_dims: int, bias: bool = True) -> None: ...
    def __call__(self, x: array) -> array: ...

class LayerNorm(Module):
    def __init__(
        self, dims: int, eps: float = 1e-5, affine: bool = True, bias: bool = True
    ) -> None: ...
    def __call__(self, x: array) -> array: ...

class GroupNorm(Module):
    def __init__(
        self,
        num_groups: int,
        dims: int,
        eps: float = 1e-5,
        affine: bool = True,
        pytorch_compatible: bool = False,
    ) -> None: ...
    def __call__(self, x: array) -> array: ...

def relu(x: array) -> array: ...
def gelu(x: array) -> array: ...
