"""Native numerical boundaries with an independent installed Torch oracle."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, cast

import numpy as np
import pytest

from biomedparse_mlx.dependency_types import callable_member
from biomedparse_mlx.ontology import ModelConfig
from tests.torch_types import from_numpy, interpolate, numpy_float

pytestmark = [pytest.mark.metal, pytest.mark.usefixtures("metal_runtime")]


@pytest.mark.parametrize(
    "mode,aa", [("nearest", False), ("bilinear", False), ("bicubic", False), ("bicubic", True)]
)
@pytest.mark.parametrize(
    "shape,size", [((2, 7, 9, 3), (13, 11)), ((1, 9, 7, 1), (4, 3)), ((1, 1, 3, 2), (5, 7))]
)
def test_resize_matches_torch(
    mode: str, aa: bool, shape: tuple[int, ...], size: tuple[int, int]
) -> None:
    """Compare native NHWC resize modes with the installed Torch oracle."""
    pytest.importorskip("torch")
    import mlx.core as mx

    from biomedparse_mlx.resize import ResizeMode, resize

    values = np.random.default_rng(7).normal(size=shape).astype(np.float32)
    kwargs = {} if mode == "nearest" else {"align_corners": False, "antialias": aa}
    expected = interpolate(from_numpy(values.transpose(0, 3, 1, 2)), size=size, mode=mode, **kwargs)
    actual = resize(mx.array(values), size, ResizeMode(mode), antialias=aa)
    np.testing.assert_allclose(
        np.array(actual, dtype=np.float32),
        numpy_float(expected).transpose(0, 2, 3, 1),
        atol=5e-6,
        rtol=5e-5,
    )


def test_strict_loader_rejects_other_formats_and_missing_tensors(
    tiny_config: ModelConfig, tmp_path: Path
) -> None:
    """Load saved safetensors strictly; reject NPZ weights and a missing tensor."""
    import mlx.core as mx

    from biomedparse_mlx.mlx_types import flatten_parameters
    from biomedparse_mlx.model import build_model, load_model

    model = build_model(tiny_config)
    weights = tmp_path / "tiny.safetensors"
    model.save_weights(str(weights))
    load_model(weights, tiny_config)
    legacy = tmp_path / "tiny.npz"
    model.save_weights(str(legacy))
    with pytest.raises(RuntimeError):
        load_model(legacy, tiny_config)
    parameters = dict(flatten_parameters(model.parameters()))
    parameters.pop("pixel_mean")

    class Writer(Protocol):
        def __call__(self, path: str, arrays: dict[str, mx.array]) -> None:
            """Write a named MLX tensor mapping as safetensors."""
            ...

    cast(Writer, callable_member(mx, "save_safetensors"))(str(weights), parameters)
    with pytest.raises(ValueError):
        load_model(weights, tiny_config)


@pytest.mark.parametrize("shape", [(1, 13, 11, 12), (2, 16, 9, 20)])
def test_large_depthwise_conv_matches_mlx_bits(shape: tuple[int, int, int, int]) -> None:
    """Match MLX's 9x9 depthwise conv bit for bit, with partial tiles and non-finite values."""
    import mlx.core as mx

    from biomedparse_mlx.depthwise import depthwise_conv

    rng = np.random.default_rng(3)
    values = rng.normal(size=shape).astype(np.float32)
    special = np.array([np.nan, np.inf, -np.inf, 1e-40, -0.0, 3e38], np.float32)
    values.reshape(-1)[rng.choice(values.size, special.size, replace=False)] = special
    weight = mx.array(rng.normal(size=(shape[-1], 9, 9, 1)).astype(np.float32))
    expected = mx.conv2d(mx.array(values), weight, padding=4, groups=shape[-1])
    actual = depthwise_conv(mx.array(values), weight)
    np.testing.assert_array_equal(
        np.array(actual).view(np.uint32), np.array(expected).view(np.uint32)
    )
