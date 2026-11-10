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


@pytest.mark.parametrize("source,target", [(218, 512), (669, 512), (513, 512), (321, 37)])
def test_uint8_bicubic_resize_matches_torch_uint8_kernel(source: int, target: int) -> None:
    """Upstream resizes uint8 volumes with Torch's uint8 kernel, which clamps per axis."""
    pytest.importorskip("torch")
    from biomedparse_mlx.resize import resize_uint8_bicubic

    rng = np.random.default_rng(source)
    # Smooth anatomy-like background with hard 0/255 edges, where clamping matters.
    yy, xx = np.mgrid[:source, :source] / source
    base = 120 + 80 * np.sin(6 * xx) * np.cos(4 * yy) + rng.normal(0, 12, (source, source))
    base[source // 4 : source // 2, source // 3 : source // 2] = 255
    base[:, : source // 8] = 0
    planes = np.clip(np.stack([base, base.T, base[::-1]]), 0, 255).astype(np.uint8)
    expected = np.asarray(
        interpolate(from_numpy(planes[None]), size=(target, target), mode="bicubic").numpy()
    )[0]
    actual = resize_uint8_bicubic(planes.transpose(1, 2, 0)[None], (target, target))[0]
    np.testing.assert_array_equal(actual.transpose(2, 0, 1), expected)


@pytest.mark.parametrize("plane", [(218, 182), (164, 100), (82, 82), (669, 551), (512, 400)])
def test_restored_labels_match_upstream_nearest_resize(plane: tuple[int, int]) -> None:
    """Labels return to the source grid as upstream's process_output resizes them."""
    pytest.importorskip("torch")
    from biomedparse_mlx.ontology import Volume
    from biomedparse_mlx.volume import restore_labels, volume_geometry

    geometry = volume_geometry(Volume(np.zeros((3, *plane), dtype=np.uint8)))
    labels = np.random.default_rng(plane[0]).integers(0, 5, (3, 512, 512)).astype(np.uint16)
    square = geometry.square_size
    resized = np.asarray(
        interpolate(
            from_numpy(labels[None].astype(np.float32)), size=(square, square), mode="nearest"
        ).numpy()
    )[0]
    (top, _), (left, _) = geometry.padding
    expected = resized[:, top : top + plane[0], left : left + plane[1]]
    np.testing.assert_array_equal(restore_labels(labels, geometry), expected)


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


def test_prompt_groups_keep_outputs_and_random_stream(tiny_config: ModelConfig) -> None:
    """Grouped decoding reads the ungrouped run's uniforms and leaves the same RNG state."""
    import mlx.core as mx

    from biomedparse_mlx.mlx_types import evaluate
    from biomedparse_mlx.mlx_types import from_numpy as to_mlx
    from biomedparse_mlx.model import build_model
    from biomedparse_mlx.ontology import AttentionReplay, Prompt, PromptFeatures

    model = build_model(tiny_config)
    rng = np.random.default_rng(5)
    images = to_mlx(rng.uniform(0, 255, (2, 32, 32, 3)).astype(np.float32))
    tokens = to_mlx(rng.normal(size=(5, 3, tiny_config.hidden_dim)).astype(np.float32))
    prompts = PromptFeatures(tuple(Prompt(i + 1, f"p{i}") for i in range(5)), tokens)
    encoded = model.encode_images(images)

    def decode(size: int | None, sampling: AttentionReplay | None = None) -> list[np.ndarray]:
        mx.random.seed(3)
        out = model.decode_prompts(encoded, prompts, sampling=sampling, prompt_batch_size=size)
        after = mx.random.uniform(shape=(4,))
        evaluate(out.mask_logits, out.existence_logits, out.edge_logits, after)
        fields = (out.mask_logits, out.existence_logits, out.edge_logits, after)
        return [np.array(value) for value in fields]

    blocked = tuple(
        to_mlx(rng.random((10, 4, 8, encoded.scales[i % 3].shape[1] ** 2)) < 0.8)
        for i in range(tiny_config.decoder_layers + 1)
    )
    for sampling in (None, AttentionReplay(blocked)):
        full = decode(None, sampling)
        # Two prompts per group leaves a partial last group.
        grouped = decode(2, sampling)
        np.testing.assert_array_equal(grouped[-1], full[-1])
        # Group sizes change FP32 matrix-product shapes, not the sampled positions.
        for actual, expected in zip(grouped[:-1], full[:-1], strict=True):
            np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
