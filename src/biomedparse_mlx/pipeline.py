"""Shared headless action layer for array inference and MedMLX/CLI callers."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from biomedparse_mlx.ontology import (
    Backend,
    ModelConfig,
    Prompt,
    Segmentation,
    Volume,
    model_config,
)
from biomedparse_mlx.volume import neighboring_batch, restore_labels, volume_geometry

if TYPE_CHECKING:
    import mlx.core as mx


def postprocess(
    masks: mx.array,
    existence: mx.array,
    labels: tuple[int, ...],
    threshold: float,
) -> mx.array:
    """Batched slice NMS and background-aware merging, following upstream's threshold rules.

    Mask logits have (prompts, batch, height, width) axes and existence logits
    have (prompts, batch) axes. Return uint16 labels shaped (batch, height,
    width), with zero for background and foreground IDs taken from ``labels``.

    With one prompt, a slice keeps the mask when sigmoid(existence) is strictly
    above ``threshold``. With several, prompts are visited in descending existence
    score; one is kept when its score is at least ``threshold``, its mask (sigmoid
    above 0.5) is non-empty, and its IoU with every previously kept prompt in that
    order is at most 0.5. Each pixel then takes the kept prompt with the highest mask
    probability, or background unless that probability exceeds 0.5.
    """
    import mlx.core as mx

    probabilities = mx.sigmoid(masks)
    scores = mx.sigmoid(existence)
    count, batch, height, width = masks.shape
    if count == 1:
        probabilities = probabilities * (scores > threshold)[..., None, None]
    else:
        binary = mx.reshape(mx.transpose(probabilities > 0.5, (1, 0, 2, 3)), (batch, count, -1))
        areas = binary.sum(axis=-1)
        intersection = binary.astype(mx.float32) @ mx.transpose(
            binary.astype(mx.float32), (0, 2, 1)
        )
        union = areas[:, :, None] + areas[:, None, :] - intersection
        iou = intersection / mx.maximum(union, 1)
        order = mx.argsort(-scores.T, axis=1)
        keep = mx.zeros((batch, count), dtype=mx.bool_)
        suppressed = mx.zeros((batch, count), dtype=mx.bool_)
        all_ids = mx.arange(count)[None, :]
        for rank in range(count):
            candidate = order[:, rank : rank + 1]
            selected_score = mx.take_along_axis(scores.T, candidate, axis=1)
            selected_area = mx.take_along_axis(areas, candidate, axis=1)
            selected_suppressed = mx.take_along_axis(suppressed, candidate, axis=1)
            active = (selected_score >= threshold) & (selected_area > 0) & ~selected_suppressed
            keep = keep | ((all_ids == candidate) & active)
            overlaps = mx.take_along_axis(iou, candidate[:, :, None], axis=1)[:, 0]
            suppressed = suppressed | ((overlaps > 0.5) & active)
        probabilities = probabilities * keep.T[..., None, None]
    with_background = mx.concatenate([mx.full((1, batch, height, width), 0.5), probabilities])
    indices = mx.argmax(with_background, axis=0)
    return mx.array((0, *labels), dtype=mx.uint16)[indices]


def _resize_prompt_logits(raw: mx.array, batch: int, count: int, image_size: int) -> mx.array:
    """Reshape a flat ``(batch*count, h, w)`` prediction into ``(count, batch, size, size)``.

    The flat axis is image-major, prompt-minor. The result has the layout
    ``postprocess`` expects, with bicubic antialiased resizing to the model's square
    size; shared by the mask and edge channels."""
    import mlx.core as mx

    from biomedparse_mlx.resize import ResizeMode, resize

    h, w = raw.shape[-2:]
    reshaped = mx.transpose(mx.reshape(raw, (batch, count, h, w)), (0, 2, 3, 1))
    resized = resize(reshaped, (image_size, image_size), ResizeMode.BICUBIC, antialias=True)
    return mx.transpose(resized, (3, 0, 1, 2))


def instance_labels(labels: NDArray[np.uint16], edges: NDArray[np.uint16]) -> NDArray[np.uint16]:
    """Upstream's example-notebook edge-refined instance split (``inference_example_3D.ipynb``).

    ``labels`` and ``edges`` are same-shaped uint16 volumes on the source grid;
    the result is a uint16 volume with instance IDs (not the prompt labels).

    Zero the predicted edge pixels out of the semantic labels, split the remaining
    interior into 3D face-connected components (``skimage.measure.label`` only connects
    equal values), re-expand by one voxel to recover the eroded boundary, then
    mask back to the semantic foreground. Expansion is not constrained by class
    and can assign an instance ID across a semantic-class boundary.

    Raises:
        ValueError: More than 65535 interior components would overflow uint16.
    """
    from biomedparse_mlx._skimage import measure, segmentation

    interior = labels * (edges == 0)
    components = measure.label(interior, connectivity=1)
    if components.max(initial=0) > np.iinfo(np.uint16).max:
        raise ValueError("Instance labels support at most 65535 components")
    expanded = segmentation.expand_labels(components, distance=1)
    return (expanded * (labels > 0)).astype(np.uint16)


def segment_volume(
    volume: Volume,
    prompts: tuple[Prompt, ...],
    weights_path: Path,
    tokenizer_path: Path,
    *,
    slice_batch_size: int = 2,
    prompt_batch_size: int | None = None,
    existence_threshold: float = 0.5,
    seed: int = 0,
    progress: Callable[[int, int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
    config: ModelConfig | None = None,
    instance_segmentation: bool = False,
) -> Segmentation:
    """Run the native model in bounded slice batches, preserving caller array geometry.

    Args:
        volume: Modality-preprocessed intensities (0..255). The slicing axis comes
            from the array shape (``volume_geometry``).
        prompts: Text prompts with distinct integer labels; they become the IDs in
            the output label volume, with 0 for background.
        weights_path: Native ``.safetensors`` weight file.
        tokenizer_path: Local CLIP ``tokenizer.json``.
        slice_batch_size: Slices per model call (bounds memory).
        prompt_batch_size: Prompts per decoder group; None decodes all together.
            Image features are encoded once per slice batch and shared by all
            prompts. Grouping keeps one prediction head's sampling field at a
            time; output logits still scale with the total prompt count.
        existence_threshold: Cutoff in [0, 1] on a prompt's per-slice existence
            probability.
        seed: Non-negative seed for MLX's global generator, which drives the
            attention sampling; it does not align with Torch's generators.
        progress: Called as ``progress(slices_done, slices_total)`` after each batch.
        cancelled: Polled before each batch; returning True raises ``InterruptedError``.
        config: Model definition; defaults to the packaged one.
        instance_segmentation: Opt into upstream's example-notebook edge-refined
            instance split (``instance_labels``): the predicted edge channel is
            postprocessed the same way as the mask channel, then used to separate
            touching same-class instances before a 3D face-connected-component split.

    Returns:
        A ``Segmentation`` whose ``labels`` has the input array's shape (uint16) and
        whose ``existence_probabilities`` is float32 ``(prompts, slices)`` along
        the slicing axis. ``instances`` is set only when requested.

    Each square-padded slice batch is bicubically resized to the model size as
    upstream's ``process_input`` does with ``F.interpolate``. A uint8 volume (the
    upstream NPZ format) takes Torch's uint8 kernel, which rounds and clamps to
    0..255 after each axis (``resize_uint8_bicubic``). Any other dtype is resized
    in float32 and cast to int32 and back, as upstream's ``.int()`` does; that
    resize can overshoot 0..255 and the cast does not clip it.

    Raises:
        ValueError: Invalid batch size, threshold, seed or prompts, or instance
            segmentation without edge queries in the model.
        InterruptedError: ``cancelled`` returned True.
    """
    if type(slice_batch_size) is not int or slice_batch_size < 1:
        raise ValueError("slice_batch_size must be a positive integer")
    if prompt_batch_size is not None and (
        type(prompt_batch_size) is not int or prompt_batch_size < 1
    ):
        raise ValueError("prompt_batch_size must be a positive integer or None")
    if not 0 <= existence_threshold <= 1 or type(seed) is not int or seed < 0:
        raise ValueError("Invalid existence threshold or random seed")
    if not prompts or len({p.label for p in prompts}) != len(prompts):
        raise ValueError("At least one prompt with distinct labels is required")
    from medmlx_core.runtime import import_mlx

    from biomedparse_mlx.mlx_types import evaluate, from_numpy
    from biomedparse_mlx.model import load_model
    from biomedparse_mlx.resize import ResizeMode, resize, resize_uint8_bicubic
    from biomedparse_mlx.text import encode_prompts

    mx = import_mlx()
    config = model_config() if config is None else config
    if instance_segmentation and config.edge_queries < 1:
        raise ValueError("instance_segmentation requires a model with edge_queries >= 1")
    model = load_model(weights_path, config)
    features = encode_prompts(
        model.sem_seg_head.predictor.language_encoder, prompts, tokenizer_path, config
    )
    mx.random.seed(seed)
    geometry = volume_geometry(volume)
    depth = volume.data.shape[geometry.axis]
    labels = np.empty((depth, config.image_size, config.image_size), dtype=np.uint16)
    scores = np.empty((len(prompts), depth), dtype=np.float32)
    edges = np.empty_like(labels) if instance_segmentation else None
    label_ids = tuple(p.label for p in prompts)
    for start in range(0, depth, slice_batch_size):
        if cancelled is not None and cancelled():
            raise InterruptedError("BioMedParse inference cancelled")
        stop = min(start + slice_batch_size, depth)
        pixels = neighboring_batch(volume, geometry, start, stop)
        model_size = (config.image_size, config.image_size)
        if pixels.dtype == np.uint8:
            resized = resize_uint8_bicubic(pixels.astype(np.uint8, copy=False), model_size)
            images = from_numpy(resized, dtype=mx.float32)
        else:
            images = resize(from_numpy(pixels, dtype=mx.float32), model_size, ResizeMode.BICUBIC)
            images = images.astype(mx.int32).astype(mx.float32)
        prediction = (
            model(images, features)
            if prompt_batch_size is None
            else model(images, features, prompt_batch_size=prompt_batch_size)
        )
        count = len(prompts)
        batch = stop - start
        logits = _resize_prompt_logits(prediction.mask_logits, batch, count, config.image_size)
        existence = mx.reshape(prediction.existence_logits, (batch, count)).T
        merged = postprocess(logits, existence, label_ids, existence_threshold)
        probabilities = mx.sigmoid(existence)
        if edges is None:
            evaluate(merged, probabilities)
        else:
            edge_logits = _resize_prompt_logits(
                prediction.edge_logits, batch, count, config.image_size
            )
            merged_edges = postprocess(edge_logits, existence, label_ids, existence_threshold)
            evaluate(merged, probabilities, merged_edges)
            edges[start:stop] = np.array(merged_edges)
        labels[start:stop] = np.array(merged)
        scores[:, start:stop] = np.array(probabilities)
        if progress is not None:
            progress(stop, depth)
    restored_labels = restore_labels(labels, geometry)
    instances = (
        None if edges is None else instance_labels(restored_labels, restore_labels(edges, geometry))
    )
    return Segmentation(restored_labels, scores, prompts, Backend.METAL, instances)
