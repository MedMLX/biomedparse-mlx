"""Upstream's 2D inference path (``inference_2d.py``): preprocessed 8-bit
grayscale/RGB images through the same v2 graph used by the 3D volume path.

v2's dedicated pretrained 2D weights are not yet released; upstream states the
released 3D checkpoint can initialize the architecture but does not establish
2D-domain accuracy. This module ports the 2D forward path (no neighbor-slice
packing; existence-gated per-prompt masks with no cross-prompt merge) so that
capability is available natively once a compatible checkpoint is used.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from biomedparse_mlx.ontology import Backend, ImageSegmentation, ModelConfig, Prompt, model_config
from biomedparse_mlx.preprocessing import Modality, normalize_intensity_slice
from biomedparse_mlx.volume import square_padding

if TYPE_CHECKING:
    import mlx.core as mx


def load_2d_image(path: Path) -> NDArray[np.uint8]:
    """Upstream's ``inference_2d.py`` ``main()``: L stays grayscale; RGB/RGBA/P become RGB.

    Returns a uint8 ``(H, W)`` array for grayscale input or ``(H, W, 3)`` for the
    rest (an alpha channel is dropped). The pixels must already be 8-bit
    display intensities; no windowing is applied.

    Raises:
        FileNotFoundError: ``path`` is not a file.
        ValueError: The image mode is not L, RGB, RGBA or P.
    """
    from PIL import Image

    if not path.is_file():
        raise FileNotFoundError(f"Input image not found: {path}")
    with Image.open(path) as handle:
        if handle.mode not in ("L", "RGB", "RGBA", "P"):
            raise ValueError("Preprocess medical intensities into an 8-bit L or RGB image first.")
        converted = handle.convert("L" if handle.mode == "L" else "RGB")
        return np.asarray(converted, dtype=np.uint8)


def load_dicom_slice(path: Path, *, modality: Modality, ct_site: str | None) -> NDArray[np.uint8]:
    """Upstream's ``inference_utils/processing_utils.py`` ``read_dicom``, stopped at
    0..255 (no resize/channel-stack; this package's own 2D resize follows instead).

    Pixels are converted with ``RescaleSlope``/``RescaleIntercept``, windowed with
    :func:`normalize_intensity_slice` and returned as a uint8 ``(H, W)`` array
    (fractions truncated).

    Raises:
        FileNotFoundError: ``path`` is not a file.
        ValueError: The file is not DICOM (no File Meta header or ``DICM``
            prefix), the dataset has no rescale slope or intercept, or the pixel
            data is not a single 2D slice that windowing accepts.
    """
    from pydicom.errors import InvalidDicomError

    from biomedparse_mlx._dicom import dicom

    if not path.is_file():
        raise FileNotFoundError(f"Input DICOM not found: {path}")
    try:
        dataset = dicom.dcmread(path.as_posix())
    except InvalidDicomError as error:
        raise ValueError(f"{path} is not a DICOM file: {error}") from error
    try:
        intensities = dataset.pixel_array * dataset.RescaleSlope + dataset.RescaleIntercept
    except AttributeError as error:
        raise ValueError(f"DICOM {path} has no rescale slope/intercept") from error
    normalized = normalize_intensity_slice(
        np.asarray(intensities), modality=modality, ct_site=ct_site
    )
    return np.clip(normalized, 0, 255).astype(np.uint8)


def _assemble_channels(image: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Return float32 ``(H, W, 3)`` channels from a uint8 ``(H, W)`` or ``(H, W, 3)`` image.

    Grayscale repeats to 3 identical channels; RGB passes through. Matches
    ``biomedparse_2D.py``'s ``image.expand(-1, 3, -1, -1)`` for single-channel input;
    resizing an identical-channel image commutes with the later channel expansion."""
    if image.ndim == 2:
        return np.stack([image] * 3, axis=-1).astype(np.float32)
    if image.ndim == 3 and image.shape[-1] == 3:
        return image.astype(np.float32)
    raise ValueError("image must be an 8-bit grayscale (H, W) or RGB (H, W, 3) array")


def existence_gated_probabilities(
    mask_logits: mx.array, existence_logits: mx.array, threshold: float
) -> tuple[mx.array, mx.array]:
    """Upstream's ``predict_image``: ``logits.sigmoid() * (existence > threshold)``,
    with no cross-prompt NMS or merge (unlike the multi-slice 3D ``postprocess``).

    ``mask_logits`` is ``(prompts, H, W)`` and ``existence_logits`` is ``(prompts,)``.
    Returns the gated mask probabilities ``(prompts, H, W)`` and the existence
    probabilities ``(prompts,)``; a prompt whose existence is at or below
    ``threshold`` gets an all-zero mask."""
    import mlx.core as mx

    existence = mx.sigmoid(existence_logits)
    probabilities = mx.sigmoid(mask_logits) * (existence > threshold)[:, None, None]
    return probabilities, existence


def segment_image(
    image: NDArray[np.uint8],
    prompts: tuple[Prompt, ...],
    weights_path: Path,
    tokenizer_path: Path,
    *,
    existence_threshold: float = 0.5,
    seed: int = 0,
    prompt_batch_size: int | None = None,
    config: ModelConfig | None = None,
) -> ImageSegmentation:
    """Port of upstream's ``predict_image``: existence-gated per-prompt probability
    maps on the original image size. Upstream applies no slice NMS or class merge
    here (that is specific to the multi-slice 3D CLI's ``postprocess``).

    ``image`` is expected to contain uint8 ``(H, W)`` or ``(H, W, 3)`` display
    intensities (0..255); this boundary checks shape but not dtype or range. It
    is zero-padded to a square and bicubically resized to the model size without
    integer truncation. Mask logits are resized back and cropped to ``(H, W)``.
    The returned ``masks`` are
    ``probabilities > 0.5``, and ``probabilities`` and ``existence_probabilities``
    are float32 with shapes ``(prompts, H, W)`` and ``(prompts,)``.
    ``prompt_batch_size`` is None or a positive maximum number of prompts per
    decoder group; output arrays still cover all prompts.

    Raises:
        ValueError: Invalid threshold, seed, prompt batch size, prompts, or image shape.
    """
    if not 0 <= existence_threshold <= 1 or type(seed) is not int or seed < 0:
        raise ValueError("Invalid existence threshold or random seed")
    if prompt_batch_size is not None and (
        type(prompt_batch_size) is not int or prompt_batch_size < 1
    ):
        raise ValueError("prompt_batch_size must be a positive integer or None")
    if not prompts or len({p.label for p in prompts}) != len(prompts):
        raise ValueError("At least one prompt with distinct labels is required")
    channels = _assemble_channels(image)
    height, width = channels.shape[:2]
    if min(height, width) < 1:
        raise ValueError("Image dimensions must be positive")
    from medmlx_core.runtime import import_mlx

    from biomedparse_mlx.mlx_types import evaluate, from_numpy
    from biomedparse_mlx.model import load_model
    from biomedparse_mlx.resize import ResizeMode, resize
    from biomedparse_mlx.text import encode_prompts

    mx = import_mlx()
    config = model_config() if config is None else config
    model = load_model(weights_path, config)
    features = encode_prompts(
        model.sem_seg_head.predictor.language_encoder, prompts, tokenizer_path, config
    )
    mx.random.seed(seed)
    pads = square_padding(height, width)
    padded = np.pad(channels, (*pads, (0, 0)))
    padded_size = max(height, width)
    batch = resize(
        from_numpy(padded[None], dtype=mx.float32),
        (config.image_size, config.image_size),
        ResizeMode.BICUBIC,
    )
    prediction = (
        model(batch, features)
        if prompt_batch_size is None
        else model(batch, features, prompt_batch_size=prompt_batch_size)
    )
    logits = resize(
        prediction.mask_logits[..., None],
        (padded_size, padded_size),
        ResizeMode.BICUBIC,
        antialias=True,
    )[..., 0]
    top, left = pads[0][0], pads[1][0]
    logits = logits[:, top : top + height, left : left + width]
    probabilities, existence = existence_gated_probabilities(
        logits, prediction.existence_logits, existence_threshold
    )
    evaluate(probabilities, existence)
    probability_array = np.array(probabilities, dtype=np.float32)
    return ImageSegmentation(
        probability_array > 0.5,
        probability_array,
        np.array(existence, dtype=np.float32),
        prompts,
        Backend.METAL,
    )
