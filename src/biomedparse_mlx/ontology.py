"""Canonical BioMedParse model definition and typed inference objects."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields
from enum import StrEnum
from importlib.resources import files
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    import numpy as np
    from mlx.core import array
    from numpy.typing import NDArray


class Backend(StrEnum):
    """Execution backend recorded in outputs; only MLX on the Metal GPU exists."""

    METAL = "mlx"


NIFTI_SUFFIXES = (".nii", ".nii.gz")
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})
DICOM_SUFFIXES = frozenset({".dcm"})


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Architecture, source pins and sampling constants from ``data/model.json``.

    ``pixel_mean`` and ``pixel_std`` are on the 0..255 intensity scale.
    ``queries`` counts all decoder queries, of which the last ``edge_queries`` feed
    the edge channel and the rest the segmentation masks. ``mask_threshold``,
    ``sample_ratio`` and ``base_temperature`` parameterize the Boltzmann attention
    sampling (see ``BoltzFormer``).
    """

    id: str
    source_repo: str
    source_revision: str
    checkpoint_repo: str
    checkpoint_revision: str
    checkpoint_filename: str
    tokenizer_repo: str
    tokenizer_revision: str
    image_size: int
    embed_dim: int
    depths: tuple[int, ...]
    focal_levels: int
    focal_window: int
    hidden_dim: int
    heads: int
    ffn_dim: int
    encoder_layers: int
    decoder_layers: int
    queries: int
    edge_queries: int
    text_layers: int
    text_context: int
    text_vocabulary: int
    pixel_mean: tuple[float, ...]
    pixel_std: tuple[float, ...]
    mask_threshold: float
    sample_ratio: float
    base_temperature: float

    def __post_init__(self) -> None:
        """Reject invalid dimensions, normalization values and sampling parameters."""
        for field in fields(self):
            value = getattr(self, field.name)
            if field.type == "int" and (type(value) is not int or value < 0):
                raise ValueError(f"{field.name} must be a nonnegative integer")
            if field.type == "str" and (not isinstance(value, str) or not value):
                raise ValueError(f"{field.name} must be a nonempty string")
        positive = (
            self.image_size,
            self.embed_dim,
            self.hidden_dim,
            self.heads,
            self.ffn_dim,
            self.focal_levels,
            self.focal_window,
            self.encoder_layers,
            self.decoder_layers,
            self.queries,
            self.text_layers,
            self.text_context,
            self.text_vocabulary,
        )
        if (
            min(positive) < 1
            or len(self.depths) != 4
            or not all(type(d) is int and d > 0 for d in self.depths)
        ):
            raise ValueError("BioMedParse requires positive dimensions and four backbone stages")
        if self.hidden_dim % self.heads or self.hidden_dim % 32 or self.hidden_dim % 4:
            raise ValueError("hidden_dim must be divisible by heads, 32 groups, and 4")
        if not 0 <= self.edge_queries < self.queries:
            raise ValueError("edge_queries must leave at least one segmentation query")
        if len(self.pixel_mean) != 3 or len(self.pixel_std) != 3 or min(self.pixel_std) <= 0:
            raise ValueError("Three pixel means and positive standard deviations are required")
        if not 0 < self.sample_ratio <= 1 or self.base_temperature <= 0:
            raise ValueError("Invalid Boltzmann sampling parameters")
        real_values = (
            *self.pixel_mean,
            *self.pixel_std,
            self.mask_threshold,
            self.sample_ratio,
            self.base_temperature,
        )
        if not all(type(v) in (int, float) and math.isfinite(v) for v in real_values):
            raise ValueError("Model numeric properties must be finite real values")

    @property
    def channels(self) -> tuple[int, ...]:
        """Backbone stage widths: ``embed_dim`` doubled at each of the four stages."""
        return tuple(self.embed_dim * 2**i for i in range(4))


def _integer(value: object, name: str) -> int:
    """Require an exact integer from the model definition, excluding booleans."""
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    return value


def _string(value: object, name: str) -> str:
    """Require a string field from the model definition."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _real(value: object, name: str) -> float:
    """Convert an integer or float field to float, excluding booleans."""
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a real number")
    # The exact-type check rejects JSON booleans, which otherwise are integers.
    return float(cast(int | float, value))


def _integer_tuple(value: object, name: str) -> tuple[int, ...]:
    """Parse a JSON list of exact integers as a tuple."""
    if not isinstance(value, list):
        raise ValueError(f"Invalid {name} in model definition")
    return tuple(_integer(item, name) for item in cast(list[object], value))


def _real_tuple(value: object, name: str) -> tuple[float, ...]:
    """Parse a JSON list of real values as a float tuple."""
    if not isinstance(value, list):
        raise ValueError(f"Invalid {name} in model definition")
    return tuple(_real(item, name) for item in cast(list[object], value))


def parse_model_config(raw: object) -> ModelConfig:
    """Validate decoded JSON fields before constructing the domain configuration."""
    if not isinstance(raw, dict):
        raise ValueError("Model definition fields do not match ModelConfig")
    values = cast(dict[object, object], raw)
    if set(values) != {field.name for field in fields(ModelConfig)}:
        raise ValueError("Model definition fields do not match ModelConfig")
    return ModelConfig(
        id=_string(values["id"], "id"),
        source_repo=_string(values["source_repo"], "source_repo"),
        source_revision=_string(values["source_revision"], "source_revision"),
        checkpoint_repo=_string(values["checkpoint_repo"], "checkpoint_repo"),
        checkpoint_revision=_string(values["checkpoint_revision"], "checkpoint_revision"),
        checkpoint_filename=_string(values["checkpoint_filename"], "checkpoint_filename"),
        tokenizer_repo=_string(values["tokenizer_repo"], "tokenizer_repo"),
        tokenizer_revision=_string(values["tokenizer_revision"], "tokenizer_revision"),
        image_size=_integer(values["image_size"], "image_size"),
        embed_dim=_integer(values["embed_dim"], "embed_dim"),
        depths=_integer_tuple(values["depths"], "depths"),
        focal_levels=_integer(values["focal_levels"], "focal_levels"),
        focal_window=_integer(values["focal_window"], "focal_window"),
        hidden_dim=_integer(values["hidden_dim"], "hidden_dim"),
        heads=_integer(values["heads"], "heads"),
        ffn_dim=_integer(values["ffn_dim"], "ffn_dim"),
        encoder_layers=_integer(values["encoder_layers"], "encoder_layers"),
        decoder_layers=_integer(values["decoder_layers"], "decoder_layers"),
        queries=_integer(values["queries"], "queries"),
        edge_queries=_integer(values["edge_queries"], "edge_queries"),
        text_layers=_integer(values["text_layers"], "text_layers"),
        text_context=_integer(values["text_context"], "text_context"),
        text_vocabulary=_integer(values["text_vocabulary"], "text_vocabulary"),
        pixel_mean=_real_tuple(values["pixel_mean"], "pixel_mean"),
        pixel_std=_real_tuple(values["pixel_std"], "pixel_std"),
        mask_threshold=_real(values["mask_threshold"], "mask_threshold"),
        sample_ratio=_real(values["sample_ratio"], "sample_ratio"),
        base_temperature=_real(values["base_temperature"], "base_temperature"),
    )


def model_config() -> ModelConfig:
    """Parse the single packaged model definition; fail on unknown/missing fields."""
    raw: object = json.loads(files("biomedparse_mlx.data").joinpath("model.json").read_text())
    return parse_model_config(raw)


@dataclass(frozen=True, slots=True)
class Prompt:
    """One text prompt and the integer label it writes into the output volume.

    ``label`` is in 1..65535 (0 is background) and must be unique within a run.
    ``text`` is a nonempty string; ``[SEP]`` is rejected so one prompt always
    means one text. Sentence count is not checked.
    """

    label: int
    text: str

    def __post_init__(self) -> None:
        """Require one nonempty prompt and a nonzero uint16 label."""
        if type(self.label) is not int or not 1 <= self.label <= 65535:
            raise ValueError("Prompt label must be an integer in 1..65535")
        if type(self.text) is not str or not self.text.strip() or "[SEP]" in self.text:
            raise ValueError("Each Prompt requires one nonempty text without [SEP]")


@dataclass(frozen=True, slots=True)
class PromptFeatures:
    """Projected language tokens; relationship to labels survives the decoder.

    ``tokens`` is ``(len(prompts), length, hidden_dim)``, one sequence per prompt in
    the same order as ``prompts``.
    """

    prompts: tuple[Prompt, ...]
    tokens: array

    def __post_init__(self) -> None:
        """Check unique prompt labels and one token sequence per prompt."""
        if not self.prompts or len({p.label for p in self.prompts}) != len(self.prompts):
            raise ValueError("PromptFeatures requires distinct, nonempty prompt bindings")
        if self.tokens.ndim != 3 or self.tokens.shape[0] != len(self.prompts):
            raise ValueError("Each prompt must bind one projected token sequence")


@dataclass(frozen=True, slots=True)
class SlicePrediction:
    """Raw per-(image, prompt) model outputs, image-major and prompt-minor.

    ``mask_logits`` and ``edge_logits`` are ``(images * prompts, h, w)`` logits at the
    pixel decoder's finest resolution; ``existence_logits`` is ``(images * prompts,)``.
    """

    mask_logits: array
    existence_logits: array
    edge_logits: array


@dataclass(frozen=True, slots=True)
class AttentionReplay:
    """Upstream sampled blocked positions for controlled numerical comparisons.

    One bool array per decoder layer plus one for the final head, each shaped like
    the decoder's attention mask; True means the position is disallowed.
    """

    blocked: tuple[array, ...]


@dataclass(frozen=True, slots=True)
class Volume:
    """Already modality-preprocessed intensities on the caller's original array grid.

    A finite, nonempty 3D array of real numbers in 0..255. Axis order is the
    caller's; the slicing axis is derived from the shape (``axis_from_shape``).
    """

    data: NDArray[
        np.int8
        | np.int16
        | np.int32
        | np.int64
        | np.uint8
        | np.uint16
        | np.uint32
        | np.uint64
        | np.float16
        | np.float32
        | np.float64
        | np.longdouble
    ]

    def __post_init__(self) -> None:
        """Require finite, nonempty 3D intensities on the 0 to 255 scale."""
        import numpy as np

        if self.data.ndim != 3 or min(self.data.shape) < 1:
            raise ValueError("Input must be one nonempty three-dimensional volume")
        if self.data.dtype.kind not in "uif" or not np.isfinite(self.data).all():
            raise ValueError("Input volume must contain finite real intensities")
        if self.data.min() < 0 or self.data.max() > 255:
            raise ValueError("Provide upstream modality-preprocessed intensities in 0..255")


@dataclass(frozen=True, slots=True)
class VolumeGeometry:
    """How a volume maps to square model slices and back.

    ``axis`` is the slicing axis, ``plane_shape`` the in-plane ``(height, width)``,
    ``padding`` the zero padding ``((top, bottom), (left, right))`` that makes the
    plane a square of side ``square_size``.
    """

    axis: int
    original_shape: tuple[int, int, int]
    plane_shape: tuple[int, int]
    padding: tuple[tuple[int, int], tuple[int, int]]
    square_size: int


@dataclass(frozen=True, slots=True)
class Segmentation:
    """Volume result: ``labels`` has the input shape, with 0 as background.

    ``existence_probabilities`` is float32 ``(prompts, slices)`` along the slicing axis.
    """

    labels: NDArray[np.uint16]
    existence_probabilities: NDArray[np.float32]
    prompts: tuple[Prompt, ...]
    backend: Backend
    # Edge-refined 3D instance split (upstream's example notebook); opt-in only.
    instances: NDArray[np.uint16] | None = None


@dataclass(frozen=True, slots=True)
class ImageSegmentation:
    """Per-prompt overlapping masks, as upstream's 2D ``inference_2d.py`` reports them.

    ``masks`` (bool) and ``probabilities`` (float32) are ``(prompts, H, W)`` on the
    input image grid; ``existence_probabilities`` is ``(prompts,)``.
    """

    masks: NDArray[np.bool_]
    probabilities: NDArray[np.float32]
    existence_probabilities: NDArray[np.float32]
    prompts: tuple[Prompt, ...]
    backend: Backend


@dataclass(frozen=True, slots=True)
class RunReceipt:
    """Provenance and settings of one run, serialized to ``run.json``.

    ``shape`` is the output label array shape (the mask array shape for 2D input).
    ``elapsed_seconds`` is wall-clock time from the start of the run to just after
    the ``segmentation.npz`` write, so it includes model loading and inference.
    """

    model: str
    source_revision: str
    prompts: tuple[Prompt, ...]
    backend: str
    precision: str
    shape: tuple[int, ...]
    slice_batch_size: int
    seed: int
    existence_threshold: float
    elapsed_seconds: float
    instance_segmentation: bool = False
    prompt_batch_size: int | None = None
