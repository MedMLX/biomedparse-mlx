"""Exact interfaces used from the optional Torch and pinned BioMedParse oracle."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypedDict, cast

import numpy as np
from numpy.typing import NDArray

from biomedparse_mlx.dependency_types import callable_member

if TYPE_CHECKING:
    from torch import Generator, Tensor


class TorchArrayFactory(Protocol):
    def __call__(self, ndarray: NDArray[np.generic]) -> Tensor:
        """Create a Torch tensor from a NumPy array."""
        ...


class TorchResize(Protocol):
    def __call__(
        self,
        input: Tensor,
        *,
        size: tuple[int, int],
        mode: str,
        align_corners: bool | None = None,
        antialias: bool = False,
    ) -> Tensor:
        """Resize a tensor with the selected Torch interpolation options."""
        ...


class TorchCheckpointLoader(Protocol):
    def __call__(
        self,
        f: Path,
        *,
        map_location: str,
        weights_only: bool,
        mmap: bool,
    ) -> object:
        """Load a checkpoint using the requested Torch safety and mapping options."""
        ...


class TorchManualSeed(Protocol):
    def __call__(self, seed: int) -> Generator:
        """Set the Torch random seed and return its generator."""
        ...


def manual_seed(seed: int) -> None:
    """Set the optional Torch oracle random seed."""
    import torch

    seed_runtime = cast(TorchManualSeed, callable_member(torch, "manual_seed"))
    seed_runtime(seed)


def from_numpy(values: NDArray[np.generic]) -> Tensor:
    """Convert a NumPy array to a Torch tensor for oracle comparisons."""
    import torch

    factory = cast(TorchArrayFactory, callable_member(torch, "from_numpy"))
    return factory(values)


def interpolate(
    values: Tensor,
    *,
    size: tuple[int, int],
    mode: str,
    align_corners: bool | None = None,
    antialias: bool = False,
) -> Tensor:
    """Apply Torch interpolation with the specified spatial settings."""
    import torch.nn.functional as functional

    resize = cast(TorchResize, functional.interpolate)
    return resize(values, size=size, mode=mode, align_corners=align_corners, antialias=antialias)


def checkpoint_state(path: Path) -> dict[str, Tensor]:
    """Read named oracle checkpoint tensors, excluding loss parameters."""
    import torch

    load = cast(TorchCheckpointLoader, torch.load)
    raw = load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(raw, Mapping):
        raise ValueError("Checkpoint must be a mapping")
    payload = cast(Mapping[object, object], raw)
    state = payload["state_dict"]
    if not isinstance(state, Mapping):
        raise ValueError("Checkpoint state_dict must be a mapping")
    tensors: dict[str, Tensor] = {}
    for key, value in cast(Mapping[object, object], state).items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise ValueError("Checkpoint state_dict must contain named tensors")
        if not key.startswith("loss_function."):
            tensors[key.removeprefix("model.")] = value
    return tensors


def numpy_float(values: Tensor) -> NDArray[np.float32]:
    """Convert a Torch tensor to a NumPy float32 array."""
    return np.asarray(values.numpy(), dtype=np.float32)


def numpy_bool(values: Tensor) -> NDArray[np.bool_]:
    """Convert a Torch tensor to a NumPy boolean array."""
    return np.asarray(values.numpy(), dtype=np.bool_)


def numpy_int(values: Tensor) -> NDArray[np.int64]:
    """Convert a Torch tensor to a NumPy int64 array."""
    return np.asarray(values.numpy(), dtype=np.int64)


class LanguageOutput(TypedDict):
    tokens: dict[str, Tensor]
    token_emb: Tensor
    class_emb: Tensor


class LanguageEncoder(Protocol):
    def encoder_transformer(self, token_ids: Tensor) -> dict[str, Tensor]:
        """Encode token IDs and return the upstream language outputs."""
        ...

    def get_text_token_embeddings(
        self,
        txts: list[str],
        name: str = "default",
        token: bool = False,
        norm: bool = False,
    ) -> LanguageOutput:
        """Encode text prompts with the upstream tokenizer and language model."""
        ...


class PredictionHead(Protocol):
    def __call__(
        self,
        output: Tensor,
        mask_features: Tensor,
        attn_mask_target_size: tuple[int, int],
        layer_id: int = -1,
    ) -> tuple[Tensor, Tensor]:
        """Return upstream mask logits and the next attention mask."""
        ...


class Predictor(Protocol):
    language_encoder: LanguageEncoder
    forward_prediction_heads: PredictionHead


class PredictionOutput(TypedDict):
    pred_gmasks: Tensor
    object_existence: Tensor


class SegmentationHead(Protocol):
    predictor: Predictor

    def __call__(
        self,
        features: Mapping[str, Tensor],
        extra: dict[str, Tensor],
    ) -> PredictionOutput:
        """Predict masks and existence scores from image features and prompts."""
        ...


class ReferenceModel(Protocol):
    sem_seg_head: SegmentationHead
    pixel_mean: Tensor
    pixel_std: Tensor

    def backbone(self, values: Tensor) -> dict[str, Tensor]:
        """Extract upstream image feature maps from an input tensor."""
        ...

    def eval(self) -> ReferenceModel:
        """Switch the reference model to evaluation mode."""
        ...

    def state_dict(self) -> Mapping[str, Tensor]:
        """Return named reference model tensors."""
        ...

    def load_state_dict(self, state_dict: Mapping[str, Tensor], strict: bool = True) -> object:
        """Load named reference tensors with optional strict key matching."""
        ...
