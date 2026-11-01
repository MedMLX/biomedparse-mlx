"""Native BioMedParse v2 inference graph; import through the lazy public runner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, cast

import mlx.core as mx
import mlx.nn as nn

from biomedparse_mlx.backbone import FocalNet
from biomedparse_mlx.decoder import BoltzFormer
from biomedparse_mlx.mlx_types import evaluate, flatten_parameters
from biomedparse_mlx.ontology import AttentionReplay, ModelConfig, PromptFeatures, SlicePrediction
from biomedparse_mlx.pixel_decoder import PixelDecoder


class SafetensorsLoader(Protocol):
    """Typed signature for ``mx.load`` with an explicit safetensors format."""

    def __call__(self, file: str, /, *, format: Literal["safetensors"]) -> dict[str, mx.array]:
        """Load named MLX arrays from a safetensors file."""
        ...


@dataclass(frozen=True)
class EncodedImages:
    """Evaluated image features reusable with this model's unchanged weights.

    Keep one instance per image batch. Releasing it drops its feature references;
    other references can keep those buffers alive. It is bound to its encoder
    instance so features cannot silently cross models.
    """

    owner: BioMedParse
    masks: mx.array
    scales: tuple[mx.array, ...]


class MaskFormerHead(nn.Module):
    """Pixel decoder plus the BoltzFormer predictor."""

    def __init__(self, config: ModelConfig) -> None:
        """Build the pixel decoder and text-conditioned query predictor."""
        super().__init__()
        self.pixel_decoder = PixelDecoder(config)
        self.predictor = BoltzFormer(config)


class BioMedParse(nn.Module):
    """FocalNet backbone, pixel decoder and BoltzFormer for three-channel images."""

    def __init__(self, config: ModelConfig) -> None:
        """Build the image backbone and segmentation head for the configuration."""
        from medmlx_core.runtime import import_mlx

        import_mlx()
        super().__init__()
        self.backbone = FocalNet(config)
        self.sem_seg_head = MaskFormerHead(config)
        # The 3D path uses neighboring slices, so RGB statistics are averaged
        # into one shared mean and std; the 2D path uses the same model buffers.
        self.pixel_mean = mx.full((1, 1, 1, 3), sum(config.pixel_mean) / 3)
        self.pixel_std = mx.full((1, 1, 1, 3), sum(config.pixel_std) / 3)
        self._config = config

    def __call__(
        self,
        images: mx.array,
        prompts: PromptFeatures,
        *,
        sampling: AttentionReplay | None = None,
        prompt_batch_size: int | None = None,
    ) -> SlicePrediction:
        """Predict masks, existence and edges for each (image, prompt) pair.

        Args:
            images: Float32 ``(B, S, S, 3)`` NHWC batch, ``S = config.image_size``,
                on the 0..255 intensity scale. Volume inference uses channels
                current/previous/next slice; 2D inference uses RGB or repeated
                grayscale. The model normalizes with averages of the configured RGB
                means and standard deviations. Bicubic interpolation can overshoot
                that range. Shape and dtype are checked; finiteness is the caller's
                responsibility at this boundary.
            prompts: Projected language tokens for ``P`` prompts.
            sampling: Upstream's sampled attention masks to replay, for numerical
                comparison; None draws new samples from MLX's global generator.
            prompt_batch_size: Maximum prompts repeated through the decoder per
                group; None retains the full batch. Sampling fields and returned
                logits still cover all prompts (see ``decode_prompts``).

        Returns:
            A ``SlicePrediction`` with leading axis ``B * P``, image-major and
            prompt-minor: ``mask_logits`` and ``edge_logits`` ``(B * P, h, w)`` at
            the pixel decoder's finest resolution (``S / 4``) and ``existence_logits``
            ``(B * P,)``. Mask logits average the segmentation queries; edge logits
            average the last ``edge_queries`` (zeros if the model has none).

        Raises:
            ValueError: Images or prompt tokens do not match the model configuration.
        """
        return self.decode_prompts(
            self.encode_images(images),
            prompts,
            sampling=sampling,
            prompt_batch_size=prompt_batch_size,
        )

    def encode_images(self, images: mx.array) -> EncodedImages:
        """Encode one NHWC three-channel batch once for repeated prompt requests."""
        config = self._config
        if images.ndim != 4 or images.shape[-1] != 3 or images.dtype != mx.float32:
            raise ValueError("BioMedParse expects float32 NHWC neighboring-slice images")
        if images.shape[1:3] != (config.image_size, config.image_size):
            raise ValueError(f"BioMedParse requires {config.image_size} square images")
        features = self.backbone((images - self.pixel_mean) / self.pixel_std)
        masks, scales = self.sem_seg_head.pixel_decoder(features)
        evaluate(masks, scales)
        return EncodedImages(self, masks, scales)

    def decode_prompts(
        self,
        images: EncodedImages,
        prompts: PromptFeatures,
        *,
        sampling: AttentionReplay | None = None,
        prompt_batch_size: int | None = None,
    ) -> SlicePrediction:
        """Decode bounded prompt groups, retaining image-major output and RNG draws.

        Prompt batching bounds repeated image features and decoder activations.
        To preserve the original random sequence, uniform fields for the original
        full batch are drawn once in layer order and sliced for each group. These
        fields and returned logits still scale with total prompt count. Replay
        masks bypass random draws entirely. Global NMS/label merging belongs after
        this call, once all prompt results have been reunited.
        """
        if images.owner is not self:
            raise ValueError("Encoded images belong to a different BioMedParse model")
        count, batch = len(prompts.prompts), images.masks.shape[0]
        if prompts.tokens.shape[-1] != self._config.hidden_dim:
            raise ValueError("Prompt token shape differs from model configuration")
        if prompt_batch_size is not None and (
            type(prompt_batch_size) is not int or prompt_batch_size < 1
        ):
            raise ValueError("prompt_batch_size must be a positive integer or None")
        if sampling is not None and len(sampling.blocked) != self._config.decoder_layers + 1:
            raise ValueError("Sampling replay must include every decoder prediction head")
        size = count if prompt_batch_size is None else min(count, prompt_batch_size)
        if size == count:
            return self._decode_group(images, prompts, sampling=sampling)
        uniforms: tuple[mx.array, ...] | None = None
        if sampling is None:
            uniforms = tuple(
                mx.random.uniform(
                    shape=(
                        batch * count,
                        self._config.heads,
                        self._config.queries,
                        images.scales[layer % 3].shape[1] * images.scales[layer % 3].shape[2],
                    )
                )
                for layer in range(self._config.decoder_layers + 1)
            )
            evaluate(uniforms)
        parts: list[SlicePrediction] = []
        for start in range(0, count, size):
            stop = min(start + size, count)
            indices = mx.reshape(
                mx.arange(batch)[:, None] * count + mx.arange(start, stop)[None], (-1,)
            )
            replay = (
                AttentionReplay(tuple(value[indices] for value in sampling.blocked))
                if sampling is not None
                else None
            )
            selected = PromptFeatures(prompts.prompts[start:stop], prompts.tokens[start:stop])
            part = self._decode_group(
                images,
                selected,
                sampling=replay,
                uniforms=None if uniforms is None else tuple(value[indices] for value in uniforms),
            )
            evaluate(part.mask_logits, part.existence_logits, part.edge_logits)
            parts.append(part)

        def join(values: list[mx.array]) -> mx.array:
            """Concatenate a group of prediction arrays along the pair axis."""
            shape = values[0].shape[1:]
            return mx.reshape(
                mx.concatenate(
                    [mx.reshape(value, (batch, -1, *shape)) for value in values], axis=1
                ),
                (batch * count, *shape),
            )

        return SlicePrediction(
            join([part.mask_logits for part in parts]),
            join([part.existence_logits for part in parts]),
            join([part.edge_logits for part in parts]),
        )

    def _decode_group(
        self,
        images: EncodedImages,
        prompts: PromptFeatures,
        *,
        sampling: AttentionReplay | None = None,
        uniforms: tuple[mx.array, ...] | None = None,
    ) -> SlicePrediction:
        """Decode one prompt group and average segmentation and edge query logits."""
        config = self._config
        count = len(prompts.prompts)
        masks, scales = images.masks, images.scales
        masks = mx.repeat(masks, count, axis=0)
        scales = tuple(mx.repeat(x, count, axis=0) for x in scales)
        text = mx.tile(prompts.tokens, (images.masks.shape[0], 1, 1))
        logits, existence = self.sem_seg_head.predictor(
            scales, masks, text, sampling=sampling, uniforms=uniforms
        )
        cutoff = config.queries - config.edge_queries
        edges = (
            logits[:, cutoff:].mean(axis=1) if config.edge_queries else mx.zeros_like(logits[:, 0])
        )
        return SlicePrediction(logits[:, :cutoff].mean(axis=1), existence[:, 0], edges)


def build_model(config: ModelConfig) -> BioMedParse:
    """Initialize a model without loading a checkpoint, after Metal admission."""
    return BioMedParse(config)


def load_model(weights_path: Path, config: ModelConfig) -> BioMedParse:
    """Build the model and load native safetensors weights.

    Strict shape/key, float32 and finite-value admission: every parameter must be
    present with the right shape and dtype, and finite. The format is explicit so
    named symlinks to extensionless Hub blobs are supported.

    Raises:
        FileNotFoundError: ``weights_path`` is not a file.
        ValueError: A tensor is not float32 or contains a non-finite value.
            ``load_weights(strict=True)`` also rejects missing, extra or
            mis-shaped keys.
    """
    if not weights_path.is_file():
        raise FileNotFoundError(f"Converted MLX weights not found: {weights_path}")
    model = build_model(config)
    load = cast(SafetensorsLoader, mx.load)
    model.load_weights(list(load(str(weights_path), format="safetensors").items()), strict=True)
    parameters = flatten_parameters(model.parameters())
    for name, tensor in parameters:
        if tensor.dtype != mx.float32:
            raise ValueError(f"{name} must be float32; found {tensor.dtype}")
    finite = [mx.all(mx.isfinite(tensor)) for _, tensor in parameters]
    evaluate(finite)
    for (name, _), valid in zip(parameters, finite, strict=True):
        if not valid.item():
            raise ValueError(f"Non-finite MLX weight: {name}")
    model.eval()
    return model
