"""BoltzFormer and instance existence decoder adapted from Microsoft BioMedParse."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from biomedparse_mlx.layers import FFN, CrossAttention, SelfAttention, position_encoding
from biomedparse_mlx.mlx_types import relu
from biomedparse_mlx.ontology import AttentionReplay, ModelConfig
from biomedparse_mlx.resize import ResizeMode, resize
from biomedparse_mlx.text import LanguageEncoder


@dataclass(frozen=True, slots=True)
class ImageMemory:
    """Prompt-independent decoder inputs for ``B`` encoded images.

    ``sources`` are the three flattened scales ``(B, h*w, hidden)`` plus their
    level embeddings, coarsest first; ``positions`` their sine encodings
    ``(1, h*w, hidden)``; ``sizes`` their ``(h, w)``. ``mask_features`` is the
    finest map ``(B, h, w, hidden)``. Prompt rows share these per-image arrays.
    """

    mask_features: mx.array
    sources: tuple[mx.array, ...]
    positions: tuple[mx.array, ...]
    sizes: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class QueryState:
    """Prompt-specific decoder state between layers, one row per image-prompt pair.

    ``queries`` ``(rows, queries, hidden)``, ``text`` ``(rows, L, hidden)`` and the
    classifier token ``cls`` ``(rows, 1, hidden)`` change per layer;
    ``query_pos`` and ``combined_pos`` (query positions plus the initial text
    tokens) stay fixed.
    """

    queries: mx.array
    text: mx.array
    cls: mx.array
    query_pos: mx.array
    combined_pos: mx.array


class MaskMlp(nn.Module):
    """Three-layer MLP (ReLU between layers) mapping query features to mask embeddings."""

    def __init__(self, width: int) -> None:
        """Create three same-width projections for query mask embeddings."""
        super().__init__()
        self.layers = [nn.Linear(width, width) for _ in range(3)]

    def __call__(self, values: mx.array) -> mx.array:
        """Project queries through two ReLU layers and a final linear layer."""
        for layer in self.layers[:-1]:
            values = relu(layer(values))
        return self.layers[-1](values)


class BoltzFormer(nn.Module):
    """Query decoder with Boltzmann-sampled cross-attention and an existence classifier.

    Queries cross-attend to three feature scales in turn (coarsest first), through
    attention masks sampled from the mask prediction made just before each
    cross-attention. A separate classifier token attends to the queries and text
    and yields one existence logit. The batch axis here is one row per image-prompt
    pair.
    """

    def __init__(self, config: ModelConfig) -> None:
        """Create query, text and classifier layers from the model configuration."""
        super().__init__()
        width, heads, hidden = config.hidden_dim, config.heads, config.ffn_dim
        self.language_encoder = LanguageEncoder(config)
        self.initial_self_attention_layer = SelfAttention(width, heads)
        self.initial_ffn_layer = FFN(width, hidden)
        self.transformer_self_attention_layers = [
            SelfAttention(width, heads) for _ in range(config.decoder_layers)
        ]
        self.transformer_cross_attention_layers = [
            CrossAttention(width, heads) for _ in range(config.decoder_layers)
        ]
        self.transformer_ffn_layers = [FFN(width, hidden) for _ in range(config.decoder_layers)]
        self.classifier_cross_attention_layers = [
            CrossAttention(width, heads) for _ in range(config.decoder_layers)
        ]
        self.classifier_ffn_layers = [FFN(width, hidden) for _ in range(config.decoder_layers)]
        self.classifier = nn.Linear(width, 1)
        self.decoder_norm = nn.LayerNorm(width)
        self.query_feat_ = nn.Embedding(config.queries, width)
        self.query_embed_ = nn.Embedding(config.queries, width)
        self.classifier_query = nn.Embedding(1, width)
        self.level_embed = nn.Embedding(3, width)
        self.mask_embed = MaskMlp(width)
        self.class_embed = mx.zeros((width, width))
        self._config = config

    def image_memory(self, scales: tuple[mx.array, ...], mask_features: mx.array) -> ImageMemory:
        """Collect the prompt-independent decoder inputs of ``B`` encoded images."""
        width = self._config.hidden_dim
        batch = mask_features.shape[0]
        return ImageMemory(
            mask_features,
            tuple(
                mx.reshape(x, (batch, -1, width)) + self.level_embed.weight[i]
                for i, x in enumerate(scales)
            ),
            tuple(position_encoding(x.shape[1], x.shape[2], width) for x in scales),
            tuple((x.shape[1], x.shape[2]) for x in scales),
        )

    def attention_shape(self, rows: int, memory: ImageMemory, layer: int) -> tuple[int, ...]:
        """Attention-mask shape of prediction head ``layer`` for ``rows`` image-prompt pairs.

        ``layer`` is ``0..decoder_layers``; the last is the final head, whose mask
        is discarded but whose uniform field is still drawn to keep the RNG stream.
        """
        height, width = memory.sizes[layer % 3]
        return (rows, self._config.heads, self._config.queries, height * width)

    def initial_state(self, text: mx.array) -> QueryState:
        """Initial query/text self-attention for prompt tokens ``(rows, L, hidden)``."""
        config = self._config
        rows = text.shape[0]
        queries = mx.broadcast_to(
            self.query_feat_.weight, (rows, config.queries, config.hidden_dim)
        )
        query_pos = mx.broadcast_to(self.query_embed_.weight, queries.shape)
        cls = mx.broadcast_to(self.classifier_query.weight, (rows, 1, config.hidden_dim))
        # Upstream keeps the initial text tokens as the text positions in every layer.
        combined_pos = mx.concatenate([query_pos, text], axis=1)
        combined = self.initial_self_attention_layer(
            mx.concatenate([queries, text], axis=1), combined_pos
        )
        return QueryState(
            self.initial_ffn_layer(combined[:, : config.queries]),
            combined[:, config.queries :],
            cls,
            query_pos,
            combined_pos,
        )

    def layer_memory(self, layer: int, memory: ImageMemory) -> tuple[mx.array, mx.array]:
        """Project the per-image keys/values read by cross-attention ``layer``."""
        level = layer % 3
        return self.transformer_cross_attention_layers[layer].project_memory(
            memory.sources[level], memory.positions[level]
        )

    def decoder_layer(
        self,
        state: QueryState,
        layer: int,
        keys_values: tuple[mx.array, mx.array],
        mask: mx.array,
        *,
        prompts: int = 1,
    ) -> QueryState:
        """Apply decoder ``layer`` after its prediction head produced ``mask``.

        ``keys_values`` comes from ``layer_memory`` for ``B`` images; ``state`` has
        ``B * prompts`` image-major rows.
        """
        config = self._config
        queries = self.transformer_cross_attention_layers[layer].attend(
            state.queries, keys_values, state.query_pos, mask, prompts=prompts
        )
        combined = self.transformer_self_attention_layers[layer](
            mx.concatenate([queries, state.text], axis=1),
            state.combined_pos,
        )
        cls = self.classifier_cross_attention_layers[layer](
            state.cls, combined, state.combined_pos
        )
        return QueryState(
            self.transformer_ffn_layers[layer](combined[:, : config.queries]),
            combined[:, config.queries :],
            self.classifier_ffn_layers[layer](cls),
            state.query_pos,
            state.combined_pos,
        )

    def prediction_heads(
        self,
        output: mx.array,
        mask_features: mx.array,
        size: tuple[int, int],
        layer_id: int,
        blocked: mx.array | None = None,
        uniform: mx.array | None = None,
        *,
        prompts: int = 1,
    ) -> tuple[mx.array, mx.array]:
        """Predict mask logits and sample the next cross-attention mask.

        Args:
            output: Query features ``(B * prompts, queries, hidden)``, image-major.
            mask_features: Per-image pixel-decoder features ``(B, h, w, hidden)``;
                each image's prompt rows contract against that image's features
                without repeating them.
            size: ``(height, width)`` of the scale the next cross-attention reads.
            layer_id: Decoder layer index (-1 for the initial head); sets the
                sampling temperature ``base_temperature / (2 + layer_id)``.
            blocked: Upstream's sampled mask to replay instead of drawing one.
            uniform: Optional FP32 uniforms with the attention-mask shape, sliced
                from original full-batch draws to retain RNG behavior in groups.
            prompts: Prompt rows per image in ``output``.

        Returns:
            Mask logits ``(B * prompts, queries, h, w)`` and a boolean attention mask
            ``(B * prompts, heads, queries, size_h * size_w)`` where True means the
            position is disallowed. A query whose positions are all disallowed is
            opened fully, as upstream does.
        """
        embedding = self.mask_embed(self.decoder_norm(output))
        batch, height, width, channels = mask_features.shape
        rows = batch * prompts
        flattened = mx.transpose(
            mx.reshape(mask_features, (batch, height * width, channels)), (0, 2, 1)
        )
        # Broadcast each image's features over its prompts; the per-row matrix
        # product keeps the (queries, hidden) x (hidden, h*w) shape of one pair.
        masks = mx.reshape(
            mx.reshape(embedding, (batch, prompts, self._config.queries, channels))
            @ flattened[:, None],
            (rows, self._config.queries, height, width),
        )
        resized = resize(mx.transpose(masks, (0, 2, 3, 1)), size, ResizeMode.BILINEAR)
        probabilities = mx.reshape(mx.sigmoid(resized), (rows, -1, self._config.queries))
        probabilities = mx.transpose(probabilities, (0, 2, 1))[:, None]
        probabilities = mx.broadcast_to(
            probabilities, (rows, self._config.heads, self._config.queries, size[0] * size[1])
        )
        if blocked is not None:
            if blocked.shape != probabilities.shape or blocked.dtype != mx.bool_:
                raise ValueError("Replay requires bool blocked positions with the attention shape")
            return masks, blocked & ~mx.all(blocked, axis=-1, keepdims=True)
        # `remaining` is the probability that a position is never selected by
        # sample_ratio * H * W categorical draws. As in the pinned upstream
        # source (`torch.rand_like(boltzmann_prob) < masked_prob`), each
        # position and head is then blocked by its own independent uniform:
        # the marginals of categorical draws, without their negative
        # correlation between positions. Only the generator differs from Torch.
        # Positions at or above mask_threshold are never blocked; the packaged
        # threshold 1.1 exceeds every sigmoid value, so all are eligible.
        eligible = probabilities < self._config.mask_threshold
        temp = self._config.base_temperature / (2 + layer_id)
        weights = mx.exp(probabilities / temp) * eligible
        weights = weights / weights.sum(axis=-1, keepdims=True)
        remaining = (1 - weights) ** int(size[0] * size[1] * self._config.sample_ratio)
        if uniform is None:
            uniform = mx.random.uniform(shape=remaining.shape)
        elif uniform.shape != remaining.shape or uniform.dtype != mx.float32:
            raise ValueError("Sampling uniforms must match the FP32 attention shape")
        disallowed = eligible & (uniform < remaining)
        disallowed = disallowed & ~mx.all(disallowed, axis=-1, keepdims=True)
        return masks, disallowed

    def __call__(
        self,
        scales: tuple[mx.array, ...],
        mask_features: mx.array,
        text: mx.array,
        *,
        sampling: AttentionReplay | None = None,
        uniforms: tuple[mx.array, ...] | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Decode queries against the pixel-decoder scales and prompt tokens.

        Args:
            scales: Three NHWC feature maps for ``B`` images, coarsest first.
            mask_features: Finest pixel-decoder features ``(B, h, w, hidden)``.
            text: Projected prompt tokens ``(B * P, L, hidden)``, image-major: one
                sequence per image-prompt pair. Image features are shared by the
                ``P`` prompt rows of their image rather than repeated.
            sampling: Upstream's sampled attention masks to replay; needs one entry per
                decoder layer plus one for the final head.
            uniforms: Pre-drawn uniform fields, one per prediction head; mutually
                exclusive with ``sampling``.

        Returns:
            Mask logits ``(B * P, queries, h, w)`` and existence logits ``(B * P, 1)``.
        """
        config = self._config
        if sampling is not None and len(sampling.blocked) != config.decoder_layers + 1:
            raise ValueError("Sampling replay must include the initial and final prediction heads")
        if uniforms is not None and (
            sampling is not None or len(uniforms) != config.decoder_layers + 1
        ):
            raise ValueError("Uniform sampling requires one field per head and no blocked replay")
        batch = mask_features.shape[0]
        if text.shape[0] % batch:
            raise ValueError("Prompt rows must be a whole number of prompts per image")
        prompts = text.shape[0] // batch
        memory = self.image_memory(scales, mask_features)
        state = self.initial_state(text)
        for i in range(config.decoder_layers):
            _, mask = self.prediction_heads(
                state.queries,
                mask_features,
                memory.sizes[i % 3],
                i - 1,
                None if sampling is None else sampling.blocked[i],
                None if uniforms is None else uniforms[i],
                prompts=prompts,
            )
            state = self.decoder_layer(
                state, i, self.layer_memory(i, memory), mask, prompts=prompts
            )
        masks, _ = self.prediction_heads(
            state.queries,
            mask_features,
            memory.sizes[config.decoder_layers % 3],
            config.decoder_layers - 1,
            None if sampling is None else sampling.blocked[-1],
            None if uniforms is None else uniforms[-1],
            prompts=prompts,
        )
        return masks, self.classifier(state.cls[:, 0])
