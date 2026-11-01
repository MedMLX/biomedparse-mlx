"""BoltzFormer and instance existence decoder adapted from Microsoft BioMedParse."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from biomedparse_mlx.layers import FFN, CrossAttention, SelfAttention, position_encoding
from biomedparse_mlx.mlx_types import relu
from biomedparse_mlx.ontology import AttentionReplay, ModelConfig
from biomedparse_mlx.resize import ResizeMode, resize
from biomedparse_mlx.text import LanguageEncoder


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

    def prediction_heads(
        self,
        output: mx.array,
        mask_features: mx.array,
        size: tuple[int, int],
        layer_id: int,
        blocked: mx.array | None = None,
        uniform: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Predict mask logits and sample the next cross-attention mask.

        Args:
            output: Query features ``(B, queries, hidden)``.
            mask_features: Pixel-decoder features ``(B, h, w, hidden)``.
            size: ``(height, width)`` of the scale the next cross-attention reads.
            layer_id: Decoder layer index (-1 for the initial head); sets the
                sampling temperature ``base_temperature / (2 + layer_id)``.
            blocked: Upstream's sampled mask to replay instead of drawing one.
            uniform: Optional FP32 uniforms with the attention-mask shape, sliced
                from original full-batch draws to retain RNG behavior in groups.

        Returns:
            Mask logits ``(B, queries, h, w)`` and a boolean attention mask
            ``(B, heads, queries, size_h * size_w)`` where True means the position
            is disallowed. A query whose positions are all disallowed is opened
            fully, as upstream does.
        """
        embedding = self.mask_embed(self.decoder_norm(output))
        batch, height, width, channels = mask_features.shape
        flattened = mx.transpose(
            mx.reshape(mask_features, (batch, height * width, channels)), (0, 2, 1)
        )
        masks = mx.reshape(embedding @ flattened, (batch, self._config.queries, height, width))
        resized = resize(mx.transpose(masks, (0, 2, 3, 1)), size, ResizeMode.BILINEAR)
        probabilities = mx.reshape(mx.sigmoid(resized), (batch, -1, self._config.queries))
        probabilities = mx.transpose(probabilities, (0, 2, 1))[:, None]
        probabilities = mx.broadcast_to(
            probabilities, (batch, self._config.heads, self._config.queries, size[0] * size[1])
        )
        if blocked is not None:
            if blocked.shape != probabilities.shape or blocked.dtype != mx.bool_:
                raise ValueError("Replay requires bool blocked positions with the attention shape")
            return masks, blocked & ~mx.all(blocked, axis=-1, keepdims=True)
        # `remaining` is the marginal probability that a position is never
        # selected by sample_ratio * H * W categorical draws. Independent
        # uniforms apply those marginals per position; they do not reproduce
        # the joint distribution of an actual set of categorical draws.
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
            scales: Three NHWC feature maps, coarsest first.
            mask_features: Finest pixel-decoder features ``(B, h, w, hidden)``.
            text: Projected prompt tokens ``(B, L, hidden)``, one sequence per batch row.
            sampling: Upstream's sampled attention masks to replay; needs one entry per
                decoder layer plus one for the final head.
            uniforms: Pre-drawn uniform fields, one per prediction head; mutually
                exclusive with ``sampling``. Used by prompt batching.

        Returns:
            Mask logits ``(B, queries, h, w)`` and existence logits ``(B, 1)``.
        """
        config = self._config
        if sampling is not None and len(sampling.blocked) != config.decoder_layers + 1:
            raise ValueError("Sampling replay must include the initial and final prediction heads")
        if uniforms is not None and (
            sampling is not None or len(uniforms) != config.decoder_layers + 1
        ):
            raise ValueError("Uniform sampling requires one field per head and no blocked replay")
        batch = mask_features.shape[0]
        src = [
            mx.reshape(x, (batch, -1, config.hidden_dim)) + self.level_embed.weight[i]
            for i, x in enumerate(scales)
        ]
        pos = [position_encoding(x.shape[1], x.shape[2], config.hidden_dim) for x in scales]
        queries = mx.broadcast_to(
            self.query_feat_.weight, (batch, config.queries, config.hidden_dim)
        )
        query_pos = mx.broadcast_to(self.query_embed_.weight, queries.shape)
        cls = mx.broadcast_to(self.classifier_query.weight, (batch, 1, config.hidden_dim))
        combined_pos = mx.concatenate([query_pos, text], axis=1)
        combined = self.initial_self_attention_layer(
            mx.concatenate([queries, text], axis=1), combined_pos
        )
        queries = self.initial_ffn_layer(combined[:, : config.queries])
        text = combined[:, config.queries :]
        for i in range(config.decoder_layers):
            level = i % 3
            _, mask = self.prediction_heads(
                queries,
                mask_features,
                (scales[level].shape[1], scales[level].shape[2]),
                i - 1,
                None if sampling is None else sampling.blocked[i],
                None if uniforms is None else uniforms[i],
            )
            queries = self.transformer_cross_attention_layers[i](
                queries,
                src[level],
                pos[level],
                query_pos,
                mask,
            )
            combined = self.transformer_self_attention_layers[i](
                mx.concatenate([queries, text], axis=1),
                combined_pos,
            )
            queries = self.transformer_ffn_layers[i](combined[:, : config.queries])
            text = combined[:, config.queries :]
            cls = self.classifier_cross_attention_layers[i](cls, combined, combined_pos)
            cls = self.classifier_ffn_layers[i](cls)
        level = config.decoder_layers % 3
        masks, _ = self.prediction_heads(
            queries,
            mask_features,
            (scales[level].shape[1], scales[level].shape[2]),
            config.decoder_layers - 1,
            None if sampling is None else sampling.blocked[-1],
            None if uniforms is None else uniforms[-1],
        )
        return masks, self.classifier(cls[:, 0])
