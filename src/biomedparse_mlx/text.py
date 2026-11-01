"""Native causal language transformer (Microsoft X-Decoder MIT adaptation)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol, cast

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from biomedparse_mlx.dependency_types import callable_member
from biomedparse_mlx.layers import Attention
from biomedparse_mlx.mlx_types import evaluate, from_numpy
from biomedparse_mlx.ontology import ModelConfig, Prompt, PromptFeatures


class TextMlp(nn.Module):
    """Two-layer text projection with QuickGELU, as in CLIP's text transformer."""

    def __init__(self, width: int) -> None:
        """Create the four-times-width text projection and output layer."""
        super().__init__()
        self.c_fc = nn.Linear(width, 4 * width)
        self.c_proj = nn.Linear(4 * width, width)

    def __call__(self, values: mx.array) -> mx.array:
        """Apply QuickGELU between the text projection layers."""
        values = self.c_fc(values)
        # QuickGELU, x * sigmoid(1.702 x), as in CLIP's text transformer.
        return self.c_proj(values * mx.sigmoid(1.702 * values))


class ResidualAttentionBlock(nn.Module):
    """Pre-norm causal transformer block, as in CLIP's text encoder."""

    def __init__(self, width: int, heads: int) -> None:
        """Create causal attention and the pre-normalized text MLP."""
        super().__init__()
        self.attn = Attention(width, heads)
        self.ln_1 = nn.LayerNorm(width, eps=1e-12)
        self.ln_2 = nn.LayerNorm(width, eps=1e-12)
        self.mlp = TextMlp(width)

    def __call__(self, values: mx.array) -> mx.array:
        """Apply causal attention and MLP residuals to token features."""
        norm = self.ln_1(values)
        values = values + self.attn(norm, norm, norm, "causal")
        return values + self.mlp(self.ln_2(values))


class LanguageTransformer(nn.Module):
    """Encode fixed-context token IDs with causal self-attention.

    Input is ``(prompts, text_context)`` and output is
    ``(prompts, text_context, hidden_dim)``; position embeddings cover the
    configured full context.
    """

    def __init__(self, config: ModelConfig) -> None:
        """Create token and position embeddings, causal blocks and final norm."""
        super().__init__()
        width = config.hidden_dim
        self.token_embedding = nn.Embedding(config.text_vocabulary, width)
        self.positional_embedding = mx.zeros((config.text_context, width))
        self.resblocks = [
            ResidualAttentionBlock(width, config.heads) for _ in range(config.text_layers)
        ]
        self.ln_final = nn.LayerNorm(width, eps=1e-12)

    def __call__(self, token_ids: mx.array) -> mx.array:
        """Encode fixed-length token IDs into normalized hidden states."""
        values = self.token_embedding(token_ids) + self.positional_embedding
        for block in self.resblocks:
            values = block(values)
        return self.ln_final(values)


class LanguageEncoder(nn.Module):
    """Project each causal-transformer token to the decoder's hidden width."""

    def __init__(self, config: ModelConfig) -> None:
        """Create the language transformer and projection to decoder width."""
        super().__init__()
        self.encoder_transformer = LanguageTransformer(config)
        self.lang_proj = mx.zeros((config.hidden_dim, config.hidden_dim))
        self.logit_scale = mx.ones(())

    def __call__(self, token_ids: mx.array) -> mx.array:
        """Project each encoded token into the decoder feature space."""
        return self.encoder_transformer(token_ids) @ self.lang_proj


class TokenEncoding(Protocol):
    """Tokenizer output for one prompt: token IDs plus the active-position mask."""

    @property
    def ids(self) -> list[int]:
        """Return the tokenizer-produced token IDs."""
        ...

    @property
    def attention_mask(self) -> list[int]:
        """Return token positions marked active by the tokenizer."""
        ...


class LocalTokenizer(Protocol):
    """Structural subset of ``tokenizers.Tokenizer`` used for local CLIP vocabularies."""

    def token_to_id(self, token: str) -> int | None:
        """Return a token ID, or None when the vocabulary lacks it."""
        ...

    def enable_truncation(self, max_length: int) -> None:
        """Limit encoded sequences to the requested maximum length."""
        ...

    def enable_padding(self, *, length: int, pad_id: int, pad_token: str) -> None:
        """Pad encoded sequences to the specified length and token."""
        ...

    def encode_batch(self, input: list[str]) -> list[TokenEncoding]:
        """Encode a batch of text prompts."""
        ...


def tokenize(
    prompts: tuple[Prompt, ...], tokenizer_path: Path, config: ModelConfig
) -> tuple[
    mx.array,
    tuple[int, ...],
]:
    """Tokenize each prompt with a local CLIP ``tokenizer.json``; downloads are never implicit.

    Returns int32 ids of shape ``(len(prompts), text_context)``, truncated to the
    context length and padded with the end-of-text id, plus each prompt's token
    count (attention-mask sum, so it includes any special tokens the tokenizer adds).
    Only the end-of-text token ID is checked against the configured vocabulary;
    this does not establish that the entire tokenizer matches the checkpoint.

    Raises:
        FileNotFoundError: ``tokenizer_path`` is not a file.
        ValueError: The end-of-text ID differs from the expected value, or the
            tokenizer returns an unexpected sequence length.
    """
    from tokenizers import Tokenizer

    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"Local CLIP tokenizer.json not found: {tokenizer_path}")
    # tokenizers 0.22 leaves its Rust factory return type unannotated.
    load_tokenizer = cast(Callable[[str], LocalTokenizer], callable_member(Tokenizer, "from_file"))
    tokenizer = load_tokenizer(str(tokenizer_path))
    eos = tokenizer.token_to_id("<|endoftext|>")
    if eos is None or eos != config.text_vocabulary - 1:
        raise ValueError("Tokenizer vocabulary does not match BioMedParse's CLIP encoder")
    tokenizer.enable_truncation(max_length=config.text_context)
    tokenizer.enable_padding(length=config.text_context, pad_id=eos, pad_token="<|endoftext|>")
    encoded = tokenizer.encode_batch([p.text for p in prompts])
    ids = np.array([e.ids for e in encoded], dtype=np.int32)
    if ids.shape != (len(prompts), config.text_context):
        raise ValueError("Tokenizer returned an unexpected sequence length")
    return from_numpy(ids), tuple(sum(e.attention_mask) for e in encoded)


def encode_prompts(
    encoder: LanguageEncoder,
    prompts: tuple[Prompt, ...],
    tokenizer_path: Path,
    config: ModelConfig,
) -> PromptFeatures:
    """Encode prompts into projected language tokens for the decoder.

    The returned ``tokens`` are ``(len(prompts), longest_prompt_tokens, hidden_dim)``
    float32; positions past each prompt's own token count are zeroed.

    Raises:
        ValueError: No prompts, or duplicate labels (also see :func:`tokenize`).
    """
    if not prompts or len({p.label for p in prompts}) != len(prompts):
        raise ValueError("Supply at least one prompt, with distinct labels")
    ids, lengths = tokenize(prompts, tokenizer_path, config)
    encoded = encoder(ids)
    length = max(lengths)
    active = mx.arange(length)[None, :] < mx.array(lengths)[:, None]
    tokens = mx.where(active[..., None], encoded[:, :length], 0)
    evaluate(tokens)
    return PromptFeatures(prompts, tokens)
