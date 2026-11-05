"""Pinned upstream oracle; never downloads source, checkpoint, or tokenizer during tests."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

from numpy.typing import NDArray

if TYPE_CHECKING:
    from torch import Tensor

    from biomedparse_mlx.model import BioMedParse
from unittest.mock import patch

import numpy as np
import pytest

from biomedparse_mlx.ontology import (
    AttentionReplay,
    ModelConfig,
    Prompt,
    PromptFeatures,
    SlicePrediction,
    model_config,
)
from tests.torch_types import (
    ReferenceModel,
    checkpoint_state,
    interpolate,
    manual_seed,
    numpy_bool,
    numpy_float,
    numpy_int,
)
from tests.torch_types import (
    from_numpy as torch_from_numpy,
)

pytestmark = [pytest.mark.oracle, pytest.mark.metal, pytest.mark.usefixtures("metal_runtime")]


def reference_model(config: ModelConfig, config_name: str = "biomedparse_3D") -> ReferenceModel:
    """Build the pinned upstream model with the requested synthetic dimensions."""
    pytest.importorskip("torch")
    pytest.importorskip("hydra")
    import hydra
    import hydra.utils

    upstream = Path("../upstream/BiomedParse").resolve()
    if not upstream.is_dir() or not Path("weights/tokenizer/tokenizer.json").is_file():
        pytest.skip("Pinned upstream checkout and local tokenizer assets are required")
    revision = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True)
    assert revision.strip() == config.source_revision
    sys.path.insert(0, str(upstream))
    width = config.hidden_dim
    overrides = [
        f"backbone.embed_dim={config.embed_dim}",
        f"backbone.depths={list(config.depths)}",
        f"sem_seg_head.pixel_decoder.conv_dim={width}",
        f"sem_seg_head.pixel_decoder.mask_dim={width}",
        f"sem_seg_head.pixel_decoder.transformer_nheads={config.heads}",
        f"sem_seg_head.pixel_decoder.transformer_dim_feedforward={config.ffn_dim}",
        f"sem_seg_head.pixel_decoder.transformer_enc_layers={config.encoder_layers}",
        f"sem_seg_head.predictor.hidden_dim={width}",
        f"sem_seg_head.predictor.in_channels={width}",
        f"sem_seg_head.predictor.dim_proj={width}",
        f"sem_seg_head.predictor.mask_dim={width}",
        f"sem_seg_head.predictor.nheads={config.heads}",
        f"sem_seg_head.predictor.dim_feedforward={config.ffn_dim}",
        f"sem_seg_head.predictor.dec_layers={config.decoder_layers}",
        f"sem_seg_head.predictor.num_queries={config.queries}",
        "sem_seg_head.predictor.language_encoder.tokenizer.pretrained_model_name_or_path=weights/tokenizer",
        f"sem_seg_head.predictor.language_encoder.encoder_transformer.config_encoder.WIDTH={width}",
        f"sem_seg_head.predictor.language_encoder.encoder_transformer.config_encoder.HEADS={config.heads}",
        f"sem_seg_head.predictor.language_encoder.encoder_transformer.config_encoder.LAYERS={config.text_layers}",
        f"sem_seg_head.predictor.language_encoder.encoder_transformer.config_encoder.CONTEXT_LENGTH={config.text_context}",
        f"sem_seg_head.predictor.language_encoder.encoder_transformer.config_encoder.VOCAB_SIZE={config.text_vocabulary}",
        f"sem_seg_head.predictor.language_encoder.lang_projection.data.size=[{width},{width}]",
        f"edge_queries={config.edge_queries}",
    ]
    with hydra.initialize_config_dir(config_dir=str(upstream / "configs/model"), version_base=None):
        cfg = hydra.compose(config_name=config_name, overrides=overrides)
    # This is the model assembled from the revision checked above and these overrides.
    reference = cast(ReferenceModel, hydra.utils.instantiate(cfg, _convert_="object"))
    return reference.eval()


def paired_prediction(
    reference: ReferenceModel,
    native: BioMedParse,
    images: NDArray[np.float32],
    tokens: NDArray[np.float32],
    prompts: tuple[Prompt, ...],
    config: ModelConfig,
) -> tuple[SlicePrediction, tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]]:
    """Run native and reference models with the same recorded attention masks."""
    import torch

    from biomedparse_mlx.mlx_types import evaluate, from_numpy

    tensor = torch_from_numpy(images.transpose(0, 3, 1, 2))
    blocked: list[NDArray[np.bool_]] = []
    original = reference.sem_seg_head.predictor.forward_prediction_heads

    def record(
        output: Tensor,
        mask_features: Tensor,
        attn_mask_target_size: tuple[int, int],
        layer_id: int = -1,
    ) -> tuple[Tensor, Tensor]:
        """Record the reference attention mask while retaining its prediction."""
        masks, attention = original(output, mask_features, attn_mask_target_size, layer_id)
        blocked.append(numpy_bool(attention.detach().clone()))
        return masks, attention

    extra = {
        "num_prompts": torch.tensor([len(prompts)] * len(images)),
        "grounding_tokens": torch_from_numpy(np.tile(tokens, (len(images), 1, 1))).transpose(0, 1),
        "class_emb": torch.zeros((len(images) * len(prompts), config.hidden_dim)),
        "logit_scale": torch.ones(()),
    }
    with (
        torch.no_grad(),
        patch.object(
            reference.sem_seg_head.predictor, "forward_prediction_heads", side_effect=record
        ),
    ):
        features = reference.backbone((tensor - reference.pixel_mean) / reference.pixel_std)
        expected = reference.sem_seg_head(features, extra)
    batch = len(images) * len(prompts)
    replay = AttentionReplay(
        tuple(from_numpy(x.reshape(batch, config.heads, config.queries, -1)) for x in blocked)
    )
    actual = native(
        from_numpy(images), PromptFeatures(prompts, from_numpy(tokens)), sampling=replay
    )
    evaluate(actual.mask_logits, actual.existence_logits, actual.edge_logits)
    cutoff = config.queries - config.edge_queries
    mask = numpy_float(expected["pred_gmasks"][:, :cutoff].mean(dim=1))
    existence = numpy_float(expected["object_existence"][:, 0])
    edge = numpy_float(expected["pred_gmasks"][:, cutoff:].mean(dim=1))
    return actual, (mask, existence, edge)


def test_full_graph_seeded_upstream(tiny_config: ModelConfig) -> None:
    """Compare a seeded small native graph with the matching upstream graph."""
    pytest.importorskip("torch")
    import mlx.core as mx
    import torch

    from biomedparse_mlx.mlx_types import from_numpy
    from biomedparse_mlx.model import build_model

    manual_seed(7)
    reference = reference_model(tiny_config)
    native = build_model(tiny_config)
    tensors: list[tuple[str, mx.array]] = []
    for original, value in reference.state_dict().items():
        if original.startswith("loss_function."):
            continue
        name = original.removeprefix("model.")
        values = numpy_float(value.detach())
        if values.ndim == 4:
            values = values.transpose(0, 2, 3, 1)
        tensors.append((name, from_numpy(np.array(values, dtype=np.float32, order="C"))))
    native.load_weights(tensors, strict=True)
    native.eval()
    rng = np.random.default_rng(9)
    ids = rng.integers(
        0, tiny_config.text_vocabulary, (2, tiny_config.text_context), dtype=np.int32
    )
    with torch.no_grad():
        expected = numpy_float(
            reference.sem_seg_head.predictor.language_encoder.encoder_transformer(
                torch_from_numpy(ids),
            )["last_hidden_state"]
        )
    actual = native.sem_seg_head.predictor.language_encoder.encoder_transformer(from_numpy(ids))
    np.testing.assert_allclose(np.array(actual, dtype=np.float32), expected, atol=2e-5, rtol=2e-5)
    images = rng.uniform(0, 255, (2, 32, 32, 3)).astype(np.float32)
    tokens = rng.normal(size=(2, 3, tiny_config.hidden_dim)).astype(np.float32)
    output, expected_outputs = paired_prediction(
        reference, native, images, tokens, (Prompt(1, "liver"), Prompt(7, "spleen")), tiny_config
    )
    for actual, expected in zip(
        (output.mask_logits, output.existence_logits, output.edge_logits),
        expected_outputs,
        strict=True,
    ):
        np.testing.assert_allclose(
            np.array(actual, dtype=np.float32), expected, atol=3e-4, rtol=3e-4
        )


def test_released_checkpoint_example() -> None:
    """Compare released checkpoint predictions on the upstream CT example."""
    pytest.importorskip("torch")
    import torch

    from biomedparse_mlx.model import load_model
    from biomedparse_mlx.resize import ResizeMode, resize
    from biomedparse_mlx.text import encode_prompts, tokenize

    source = Path("weights/source/biomedparse_v2.ckpt")
    weights = Path("weights/biomedparse-v2.safetensors")
    example = Path("../upstream/BiomedParse/examples/imgs/CT_AMOS_amos_0018.npz")
    if not source.is_file() or not weights.is_file() or not example.is_file():
        pytest.skip("Download the released native weights before qualification")
    config = model_config()
    reference = reference_model(config)
    state = checkpoint_state(source)
    reference.load_state_dict(state, strict=True)
    native = load_model(weights, config)
    prompts = (Prompt(6, "Visualization of the liver in abdominal CT imaging"),)
    tokenizer_path = Path("weights/tokenizer.json")
    ids, lengths = tokenize(prompts, tokenizer_path, config)
    with torch.no_grad():
        language = reference.sem_seg_head.predictor.language_encoder.get_text_token_embeddings(
            [p.text for p in prompts],
            norm=False,
        )
    np.testing.assert_array_equal(
        np.array(ids, dtype=np.int32), numpy_int(language["tokens"]["input_ids"])
    )
    tokens = numpy_float(language["token_emb"][:, : max(lengths)])
    text = encode_prompts(
        native.sem_seg_head.predictor.language_encoder, prompts, tokenizer_path, config
    )
    np.testing.assert_allclose(
        np.array(text.tokens, dtype=np.float32), tokens, atol=3e-4, rtol=3e-4
    )
    with np.load(example, allow_pickle=False) as data:
        slices = data["imgs"][[30, 29, 31]]
    images = slices.transpose(1, 2, 0)[None].astype(np.float32)
    output, expected = paired_prediction(reference, native, images, tokens, prompts, config)
    for actual, ref in zip(
        (output.mask_logits, output.existence_logits, output.edge_logits), expected, strict=True
    ):
        np.testing.assert_allclose(np.array(actual, dtype=np.float32), ref, atol=3e-3, rtol=3e-4)
    actual_up = np.array(
        resize(output.mask_logits[..., None], (512, 512), ResizeMode.BICUBIC, antialias=True),
        dtype=np.float32,
    )[..., 0]
    expected_up = numpy_float(
        interpolate(
            torch_from_numpy(expected[0])[:, None],
            size=(512, 512),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )[:, 0]
    )
    agreement = float(np.mean((actual_up > 0) == (expected_up > 0), dtype=np.float64))
    assert agreement > 0.999
