from __future__ import annotations

from dataclasses import replace

import pytest

from biomedparse_mlx.ontology import ModelConfig, model_config


@pytest.fixture
def tiny_config() -> ModelConfig:
    """Provide a small four-stage model configuration for synthetic checks."""
    return replace(
        model_config(),
        image_size=32,
        embed_dim=8,
        depths=(1, 1, 1, 1),
        hidden_dim=32,
        heads=4,
        ffn_dim=64,
        encoder_layers=1,
        decoder_layers=3,
        queries=8,
        edge_queries=2,
        text_layers=2,
        text_context=8,
        text_vocabulary=32,
    )


@pytest.fixture
def metal_runtime() -> None:
    """Require and select the MLX Metal runtime, or skip when unavailable."""
    from medmlx_core.runtime import import_mlx, probe_mlx_runtime

    report = probe_mlx_runtime()
    if not report.available:
        pytest.skip(report.reason or "MLX is unavailable")
    import_mlx()
