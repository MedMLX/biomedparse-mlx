"""Runner argument rejection before the automatic weights download."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import cast

import pytest
from medmlx_core.errors import InvalidInputError

from biomedparse_mlx.medmlx import run


@pytest.mark.parametrize(
    ("image", "threshold", "overwrite", "existing", "error"),
    [
        ("image.txt", 0.5, False, False, InvalidInputError),
        ("image.png", 1.5, False, False, InvalidInputError),
        ("image.png", 0.5, "false", False, InvalidInputError),
        ("image.png", 0.5, False, True, FileExistsError),
    ],
    ids=["suffix", "range", "boolean", "existing output"],
)
def test_run_checks_arguments_before_downloading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    image: str,
    threshold: float,
    overwrite: object,
    existing: bool,
    error: type[Exception],
) -> None:
    """Refuse a bad argument without starting the automatic weights download."""

    def download(model_id: str) -> Path:
        raise AssertionError(f"downloaded {model_id} before checking arguments")

    monkeypatch.setattr(import_module("medmlx_core.weights"), "weights_dir", download)
    for name in ("image.txt", "image.png"):
        (tmp_path / name).write_bytes(b"unread")
    output_dir = tmp_path / "out"
    if existing:
        output_dir.mkdir()
        (output_dir / "segmentation.npz").write_bytes(b"intact")
    with pytest.raises(error):
        run(
            image_path=str(tmp_path / image),
            prompt="liver",
            output_dir=str(output_dir),
            existence_threshold=threshold,
            # JSON callers can send a string; the runner must reject it.
            overwrite=cast(bool, overwrite),
        )
