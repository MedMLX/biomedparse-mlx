"""Runner argument rejection before the automatic weights download."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import cast

import pytest
from medmlx_core.errors import InvalidInputError
from medmlx_core.outputs import validate_outputs

from biomedparse_mlx.medmlx import output_artifacts, run
from biomedparse_mlx.ontology import Prompt


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


def test_output_artifacts_name_each_written_file() -> None:
    """Each output role is described: prompt labels on masks, source grid on NIfTI."""
    prompts = (Prompt(3, "liver"), Prompt(5, "spleen"))
    outputs = {
        "mask": "segmentation.npz",
        "receipt": "run.json",
        "mask_nifti": "segmentation.nii.gz",
        "instances_nifti": "instances.nii.gz",
    }
    artifacts = output_artifacts(outputs, prompts, is_nifti=True, is_2d=False)
    _, checked = validate_outputs({"outputs": outputs, "artifacts": artifacts})
    labels = [{"value": 3, "name": "liver"}, {"value": 5, "name": "spleen"}]
    assert checked["mask_nifti"]["kind"] == "label_map"
    assert checked["mask_nifti"].get("labels") == labels
    assert checked["mask_nifti"].get("grid_of") == "image_path"
    assert checked["mask_nifti"]["coordinate_space"] == "image_physical"
    assert checked["mask"].get("grid_of") is None
    assert checked["instances_nifti"].get("labels") == []
    two_d = output_artifacts({"mask": "m", "receipt": "r"}, prompts, is_nifti=False, is_2d=True)
    assert two_d["mask"]["kind"] == "binary_mask"
    assert two_d["mask"].get("grid_of") == "image_path"
