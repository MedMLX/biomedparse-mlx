"""External HCC-TACE and chest radiograph inputs for imaging tests."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from biomedparse_mlx._nifti import nib

ROOT = Path.home() / ".cache/medmlx/test-data"


def xray(size: tuple[int, int]) -> NDArray[np.uint8]:
    """Resize the sourced chest radiograph to a height, width test grid."""
    with Image.open(ROOT / "chest_xray.jpg") as image:
        return np.asarray(image.convert("L").resize((size[1], size[0])), dtype=np.uint8)


def ct_crop() -> NDArray[np.float32]:
    """Read and window an HCC-TACE CT crop to the model's 0..255 range."""
    ct = np.asarray(nib.load(ROOT / "hcc_063_ct.nii.gz").dataobj, dtype=np.float32)[
        100:104, 210:214, 55:59
    ]
    return (np.clip(ct, -150.0, 250.0) + 150.0) * np.float32(255.0 / 400.0)


def liver_labels() -> NDArray[np.uint16]:
    """Resize three source-aligned liver and tumor slices onto the model grid."""
    ct = nib.load(ROOT / "hcc_063_ct.nii.gz")
    liver = nib.load(ROOT / "hcc_063_liver.nii.gz")
    tumor = nib.load(ROOT / "hcc_063_tumor.nii.gz")
    if not np.allclose(liver.affine, ct.affine) or not np.allclose(tumor.affine, ct.affine):
        raise ValueError("HCC-TACE image and mask affines differ")
    liver_data = np.asarray(liver.dataobj, dtype=np.uint8)
    tumor_data = np.asarray(tumor.dataobj, dtype=np.uint8)
    if liver_data.shape != tumor_data.shape or liver_data.shape != np.asarray(ct.dataobj).shape:
        raise ValueError("HCC-TACE image and mask shapes differ")
    result: list[NDArray[np.uint16]] = []
    for z in (55, 60, 65):
        labels = liver_data[80:208, 160:288, z].copy()
        labels[tumor_data[80:208, 160:288, z] > 0] = 2
        image = Image.fromarray(labels).resize((512, 512), Image.Resampling.NEAREST)
        result.append(np.asarray(image, dtype=np.uint16))
    return np.stack(result)
