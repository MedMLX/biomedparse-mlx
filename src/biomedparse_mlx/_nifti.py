"""Verified NiBabel NIfTI interface (same typed-protocol pattern as sibling MedMLX ports)."""

from __future__ import annotations

from os import PathLike
from typing import Protocol, cast

import nibabel
import numpy as np
from nibabel.filebasedimages import ImageFileError as ImageFileError
from numpy.typing import NDArray

type PathArg = str | PathLike[str]


class ArrayBuffer(Protocol):
    def __array__(
        self, dtype: np.dtype[np.generic] | None = None, copy: bool | None = None
    ) -> NDArray[np.generic]:
        """Expose the image data as a NumPy array, honoring dtype and copy options."""
        ...


class NiftiHeader(Protocol):
    def get_zooms(self) -> tuple[float, ...]:
        """Return the voxel spacing recorded in the image header."""
        ...

    def get_xyzt_units(self) -> tuple[str, str]:
        """Return spatial and temporal unit names from the header."""
        ...

    def set_xyzt_units(self, xyz: str) -> None:
        """Set the spatial unit name in the header."""
        ...

    def __getitem__(self, key: str) -> NDArray[np.generic]:
        """Read a named header field."""
        ...

    def __setitem__(self, key: str, value: int) -> None:
        """Set an integer-valued header field."""
        ...


class NiftiImage(Protocol):
    shape: tuple[int, ...]
    affine: NDArray[np.float64]
    dataobj: ArrayBuffer | NDArray[np.generic]
    header: NiftiHeader

    def __init__(self, dataobj: NDArray[np.generic], affine: NDArray[np.float64]) -> None:
        """Construct a NIfTI image from an array and its affine."""
        ...

    def set_sform(self, affine: NDArray[np.float64], code: int | None = None) -> None:
        """Set the spatial sform affine and its optional code."""
        ...

    def set_qform(self, affine: NDArray[np.float64], code: int | None = None) -> None:
        """Set the spatial qform affine and its optional code."""
        ...


class Nibabel(Protocol):
    Nifti1Image: type[NiftiImage]

    def load(self, filename: PathArg) -> NiftiImage:
        """Load an image from a NIfTI path."""
        ...

    def save(self, img: NiftiImage, filename: PathArg) -> None:
        """Write a NIfTI image to the requested path."""
        ...


# Bound to the real module; this runner admits only NIfTI images.
nib = cast(Nibabel, nibabel)
