"""Verified pydicom interface (same typed-protocol pattern as sibling MedMLX ports)."""

from __future__ import annotations

from typing import Protocol, cast

import numpy as np
import pydicom
from numpy.typing import NDArray


class DicomDataset(Protocol):
    pixel_array: NDArray[np.integer | np.floating]
    RescaleSlope: float
    RescaleIntercept: float


class Pydicom(Protocol):
    def dcmread(self, fp: str) -> DicomDataset:
        """Read a DICOM dataset from the given path."""
        ...


# Bound to the real module; a missing DICOM tag still raises AttributeError at use.
dicom = cast(Pydicom, pydicom)
