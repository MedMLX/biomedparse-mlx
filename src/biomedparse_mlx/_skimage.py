"""Verified scikit-image interface (same typed-protocol pattern as sibling MedMLX ports)."""

from __future__ import annotations

from typing import Protocol, cast

import numpy as np
import skimage.measure
import skimage.segmentation
from numpy.typing import NDArray


class SkimageMeasure(Protocol):
    def label(
        self, label_image: NDArray[np.integer], *, connectivity: int | None = None
    ) -> NDArray[np.int64]:
        """Label connected regions with the specified connectivity."""
        ...


class SkimageSegmentation(Protocol):
    def expand_labels(
        self, label_image: NDArray[np.integer], *, distance: int = 1
    ) -> NDArray[np.int64]:
        """Expand integer labels into background up to the given distance."""
        ...


# Bound to the real modules; both functions only connect/expand same-valued regions.
measure = cast(SkimageMeasure, skimage.measure)
segmentation = cast(SkimageSegmentation, skimage.segmentation)
