"""Upstream modality-specific intensity preprocessing.

Ports the windowing/normalization math from upstream's
``inference_utils/processing_utils.py::process_intensity_image``: CT applies a
site-specific Hounsfield window, other modalities use a 0.5-99.5 percentile
window over the positive intensities. Both branches then clip and rescale to
0..255 using the clipped slice's own min/max, following upstream's formula.

Upstream's ``process_intensity_image`` also pads to square and resizes to a
fixed 1024x1024 before stacking into 3 channels; that sizing is specific to
that (unused-by-v2) legacy utility, not to the v2 model, which upstream's own
``inference.py``/``inference_2d.py`` always resize to 512. This module stops
at the 0..255 intensity array so callers can continue through this package's
own v2-consistent padding and resize (``volume.py``, ``image2d.py``).
"""

from __future__ import annotations

from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

CT_WINDOWS: dict[str, tuple[float, float]] = {
    "abdomen": (-150, 250),
    "lung": (-1000, 1000),
    "pelvis": (-55, 200),
    "liver": (-25, 230),
    "colon": (-68, 187),
    "pancreas": (-100, 200),
}


class Modality(StrEnum):
    """Upstream's binary ``is_CT`` switch; every other modality shares one branch."""

    CT = "ct"
    OTHER = "other"


def normalize_intensity_slice(
    data: NDArray[np.floating | np.integer],
    *,
    modality: Modality,
    ct_site: str | None,
) -> NDArray[np.float32]:
    """Clip and rescale one 2D intensity slice to 0..255, following upstream's formula.

    Args:
        data: One 2D slice of raw intensities (HU for CT).
        modality: ``CT`` windows to the Hounsfield range of ``ct_site``; ``OTHER``
            windows to the 0.5th..99.5th percentiles of the positive intensities.
        ct_site: Key of ``CT_WINDOWS``; required for CT and rejected otherwise.

    Returns:
        Float32 array of the slice's shape. Finite, nonconstant clipped data is
        rescaled using its own minimum and maximum to span 0..255. This helper does
        not check input finiteness; non-finite values can propagate into the result.

    Raises:
        ValueError: ``data`` is not 2D, the CT site is missing or unknown,
            ``ct_site`` is given for a non-CT modality, a non-CT slice has no
            positive intensities, or the windowed slice is constant.
    """
    if data.ndim != 2:
        raise ValueError("normalize_intensity_slice requires a single 2D slice")
    values = data.astype(np.float64)
    if modality is Modality.CT:
        if ct_site is None:
            raise ValueError(f"Choose a CT site from {sorted(CT_WINDOWS)}")
        if ct_site not in CT_WINDOWS:
            raise ValueError(f"Unknown CT site {ct_site!r}; choose from {sorted(CT_WINDOWS)}")
        lower, upper = CT_WINDOWS[ct_site]
    else:
        if ct_site is not None:
            raise ValueError("ct_site only applies to CT modality")
        positive = values[values > 0]
        if positive.size == 0:
            raise ValueError("Non-CT slice has no positive intensities to normalize")
        lower, upper = (
            float(np.percentile(positive, 0.5)),
            float(np.percentile(positive, 99.5)),
        )
    clipped = np.clip(values, lower, upper)
    span = float(clipped.max() - clipped.min())
    if span <= 0:
        raise ValueError("Slice has no intensity range after windowing")
    normalized = (clipped - clipped.min()) / span * 255.0
    return normalized.astype(np.float32)


def normalize_volume_slices(
    data: NDArray[np.floating | np.integer],
    axis: int,
    *,
    modality: Modality,
    ct_site: str | None,
) -> NDArray[np.float32]:
    """Apply upstream's per-slice windowing along ``axis``, preserving the volume shape.

    Each slice along ``axis`` is windowed independently (see
    :func:`normalize_intensity_slice`), so percentiles and min/max are per slice.
    """
    if data.ndim != 3:
        raise ValueError("normalize_volume_slices requires one 3D volume")
    moved = np.moveaxis(data, axis, 0)
    normalized = np.stack(
        [
            normalize_intensity_slice(moved[index], modality=modality, ct_site=ct_site)
            for index in range(moved.shape[0])
        ]
    )
    return np.moveaxis(normalized, 0, axis)
