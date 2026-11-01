"""Upstream modality-specific intensity windowing/normalization parity."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from biomedparse_mlx.preprocessing import CT_WINDOWS, Modality, normalize_intensity_slice


class ProcessingUtils(Protocol):
    """The subset of upstream's ``inference_utils/processing_utils.py`` used here.

    Only reachable via ``sys.path`` once the pinned checkout is located, so this
    is imported dynamically (``importlib.import_module``) rather than with a
    static ``from inference_utils.processing_utils import ...`` pyright cannot
    resolve, exactly as ``_nifti.py``/``_dicom.py`` type the installed packages.
    """

    CT_WINDOWS: dict[str, list[float]]

    def process_intensity_image(
        self, image_data: NDArray[np.float64], is_CT: bool, site: str | None = None
    ) -> NDArray[np.uint8]:
        """Run the upstream modality-specific intensity preprocessing."""
        ...

    def read_dicom(
        self, image_path: str, is_CT: bool, site: str | None = None
    ) -> NDArray[np.uint8]:
        """Read and preprocess a DICOM image with the upstream helper."""
        ...


def _processing_utils(upstream: Path) -> ProcessingUtils:
    """Import preprocessing helpers from the selected upstream checkout."""
    import importlib

    sys.path.insert(0, str(upstream))
    return cast(ProcessingUtils, importlib.import_module("inference_utils.processing_utils"))


UPSTREAM = Path(__file__).resolve().parent.parent.parent / "upstream" / "BiomedParse"


@pytest.mark.oracle
def test_ct_windows_match_upstream() -> None:
    """Compare the supported CT window bounds with the pinned upstream checkout."""
    if not UPSTREAM.is_dir():
        pytest.skip("Pinned upstream checkout is required")
    upstream_windows = _processing_utils(UPSTREAM).CT_WINDOWS

    assert set(CT_WINDOWS) == set(upstream_windows)
    for site, bounds in CT_WINDOWS.items():
        assert list(bounds) == upstream_windows[site]


@pytest.mark.oracle
@pytest.mark.parametrize(
    "modality,ct_site",
    [(Modality.CT, "lung"), (Modality.CT, "liver"), (Modality.OTHER, None)],
)
def test_normalize_intensity_slice_matches_upstream_process_intensity_image(
    modality: Modality, ct_site: str | None
) -> None:
    """Upstream's ``process_intensity_image`` pads to square and resizes to 1024
    before stacking channels; this package stops at the windowed 0..255 array and
    resizes to the v2 model's own 512 elsewhere. Using square synthetic data and
    applying upstream's own resize to our output isolates the windowing math."""
    if not UPSTREAM.is_dir():
        pytest.skip("Pinned upstream checkout is required")
    skimage = pytest.importorskip("skimage")
    pytest.importorskip("SimpleITK")
    process_intensity_image = _processing_utils(UPSTREAM).process_intensity_image

    rng = np.random.default_rng(23)
    lower = -1200.0 if modality is Modality.CT else 10.0
    data = rng.uniform(lower, 3000.0, size=(40, 40))
    expected = process_intensity_image(data.copy(), modality is Modality.CT, ct_site)
    native = normalize_intensity_slice(data, modality=modality, ct_site=ct_site)
    # At native resolution (no resize) this package's float32 output already matches
    # upstream's float64 windowing formula to float32 precision.
    if modality is Modality.CT:
        assert ct_site is not None
        lower_bound, upper_bound = CT_WINDOWS[ct_site]
    else:
        lower_bound, upper_bound = (
            np.percentile(data[data > 0], 0.5),
            np.percentile(data[data > 0], 99.5),
        )
    clipped = np.clip(data, lower_bound, upper_bound)
    pre_resize = (clipped - clipped.min()) / (clipped.max() - clipped.min()) * 255.0
    np.testing.assert_allclose(native, pre_resize.astype(np.float32), atol=1e-3, rtol=1e-5)
    # Feeding that same float32 array through upstream's own resize then isolates any
    # residual float32-vs-float64 rounding at the final uint8 cast, which this package
    # never performs in its own pipeline (it resizes to the v2 model's 512 in MLX,
    # oracle-tested separately in test_metal.py): allow the resulting handful of
    # exactly-1-off pixels at interpolated half-integer boundaries.
    resized = np.asarray(
        skimage.transform.resize(
            native, (1024, 1024), order=3, mode="constant", preserve_range=True, anti_aliasing=True
        ),
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(expected[..., 0], expected[..., 1])
    difference = np.abs(resized.astype(np.int16) - expected[..., 0].astype(np.int16))
    assert difference.max() <= 1
    assert float(np.mean(difference == 0, dtype=np.float64)) > 0.999


@pytest.mark.oracle
def test_load_dicom_slice_matches_upstream_read_dicom() -> None:
    """Real-world fixture: upstream's ``read_dicom`` on the shipped CT example."""
    dicom_path = UPSTREAM / "examples" / "CT_lung_nodule.dcm"
    if not dicom_path.is_file():
        pytest.skip("Pinned upstream example DICOM is required")
    skimage = pytest.importorskip("skimage")
    pytest.importorskip("SimpleITK")
    read_dicom = _processing_utils(UPSTREAM).read_dicom
    from biomedparse_mlx._dicom import dicom

    dataset = dicom.dcmread(dicom_path.as_posix())
    intensities = dataset.pixel_array * dataset.RescaleSlope + dataset.RescaleIntercept
    assert intensities.shape[0] == intensities.shape[1]  # square; padding is a no-op

    expected = read_dicom(dicom_path.as_posix(), True, "lung")
    native = normalize_intensity_slice(
        np.asarray(intensities), modality=Modality.CT, ct_site="lung"
    )
    lower_bound, upper_bound = CT_WINDOWS["lung"]
    clipped = np.clip(intensities, lower_bound, upper_bound)
    pre_resize = (clipped - clipped.min()) / (clipped.max() - clipped.min()) * 255.0
    np.testing.assert_allclose(native, pre_resize.astype(np.float32), atol=1e-3, rtol=1e-5)
    resized = np.asarray(
        skimage.transform.resize(
            native, (1024, 1024), order=3, mode="constant", preserve_range=True, anti_aliasing=True
        ),
        dtype=np.uint8,
    )
    difference = np.abs(resized.astype(np.int16) - expected[..., 0].astype(np.int16))
    assert difference.max() <= 1
    assert float(np.mean(difference == 0, dtype=np.float64)) > 0.999
