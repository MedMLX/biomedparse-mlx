"""Typed array input boundary and reversible upstream volume geometry."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from biomedparse_mlx.ontology import Volume, VolumeGeometry
from biomedparse_mlx.preprocessing import Modality, normalize_volume_slices


def load_volume(path: Path) -> Volume:
    """Load a preprocessed ``.npy`` volume or an upstream ``.npz`` with an ``imgs`` array.

    The array must already be modality-preprocessed intensities in 0..255; nothing
    is windowed here. Pickled content is never loaded (so an upstream ``.npz``'s
    ``text_prompts`` are ignored).

    Raises:
        FileNotFoundError: ``path`` is not a file.
        ValueError: Wrong suffix, no ``imgs`` array, or the data fails ``Volume``
            validation (3D, finite, 0..255).
    """
    if not path.is_file():
        raise FileNotFoundError(f"Input volume not found: {path}")
    if path.suffix == ".npy":
        return Volume(np.load(path, mmap_mode="r", allow_pickle=False))
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            # Read only numeric imgs; upstream's pickled text_prompts are not loaded.
            if "imgs" not in archive:
                raise ValueError("Upstream NPZ volume must contain the numeric 'imgs' array")
            return Volume(archive["imgs"])
    raise ValueError("Input must be .npy or upstream .npz with numeric imgs")


def axis_from_shape(shape: tuple[int, int, int]) -> int:
    """Upstream's ``get_axis``: the slicing axis, chosen from the array shape alone.

    Axis 0 when axes 1 and 2 differ by less than 50% of their mean (they are taken
    as the in-plane pair); otherwise the shortest axis.
    """
    ratio = 2 * abs(shape[1] - shape[2]) / (shape[1] + shape[2])
    return 0 if ratio < 0.5 else int(np.argmin(shape))


def load_nifti_volume(
    path: Path, *, modality: Modality, ct_site: str | None
) -> tuple[Volume, NDArray[np.float64]]:
    """Load a raw-intensity NIfTI volume and apply upstream's per-slice windowing.

    Upstream never reorients by affine for this 3D pipeline (only its separate
    single-slice ``read_nifti`` utility takes an explicit ``HW_index``); the
    slicing axis is chosen the same way as the rest of this package, by shape.
    Returns the modality-normalized ``Volume`` (float32 in 0..255, same array
    shape and axis order as the file) plus the source affine expressed in mm, so a
    caller can write labels on the same physical grid with ``save_nifti_labels``.
    Unspecified spatial units are assumed to be mm; metres and microns are converted.

    Raises:
        FileNotFoundError: ``path`` is not a file.
        ValueError: The image or its units are invalid, voxels are complex or
            non-finite, the affine is non-finite or singular, or windowing fails
            (see :func:`normalize_intensity_slice`).
    """
    from biomedparse_mlx._nifti import ImageFileError, nib

    if not path.is_file():
        raise FileNotFoundError(f"Input NIfTI not found: {path}")
    try:
        image = nib.load(path.as_posix())
    except (OSError, ImageFileError) as error:
        raise ValueError(f"Invalid input NIfTI: {path}") from error
    # NIfTI affine values use the header's spatial unit, including translation.
    spatial_unit = int(image.header["xyzt_units"]) & 0x07
    units_to_mm = {0: 1.0, 1: 1000.0, 2: 1.0, 3: 0.001}
    if spatial_unit not in units_to_mm:
        raise ValueError("Input NIfTI has invalid spatial units")
    # NiBabel applies scl_slope/scl_inter here; unscaled data keeps its stored
    # dtype (for example int16 CT) instead of a full-volume float64 copy. Each
    # windowed slice is converted to float64 separately, which gives the same
    # values as converting the whole volume first.
    data = np.asanyarray(image.dataobj)
    if data.dtype.kind not in "biuf":
        raise ValueError("Input NIfTI must contain real intensities")
    # Reject empty axes here: axis_from_shape and per-slice windowing run before
    # Volume's own nonempty check.
    if data.ndim != 3 or min(data.shape) < 1:
        raise ValueError(
            f"Input NIfTI must be a single nonempty 3D volume; got shape {data.shape}"
        )
    if not _finite_as_float64(data):
        raise ValueError("Input NIfTI contains non-finite voxels")
    affine = np.array(image.affine, dtype=np.float64, copy=True)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError("Input NIfTI affine must be a finite 4x4 matrix")
    affine[:3, :] *= units_to_mm[spatial_unit]
    if not np.isfinite(affine).all() or np.linalg.matrix_rank(affine[:3, :3]) < 3:
        raise ValueError("Input NIfTI affine must define an invertible grid in mm")
    axis = axis_from_shape((data.shape[0], data.shape[1], data.shape[2]))
    # Booleans also convert to float64 0/1 slices, as before.
    real = cast(NDArray[np.floating | np.integer], data)
    normalized = normalize_volume_slices(real, axis, modality=modality, ct_site=ct_site)
    return Volume(normalized), affine


def _finite_as_float64(data: NDArray[np.generic]) -> bool:
    """Whether every voxel stays finite after conversion to float64.

    Integers and floats up to 64 bits convert finite values to finite float64, so
    their own ``isfinite`` decides. Wider floats (``longdouble``) can overflow
    to infinity in float64, so they are checked one converted slice at a time.
    """
    if data.dtype.kind in "biu":
        return True
    if data.dtype.itemsize <= 8:
        return bool(np.isfinite(data).all())
    return all(np.isfinite(plane.astype(np.float64)).all() for plane in data)


def save_nifti_labels(labels: NDArray[np.uint16], affine: NDArray[np.float64], path: Path) -> None:
    """Write segmentation labels back on the source NIfTI grid.

    The supplied affine is in mm. Upstream writes no NIfTI; this preserves the
    input's physical geometry. Both sform and qform are written with code 1;
    sform retains the supplied affine within NIfTI precision, while qform cannot
    represent shear and NiBabel removes it. Labels are written as uint16 (NIfTI-1
    datatype code 512), matching ``Prompt.label``'s full 1..65535 range.
    """
    from biomedparse_mlx._nifti import nib

    image = nib.Nifti1Image(labels.astype(np.uint16), affine)
    image.header.set_xyzt_units("mm")
    image.set_sform(affine, code=1)
    image.set_qform(affine, code=1)
    nib.save(image, path.as_posix())


def square_padding(height: int, width: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Upstream's ``get_padding``: pad a ``height`` by ``width`` plane to a square.

    Returns ``((top, bottom), (left, right))``; the smaller half goes first and the
    remainder second.
    """
    difference = abs(height - width)
    padding = (difference // 2, difference - difference // 2)
    return ((0, 0), padding) if height > width else (padding, (0, 0))


def volume_geometry(volume: Volume) -> VolumeGeometry:
    """Slicing axis, in-plane shape and square padding for a volume's shape."""
    shape = volume.data.shape
    axis = axis_from_shape(shape)
    plane = tuple(shape[i] for i in range(3) if i != axis)
    height, width = plane
    pads = square_padding(height, width)
    return VolumeGeometry(
        axis, (shape[0], shape[1], shape[2]), (height, width), pads, max(height, width)
    )


def neighboring_batch(
    volume: Volume,
    geometry: VolumeGeometry,
    start: int,
    stop: int,
) -> NDArray[np.generic]:
    """Pack slices ``start:stop`` with their neighbors into a square-padded batch.

    Returns an array of shape ``(stop - start, S, S, 3)``, with ``S`` the padded
    square side (``geometry.square_size``) and channels ordered current, previous,
    next along the slicing axis. The first slice's previous neighbor is slice 1 and
    the last slice's next neighbor is the second-to-last (reflection, as upstream).
    A single-slice volume repeats that slice. Padding is constant zero.
    """
    slices = np.moveaxis(volume.data, geometry.axis, 0)
    depth = len(slices)
    current = np.arange(start, stop)
    previous = np.where(current == 0, min(1, depth - 1), current - 1)
    following = np.where(current == depth - 1, max(depth - 2, 0), current + 1)
    packed = slices[np.stack((current, previous, following), axis=1)].transpose(0, 2, 3, 1)
    return np.pad(packed, ((0, 0), *geometry.padding, (0, 0)), mode="constant")


def restore_labels(labels: NDArray[np.uint16], geometry: VolumeGeometry) -> NDArray[np.uint16]:
    """Map ``(depth, S', S')`` labels on the model grid back to the source array.

    Nearest-neighbor resize to the padded square side (Torch's legacy ``nearest``,
    as upstream's ``process_output``), remove the padding, and move the slice axis
    back, returning the original ``geometry.original_shape``. Only the
    unpadded rows and columns are gathered.
    """
    from biomedparse_mlx.resize import nearest_indices

    source_h, source_w = labels.shape[-2:]
    target = geometry.square_size
    rows = nearest_indices(source_h, target)
    columns = nearest_indices(source_w, target)
    (top, _), (left, _) = geometry.padding
    height, width = geometry.plane_shape
    # Gather only the unpadded window of the nearest-resized square; the values
    # equal resizing the whole square and cropping it afterwards.
    rows, columns = rows[top : top + height], columns[left : left + width]
    restored = labels[:, rows[:, None], columns]
    return np.moveaxis(restored, 0, geometry.axis)
