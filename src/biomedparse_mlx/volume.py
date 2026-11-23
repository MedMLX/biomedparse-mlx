"""Typed array input boundary and reversible upstream volume geometry."""

from __future__ import annotations

from pathlib import Path

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
    caller can write labels on the same physical grid with
    ``medmlx_core.volumes.write_volume``. Unspecified spatial units are assumed to
    be mm; meters and microns are converted.

    Raises:
        InvalidInputError: ``path`` is missing, or ``medmlx_core.volumes.read_volume``
            rejects the file, or windowing fails (see
            :func:`normalize_intensity_slice`).
    """
    from medmlx_core.volumes import read_volume

    # Keep the stored dtype (for example int16 CT) instead of a full-volume float64
    # copy: each windowed slice is converted to float64 separately, which gives the
    # same values as converting the whole volume first.
    image = read_volume(path, role="image_path", dtype=None)
    data, affine = image.data, image.affine
    axis = axis_from_shape((data.shape[0], data.shape[1], data.shape[2]))
    normalized = normalize_volume_slices(data, axis, modality=modality, ct_site=ct_site)
    return Volume(normalized), affine


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
