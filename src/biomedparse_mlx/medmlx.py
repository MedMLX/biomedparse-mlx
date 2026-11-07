"""Host-safe MedMLX declaration; the CLI, Python API and MedMLX call the same inference action."""

from __future__ import annotations

import json
from dataclasses import asdict
from itertools import combinations
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, cast
from uuid import uuid4

from biomedparse_mlx.ontology import (
    DICOM_SUFFIXES,
    IMAGE_SUFFIXES,
    NIFTI_SUFFIXES,
    Prompt,
    RunReceipt,
    model_config,
)

if TYPE_CHECKING:
    from biomedparse_mlx.preprocessing import Modality


MODEL = {
    "id": model_config().id,
    "label": "BioMedParse v2",
    "task": "segmentation",
    "run": "biomedparse_mlx.medmlx:run",
    # Measured MLX peak for four prompts on a 768x768x101 abdominal CT; larger inputs need more.
    "peak_memory_gb": 3.9,
    "summary": "Text-guided 3D segmentation with native MLX and slice existence classification.",
}


def _prompt_texts(value: object) -> list[str]:
    """Accept one prompt string or a nonempty list containing only prompt strings."""
    from medmlx_core.errors import InvalidInputError

    if isinstance(value, str):
        return [value]
    if not isinstance(value, list) or not value:
        raise InvalidInputError("prompt must be a string or a nonempty list of strings")
    items = cast(list[object], value)
    if not all(isinstance(item, str) for item in items):
        raise InvalidInputError("prompt must be a string or a nonempty list of strings")
    return cast(list[str], value)


def _parse_modality(modality: str | None, ct_site: str | None) -> tuple[Modality, str | None]:
    """Validate the raw-input windowing options; ``modality`` is required here."""
    from medmlx_core.errors import InvalidInputError

    from biomedparse_mlx.preprocessing import CT_WINDOWS, Modality

    if modality is None:
        raise InvalidInputError(
            'modality is required for raw NIfTI or DICOM input; pass modality="ct" or "other"'
        )
    try:
        parsed = Modality(modality)
    except ValueError as error:
        raise InvalidInputError('modality must be "ct" or "other"') from error
    if parsed is not Modality.CT and ct_site is not None:
        raise InvalidInputError('ct_site only applies when modality is "ct"')
    if parsed is Modality.CT and ct_site not in CT_WINDOWS:
        raise InvalidInputError(f'ct_site must be one of {sorted(CT_WINDOWS)} for modality="ct"')
    return parsed, ct_site


def _assets(bundle_dir: Path, tokenizer_path: str | None) -> tuple[Path, Path]:
    """Return the weights and tokenizer files for a model folder or weights file."""
    # Keep the named snapshot link; MLX loads safetensors explicitly.
    weights = (
        bundle_dir / "biomedparse-v2.safetensors" if bundle_dir.is_dir() else bundle_dir
    ).absolute()
    tokenizer = (
        weights.parent / "tokenizer.json"
        if tokenizer_path is None
        else Path(tokenizer_path).expanduser()
    )
    return weights, tokenizer


def _refuse_aliases(outputs: set[Path], assets: list[Path]) -> None:
    """Refuse outputs that would replace an input asset or each other."""
    from medmlx_core.errors import InvalidInputError

    if any(
        output.resolve() == asset.resolve()
        or (output.exists() and asset.exists() and output.samefile(asset))
        for output in outputs
        for asset in assets
    ):
        raise InvalidInputError("Inference output would overwrite an input asset")
    if any(
        first.resolve() == second.resolve()
        or (first.exists() and second.exists() and first.samefile(second))
        for first, second in combinations(outputs, 2)
    ):
        raise InvalidInputError("Inference outputs would refer to the same file")


def run(
    *,
    image_path: str,
    prompt: str | list[str],
    weights_path: str | None = None,
    tokenizer_path: str | None = None,
    output_dir: str,
    labels: list[int] | None = None,
    slice_batch_size: int = 2,
    prompt_batch_size: int | None = None,
    existence_threshold: float = 0.5,
    seed: int = 0,
    modality: str | None = None,
    ct_site: str | None = None,
    overwrite: bool = False,
    instance_segmentation: bool = False,
) -> dict[str, object]:
    """Segment a volume or a single 2D image; write a versioned receipt alongside it.

    ``image_path`` dispatches on extension: ``.npy``/``.npz`` are already
    upstream-preprocessed numeric volumes (unchanged contract); ``.nii``/``.nii.gz``
    are raw-intensity volumes windowed with upstream's own CT/other normalization,
    selected by ``modality``/``ct_site``, and also written back as NIfTI with the
    source affine. ``.png``/``.jpg``/``.jpeg``/``.bmp``/``.tif``/``.tiff``/``.dcm``
    run upstream's separate 2D inference path (per-prompt overlapping masks, no
    cross-prompt merge); ``.dcm`` additionally requires ``modality``/``ct_site``.
    ``instance_segmentation`` opts into upstream's example-notebook edge-refined
    instance split for volume (non-2D) input only.

    Args:
        image_path: Input volume or image (see above).
        prompt: One text prompt, or a list of them; each is tokenized on its own.
        weights_path: Model folder containing ``biomedparse-v2.safetensors`` and
            ``tokenizer.json``. Omit to download the cached MedMLX model automatically.
            An explicit ``.safetensors`` file is also accepted.
        tokenizer_path: Optional explicit tokenizer file; defaults to the model folder.
        output_dir: Directory for the outputs; created if missing.
        labels: Positive integer label per prompt (1..65535, distinct), written into
            the volume output label array. Defaults to 1, 2, ... in prompt order.
        slice_batch_size: Slices per model call for volumes; ignored for 2D input.
        prompt_batch_size: Maximum prompts per decoder group, a positive integer;
            None decodes all prompts together. Grouping bounds repeated features and
            decoder activations, not the full output or random-sampling fields.
        existence_threshold: Cutoff in [0, 1] on the per-slice (or per-image)
            existence probability of a prompt.
        seed: Non-negative seed for MLX's generator, which drives the attention
            sampling; the same seed reproduces a run only on the same backend,
            prompt order, input shape and batch settings. Grouped and ungrouped
            decoding can differ numerically even when random draws are preserved.
        modality: ``"ct"`` or ``"other"``; required for raw NIfTI or DICOM input and
            rejected otherwise.
        ct_site: CT window name (abdomen, lung, pelvis, liver, colon or pancreas);
            required for ``modality="ct"``, rejected otherwise.
        overwrite: Replace existing outputs instead of raising.
        instance_segmentation: Also write an ``instances`` array (and, for NIfTI,
            ``instances.nii.gz``).

    Returns:
        ``{"outputs": ..., "details": ...}``. ``outputs`` maps ``mask`` to
        ``segmentation.npz`` and ``receipt`` to ``run.json``, plus ``mask_nifti``
        (``segmentation.nii.gz``) and ``instances_nifti`` when written. The volume
        ``segmentation.npz`` holds ``segs`` (uint16 labels on the input grid, 0
        for background), ``existence`` (prompts by slices) and optionally
        ``instances``; the 2D one holds ``masks``, ``probabilities`` and
        ``existence`` per prompt, plus the prompt texts. ``details`` is the run receipt.

    Every argument, the input file and the outputs are checked before the
    automatic weights download; aliases of downloaded files are checked after it.
    Paths are not locked against concurrent filesystem changes after that check.

    Raises:
        FileExistsError: An output exists and ``overwrite`` is false.
        InvalidInputError: ``image_path`` is missing, outputs alias an input or each
            other, the input suffix or modality/ct_site/instance options do not fit,
            or an argument is invalid (a ``ValueError``).
        FileNotFoundError: A missing weights or tokenizer file.
    """
    from medmlx_core.errors import InvalidInputError
    from medmlx_core.weights import weights_dir, weights_revision

    # Check every argument before downloading weights or loading MLX.
    for name, flag in (("overwrite", overwrite), ("instance_segmentation", instance_segmentation)):
        if not isinstance(cast(object, flag), bool):  # JSON callers can send "false"
            raise InvalidInputError(f"{name} must be true or false")
    if prompt_batch_size is not None and (
        type(prompt_batch_size) is not int or prompt_batch_size < 1
    ):
        raise InvalidInputError("prompt_batch_size must be a positive integer or null")
    if type(slice_batch_size) is not int or slice_batch_size < 1:
        raise InvalidInputError("slice_batch_size must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise InvalidInputError("seed must be a non-negative integer")
    if type(existence_threshold) not in (int, float) or not 0 <= existence_threshold <= 1:
        raise InvalidInputError("existence_threshold must be a number in [0, 1]")
    texts = _prompt_texts(prompt)
    if labels is not None and (
        not isinstance(cast(object, labels), list) or len(labels) != len(texts)
    ):
        raise InvalidInputError("labels must be a list with one label per prompt")
    ids = list(range(1, len(texts) + 1)) if labels is None else labels
    try:
        prompts = tuple(Prompt(label, text) for label, text in zip(ids, texts, strict=True))
    except ValueError as error:
        raise InvalidInputError(str(error)) from error
    if len({item.label for item in prompts}) != len(prompts):
        raise InvalidInputError("labels must be distinct")

    source = Path(image_path).expanduser().resolve()
    directory = Path(output_dir).expanduser().resolve()
    output = directory / "segmentation.npz"
    receipt_path = directory / "run.json"
    name = source.name.lower()
    is_nifti = name.endswith(NIFTI_SUFFIXES)
    suffix = source.suffix.lower()
    is_dicom = suffix in DICOM_SUFFIXES
    is_2d = suffix in IMAGE_SUFFIXES or is_dicom
    nifti_output = directory / "segmentation.nii.gz" if is_nifti else None
    write_instances = is_nifti and instance_segmentation
    instances_output = directory / "instances.nii.gz" if write_instances else None
    if (modality is not None or ct_site is not None) and not (is_nifti or is_dicom):
        raise InvalidInputError("modality/ct_site only apply to raw NIfTI or DICOM input")
    if not (is_nifti or is_2d) and source.suffix not in (".npy", ".npz"):
        raise InvalidInputError("Input must be .npy, .npz, .nii[.gz], or a 2D image/DICOM file")
    if instance_segmentation and is_2d:
        raise InvalidInputError("instance_segmentation only applies to volume (non-2D) input")
    windowing = _parse_modality(modality, ct_site) if is_nifti or is_dicom else None
    if directory.exists() and not directory.is_dir():
        raise InvalidInputError(f"output_dir is not a directory: {directory}")
    reserved: set[Path] = {output, receipt_path}
    if nifti_output is not None:
        reserved.add(nifti_output)
    if instances_output is not None:
        reserved.add(instances_output)
    if not overwrite and any(path.exists() for path in reserved):
        raise FileExistsError(f"Output exists in {directory}; pass overwrite=true to replace it")
    known = [source]
    if weights_path is not None:
        known.extend(_assets(Path(weights_path).expanduser(), tokenizer_path))
    elif tokenizer_path is not None:
        known.append(Path(tokenizer_path).expanduser())
    _refuse_aliases(reserved, known)
    if not source.is_file():
        raise InvalidInputError(f"image_path does not exist: {source}")

    bundle_dir = (
        weights_dir("biomedparse-v2") if weights_path is None else Path(weights_path).expanduser()
    )
    weights, tokenizer = _assets(bundle_dir, tokenizer_path)
    if weights_path is None:
        # The downloaded folder is known only now; recheck it against the outputs.
        _refuse_aliases(reserved, [weights, tokenizer])

    import numpy as np

    started = perf_counter()
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / f".{uuid4().hex}.npz"
    try:
        if is_2d:
            from biomedparse_mlx.image2d import load_2d_image, load_dicom_slice, segment_image

            if windowing is not None:
                image = load_dicom_slice(source, modality=windowing[0], ct_site=windowing[1])
            else:
                image = load_2d_image(source)
            image_result = segment_image(
                image,
                prompts,
                weights,
                tokenizer,
                existence_threshold=existence_threshold,
                seed=seed,
                prompt_batch_size=prompt_batch_size,
            )
            np.savez_compressed(
                temporary,
                masks=image_result.masks,
                probabilities=image_result.probabilities,
                existence=image_result.existence_probabilities,
                prompts=np.array([p.text for p in prompts]),
                allow_pickle=False,
            )
            receipt = RunReceipt(
                model_config().id,
                model_config().source_revision,
                prompts,
                image_result.backend.value,
                "float32",
                image_result.masks.shape,
                slice_batch_size,
                seed,
                existence_threshold,
                perf_counter() - started,
                prompt_batch_size=prompt_batch_size,
            )
        else:
            from biomedparse_mlx.pipeline import segment_volume
            from biomedparse_mlx.volume import load_nifti_volume, load_volume, save_nifti_labels

            if windowing is not None:
                volume, affine = load_nifti_volume(
                    source, modality=windowing[0], ct_site=windowing[1]
                )
            else:
                volume, affine = load_volume(source), None
            result = segment_volume(
                volume,
                prompts,
                weights,
                tokenizer,
                slice_batch_size=slice_batch_size,
                prompt_batch_size=prompt_batch_size,
                existence_threshold=existence_threshold,
                seed=seed,
                instance_segmentation=instance_segmentation,
            )
            if result.instances is None:
                np.savez_compressed(
                    temporary,
                    segs=result.labels,
                    existence=result.existence_probabilities,
                    allow_pickle=False,
                )
            else:
                np.savez_compressed(
                    temporary,
                    segs=result.labels,
                    existence=result.existence_probabilities,
                    instances=result.instances,
                    allow_pickle=False,
                )
            receipt = RunReceipt(
                model_config().id,
                model_config().source_revision,
                prompts,
                result.backend.value,
                "float32",
                result.labels.shape,
                slice_batch_size,
                seed,
                existence_threshold,
                perf_counter() - started,
                instance_segmentation,
                prompt_batch_size=prompt_batch_size,
            )
            if affine is not None and nifti_output is not None:
                save_nifti_labels(result.labels, affine, nifti_output)
            if affine is not None and instances_output is not None and result.instances is not None:
                save_nifti_labels(result.instances, affine, instances_output)
        temporary.replace(output)
        receipt_path.write_text(json.dumps(asdict(receipt), indent=2) + "\n")
    finally:
        temporary.unlink(missing_ok=True)
    outputs = {"mask": str(output), "receipt": str(receipt_path)}
    if nifti_output is not None:
        outputs["mask_nifti"] = str(nifti_output)
    if instances_output is not None:
        outputs["instances_nifti"] = str(instances_output)
    details = asdict(receipt)
    details["weights_revision"] = weights_revision(weights.parent)
    return {"outputs": outputs, "details": details}
