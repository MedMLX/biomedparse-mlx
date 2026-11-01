"""Thin CLI projection of the inference actions."""

from __future__ import annotations

import argparse
import json


def main() -> None:
    """Entry point of ``biomedparse-mlx``: ``segment`` and ``segment-2d``.

    Prints the action's result as JSON. An OS, value, runtime or cancellation error
    is reported as ``ErrorType: message`` and exits with status 1. ``segment-2d``
    takes no ``--slice-batch-size`` or ``--instance-segmentation`` option.
    """
    parser = argparse.ArgumentParser(description="BioMedParse v2 native MLX inference")
    commands = parser.add_subparsers(dest="command", required=True)
    infer = commands.add_parser(
        "segment",
        help="Segment a volume: upstream-preprocessed .npy/.npz, or raw .nii/.nii.gz "
        "windowed with --modality/--ct-site",
    )
    infer.add_argument("--image", required=True)
    infer.add_argument("--weights", required=True, help="Native .safetensors weight file")
    infer.add_argument("--tokenizer", required=True)
    infer.add_argument("--output-dir", required=True)
    infer.add_argument("--prompt", action="append", required=True)
    infer.add_argument("--label", type=int, action="append")
    infer.add_argument("--slice-batch-size", type=int, default=2)
    infer.add_argument("--prompt-batch-size", type=int)
    infer.add_argument("--existence-threshold", type=float, default=0.5)
    infer.add_argument("--seed", type=int, default=0)
    infer.add_argument("--modality", choices=("ct", "other"))
    infer.add_argument("--ct-site")
    infer.add_argument("--overwrite", action="store_true")
    infer.add_argument("--instance-segmentation", action="store_true")
    infer2d = commands.add_parser(
        "segment-2d",
        help="Segment a single preprocessed 2D image (PNG/JPEG/BMP/TIFF), or a raw "
        "DICOM slice windowed with --modality/--ct-site",
    )
    infer2d.add_argument("--image", required=True)
    infer2d.add_argument("--weights", required=True, help="Native .safetensors weight file")
    infer2d.add_argument("--tokenizer", required=True)
    infer2d.add_argument("--output-dir", required=True)
    infer2d.add_argument("--prompt", action="append", required=True)
    infer2d.add_argument("--label", type=int, action="append")
    infer2d.add_argument("--existence-threshold", type=float, default=0.5)
    infer2d.add_argument("--seed", type=int, default=0)
    infer2d.add_argument("--prompt-batch-size", type=int)
    infer2d.add_argument("--modality", choices=("ct", "other"))
    infer2d.add_argument("--ct-site")
    infer2d.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        from biomedparse_mlx.medmlx import run

        result = run(
            image_path=args.image,
            weights_path=args.weights,
            tokenizer_path=args.tokenizer,
            output_dir=args.output_dir,
            prompt=args.prompt,
            labels=args.label,
            slice_batch_size=args.slice_batch_size if args.command == "segment" else 2,
            existence_threshold=args.existence_threshold,
            seed=args.seed,
            prompt_batch_size=args.prompt_batch_size,
            modality=args.modality,
            ct_site=args.ct_site,
            overwrite=args.overwrite,
            instance_segmentation=(
                args.instance_segmentation if args.command == "segment" else False
            ),
        )
    except (OSError, ValueError, RuntimeError, InterruptedError) as error:
        parser.exit(1, f"{type(error).__name__}: {error}\n")
    print(json.dumps(result, indent=2))
