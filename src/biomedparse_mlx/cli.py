"""Command-line projection of ``run``: ``segment`` for volumes, ``segment-2d`` for images."""

from __future__ import annotations

import argparse
import sys
from functools import partial
from importlib.metadata import version

from medmlx_core.cli import add_options, call, execute

from biomedparse_mlx.medmlx import run

PROG = "biomedparse-mlx"


def main() -> None:
    """Entry point of ``biomedparse-mlx``.

    Both subcommands take their options from ``run``. ``segment-2d`` binds
    ``slice_batch_size`` and ``instance_segmentation``, which do not apply to a
    single image, so it has no such options.
    """
    parser = argparse.ArgumentParser(prog=PROG, description="BioMedParse v2 native MLX inference")
    parser.add_argument("--version", action="version", version=version(PROG))
    commands = parser.add_subparsers(dest="command", required=True)
    volume = commands.add_parser(
        "segment",
        help="Segment a volume: upstream-preprocessed .npy/.npz, or raw .nii/.nii.gz "
        "windowed with --modality/--ct-site",
    )
    add_options(volume, run)
    image = commands.add_parser(
        "segment-2d",
        help="Segment a single preprocessed 2D image (PNG/JPEG/BMP/TIFF), or a raw "
        "DICOM slice windowed with --modality/--ct-site",
    )
    runner_2d = partial(run, slice_batch_size=2, instance_segmentation=False)
    add_options(image, runner_2d)
    arguments = parser.parse_args()
    runner = run if arguments.command == "segment" else runner_2d
    sys.exit(execute(PROG, lambda: call(runner, arguments)))
