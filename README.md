<div align="center">

# biomedparse-mlx

Segment 3D medical images from text prompts with Microsoft BioMedParse v2 and native MLX.

[![PyPI](https://img.shields.io/pypi/v/biomedparse-mlx)](https://pypi.org/project/biomedparse-mlx/)
[![Python](https://img.shields.io/pypi/pyversions/biomedparse-mlx)](https://pypi.org/project/biomedparse-mlx/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue)](https://github.com/MedMLX/biomedparse-mlx/blob/v0.1.5/LICENSE)
[![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-MLX-black?logo=apple)](https://github.com/ml-explore/mlx)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97_Hugging_Face-weights-yellow)](https://huggingface.co/MedMLX/biomedparse-v2-mlx)
[![MedMLX](https://img.shields.io/badge/MedMLX-runtime-A51C30)](https://github.com/MedMLX/medmlx)

[Model weights](https://huggingface.co/MedMLX/biomedparse-v2-mlx) · [MedMLX](https://github.com/MedMLX/medmlx)

<img src="https://raw.githubusercontent.com/MedMLX/biomedparse-mlx/v0.1.5/docs/example-abdomen.png" alt="Abdominal CT segmented from four text prompts: liver, spleen and both kidneys" width="100%">

</div>

| Input | Output |
| :--- | :--- |
| Medical image + text prompts | Labeled segmentation |

## Use with MedMLX

The [MedMLX MCP server](https://github.com/MedMLX/medmlx/blob/main/docs/mcp.md) is the lowest-friction way to use this model. It installs this runtime, downloads its weights on first use, and lets an AI agent describe, check and run `biomedparse-v2` for you.

```sh
uv tool install --python 3.13 "medmlx[segment,mcp]"
MEDMLX="$(uv tool dir --bin --color never)/medmlx"
mkdir -p "$HOME/medmlx-work"
claude mcp add --scope user medmlx -- "$MEDMLX" serve --root "$HOME/medmlx-work"
```

For Codex, use `codex mcp add medmlx -- "$MEDMLX" serve --root "$HOME/medmlx-work"`. Other MCP clients take the same command and arguments. The same install provides the `medmlx` CLI, for example `medmlx describe biomedparse-v2`.

> **For AI agents:** prefer the MedMLX MCP tools (`list_models`, `describe_model`, `model_readiness`, `run_model`, `get_job`) over writing Python against this package.

## Install

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install "biomedparse-mlx>=0.1.5"
```

## Run

```python
from biomedparse_mlx.medmlx import run

result = run(
    image_path="scan.nii.gz",
    modality="ct",
    ct_site="abdomen",
    prompt="Visualization of the liver in abdominal CT imaging",
    labels=[6],
    output_dir="outputs/liver",
)
print(result["outputs"]["mask"])
```

Omitting `weights_path` downloads `MedMLX/biomedparse-v2-mlx` into the shared MedMLX
cache on first use; interrupted downloads resume and later runs work offline. With
MedMLX installed, `medmlx setup biomedparse-v2` downloads it ahead of time. To use your
own copy, pass a folder containing `biomedparse-v2.safetensors` and `tokenizer.json` as
`weights_path`. Standalone checkpoint files and an explicit `tokenizer_path` remain
supported. `details.weights_revision` records the downloaded commit, or `null` for a
folder without a MedMLX download record.

### Inputs and outputs

| Argument | Meaning |
| --- | --- |
| `image_path` | A volume (`.npy`, upstream `.npz` with `imgs`, `.nii` or `.nii.gz`) or a 2D image (PNG, JPEG, BMP, TIFF or DICOM) |
| `prompt` | One text prompt or a list of them |
| `output_dir` | Directory for the outputs; created if missing |
| `labels` | Distinct label per prompt, 1 to 65535; defaults to 1, 2, … in prompt order |
| `modality`, `ct_site` | Required for raw NIfTI and DICOM: `"ct"` with a site (abdomen, lung, pelvis, liver, colon or pancreas), or `"other"` |
| `weights_path`, `tokenizer_path` | Optional model folder or `.safetensors` file, and tokenizer file |
| `slice_batch_size`, `prompt_batch_size` | Slices per model call (default 2) and maximum prompts per decoder group (default all) |
| `existence_threshold`, `seed` | Existence-probability cutoff (default 0.5) and attention-sampling seed (default 0) |
| `instance_segmentation` | Also split touching same-class structures into instances (volumes only) |
| `overwrite` | `False` by default; existing outputs are never replaced |

`.npy` and `.npz` volumes must already hold upstream's modality-specific
preprocessing in the 0–255 range. Raw NIfTI volumes and DICOM slices are windowed per
slice with upstream's formulas: six fixed CT Hounsfield windows by site, or a
0.5–99.5 percentile window for other modalities. The slicing axis follows upstream's
fixed shape heuristic.

A volume writes `segmentation.npz` (`segs`: uint16 labels on the input grid;
`existence`: prompt-by-slice probabilities) and `run.json` with the source revision,
prompts and settings; NIfTI input also gets `segmentation.nii.gz` on the source affine.
Several prompts are merged with per-slice mask NMS and an argmax against a 0.5
background. `instance_segmentation=True` adds an `instances` array (and
`instances.nii.gz`), using upstream's edge-refined connected-component split. A 2D
image follows upstream's `inference_2d.py`: `segmentation.npz` holds per-prompt
`masks`, `probabilities` and `existence`, and masks of different prompts can overlap.
The released checkpoint is trained for 3D modalities, so 2D accuracy is not
established.

Attention is sampled during inference. A seed reproduces a run only on the same
backend, prompt order, input shape and batch settings, and does not align with
Torch's generator.

### Command line

```bash
biomedparse-mlx segment \
  --image scan.npy --weights weights/biomedparse-v2.safetensors --tokenizer weights/tokenizer.json \
  --prompt 'Visualization of the liver in abdominal CT imaging' --label 6 \
  --output-dir outputs/liver
```

`segment-2d` does the same for a 2D image. Repeat `--prompt` and `--label` for several
structures; the other options mirror the `run` arguments.

### Python API

`biomedparse_mlx.pipeline.segment_volume` and `biomedparse_mlx.image2d.segment_image`
run the same pipeline on preprocessed arrays; `segment_volume` also takes progress and
cancellation callbacks. For interactive prompt changes, `model.encode_images(images)`
returns image features that `model.decode_prompts(encoded, prompt_features,
prompt_batch_size=2)` reuses without recomputing the backbone; features from another
model instance are rejected. Prompt batching bounds repeated image features and
decoder activations, not total memory. Inference needs Apple Silicon with Metal and
has no CPU fallback.

## License

[Apache-2.0](https://github.com/MedMLX/biomedparse-mlx/blob/v0.1.5/LICENSE). See [third-party notices](https://github.com/MedMLX/biomedparse-mlx/blob/v0.1.5/THIRD_PARTY_NOTICES.md). Model weights retain their publisher terms. BioMedParse weights are CC-BY-NC-SA-4.0.

For research and education. Outputs require human review.
