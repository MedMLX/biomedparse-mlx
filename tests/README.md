Image tests use `~/.cache/medmlx/test-data/chest_xray.jpg` and the converted
HCC-TACE-Seg HCC_063 CT, liver mask, and tumor mask named `hcc_063_ct.nii.gz`,
`hcc_063_liver.nii.gz`, and `hcc_063_tumor.nii.gz` in the same directory. The
images remain outside Git. Mask tests check that the source grids align.

The upstream oracle tests use the pinned BiomedParse checkout and CLIP tokenizer.
Set `BIOMEDPARSE_UPSTREAM` to the checkout and `BIOMEDPARSE_TOKENIZER` to the
local tokenizer directory when they are not in the default locations.
