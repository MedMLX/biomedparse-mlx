# Third-party notices

MedMLX-authored additions are licensed under [Apache-2.0](LICENSE).
The upstream attribution and license notices are retained below.

biomedparse-mlx

This package adapts the inference architecture and preprocessing of Microsoft's
BioMedParse v2, revision f571d5ccaa4f8c0479bb89e390a7ef0dfc0736f4,
https://github.com/microsoft/BiomedParse, licensed under Apache-2.0.
The implementation was changed to use native MLX, channel-last tensors, local
tokenizer assets, strict loading, and typed inference actions.

FocalNet: Copyright (c) 2022 Microsoft; written by Jianwei Yang.
X-Decoder language encoder: Copyright (c) 2022 Microsoft; written by Xueyan Zou.
These portions carry MIT notices; full terms are retained below.

Transformer/FPN, attention layers, and sine positional encoding retain upstream
copyright (c) Facebook, Inc. and its affiliates (Meta). The BioMedParse source
distributes these adaptations under its root Apache-2.0 license.

Interpolation formulas follow PyTorch's UpSample.h and UpSampleKernel.cpp;
PyTorch BSD terms are retained in THIRD_PARTY_NOTICES.md.

Pretrained weights are separate assets. The Microsoft Hugging Face model card
declares CC-BY-NC-SA-4.0; no checkpoint, tokenizer, dataset, or converted weights
are distributed in this repository. Cite BioMedParse, BoltzFormer, and the
CVPR 2025 Text-guided 3D Segmentation Challenge as requested by upstream.

Dependencies declared in [pyproject.toml](pyproject.toml) are installed separately
and retain their respective licenses.

## README example image

The abdominal CT in `docs/example-abdomen.png` is derived from AMOS22 by
Yuanfeng Ji et al., *AMOS: A Large-Scale Abdominal Multi-Organ Benchmark for
Versatile Medical Image Segmentation*, NeurIPS 2022.
The [dataset record](https://doi.org/10.5281/zenodo.7262581) declares
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), while the
dataset's own metadata lists
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
The illustration crops the public BioMedParse AMOS example and adds native MLX
predictions, colors and a sampled 3D mask rendering. It contains no raw volume.
The example is shared under CC BY-SA 4.0 pending clarification of the AMOS22
terms. BioMedParse weights retain their separate CC BY-NC-SA 4.0 terms; the
Apache-2.0 code license does not replace the image or weight terms.


## Full license: FocalNet-MIT.txt

    MIT License

    Copyright (c) Microsoft Corporation.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE


## Full license: PyTorch-BSD.txt

From PyTorch:

Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

From Caffe2:

Copyright (c) 2016-present, Facebook Inc. All rights reserved.

All contributions by Facebook:
Copyright (c) 2016 Facebook Inc.

All contributions by Google:
Copyright (c) 2015 Google Inc.
All rights reserved.

All contributions by Yangqing Jia:
Copyright (c) 2015 Yangqing Jia
All rights reserved.

All contributions by Kakao Brain:
Copyright 2019-2020 Kakao Brain

All contributions by Cruise LLC:
Copyright (c) 2022 Cruise LLC.
All rights reserved.

All contributions by Tri Dao:
Copyright (c) 2024 Tri Dao.
All rights reserved.

All contributions by Arm:
Copyright (c) 2021, 2023-2025 Arm Limited and/or its affiliates

All contributions from Caffe:
Copyright(c) 2013, 2014, 2015, the respective contributors
All rights reserved.

All other contributions:
Copyright(c) 2015, 2016 the respective contributors
All rights reserved.

Caffe2 uses a copyright model similar to Caffe: each contributor holds
copyright over their contributions to Caffe2. The project versioning records
all such contribution and copyright details. If a contributor wants to further
mark their specific copyright on a particular contribution, they should
indicate their copyright solely in the commit message of the change when it is
committed.

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
   and IDIAP Research Institute nor the names of its contributors may be
   used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
