# Third-Party Notices

This project (DC-LIDM) is distributed under the Apache License, Version 2.0 (see `LICENSE`).
It contains or derives from the following third-party works. All copyright notices are retained
as required by their licenses.

## 1. BasicSR — the `basicsr/` framework

- Source: https://github.com/XPixelGroup/BasicSR
- Copyright 2019-2022 Xintao Wang (and BasicSR contributors)
- License: Apache License 2.0

The `basicsr/` directory in this repository is a trimmed derivative of BasicSR: the data pipeline,
model base classes, learning-rate schedulers, metrics, logging and utility modules are modified
versions of the originals, with project-specific additions (mask-aware datasets, the AIFlashModel
wrapper, masked PSNR/SSIM/ΔE metrics, and the visualization utilities).
Original files retain their Apache-2.0 terms; modifications are released under the same license.

## 2. Restormer — architecture reference for the main model

- Source: https://github.com/swz30/Restormer
- The main network in `basicsr/models/archs/my_restormer_arch.py` follows the Restormer
  encoder–decoder architecture (multi-Dconv head transposed attention, gated-Dconv feed-forward
  networks, progressive up/down-sampling) and was re-implemented/extended for DC-LIDM.
- Please refer to the original repository for its license terms and cite the Restormer paper when
  using this architecture.

## 3. NTIRE 2026 AI Flash Portrait dataset

- Source: https://www.codabench.org/competitions/12885/
- The dataset is **not** redistributed in this repository. Download it from the challenge website
  and comply with its terms of use. If you publish results obtained with it, please cite the
  challenge report (`guan2026` in the paper's bibliography).

## 4. Python dependencies

Runtime dependencies are listed in `requirements.txt`. Each package is distributed under its own
license (e.g. PyTorch and torchvision under BSD-3-Clause, NumPy under BSD-3-Clause, OpenCV under
Apache-2.0, Kornia under Apache-2.0, scikit-image under BSD-3-Clause). They are installed
separately by the user and are not vendored here.

## 5. Learned parameters

The released checkpoint `best_Restormer_LIDM.pth` contains only weights trained by the authors of
this repository on the public challenge data; it does not include any third-party pre-trained
weights. The ImageNet-pretrained VGG19 used by the perceptual loss is downloaded at runtime by
`torchvision` and is subject to its own license.