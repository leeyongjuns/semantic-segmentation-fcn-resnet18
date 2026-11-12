# Semantic Segmentation with FCN-ResNet18

A **reorganized undergraduate computer vision experiment** using a lightweight fully convolutional segmentation head on a pretrained ResNet-18 backbone, trained on **PASCAL VOC 2012**. The original course project used PyTorch, a 128×128 input size, cross-entropy loss with ignore label 255, and Adam optimization. This repository presents an improved, portable *code package* derived from that experiment.

> **Reproducibility status:** Code has been statically checked and a smoke test is provided. Full training and quantitative performance have **not** been reproduced for this packaged version. The example figures below are archived undergraduate outputs, **not** new benchmark results.

## Sample undergraduate outputs (qualitative only)

![Archived example 1](results/examples/result_1.png)
![Archived example 2](results/examples/result_2.png)

More original output figures are in [`results/examples/`](results/examples/). Figures are retained as exported and may use a different color visualization from the reorganized script.

## Architecture

`RGB image → ResNet-18 backbone (remove avgpool/FC) → 1×1 convolution (21 channels) → bilinear resize to input dimensions → per-pixel class scores`

This is a **simple FCN-style baseline**, not a reproduction of the complete FCN-8s architecture.

## Dataset

Download PASCAL VOC 2012 from the [official VOC website](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) and extract the `VOCdevkit/VOC2012` directory locally. The reorganized code uses the *official* `ImageSets/Segmentation/train.txt` and `val.txt` lists, preserves VOC indexed mask labels, resizes masks with nearest-neighbor interpolation, and ignores class ID `255`. Data is not distributed here.

Expected directory layout:

```text
VOCdevkit/VOC2012/
  JPEGImages/
  SegmentationClass/
  ImageSets/Segmentation/train.txt
  ImageSets/Segmentation/val.txt
```

## Setup

Python **3.10+** is recommended. Install compatible `torch` and `torchvision` builds for your operating system from the [PyTorch installation guide](https://pytorch.org/get-started/locally/), then:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows use `.venv\Scripts\activate`. Pretrained ResNet-18 weights will be downloaded by torchvision on the first training run unless `--no-pretrained` is passed.

## Train

From the repository root:

```bash
python -m segmentation.train \
  --voc-root /path/to/VOCdevkit/VOC2012 \
  --epochs 10 --batch-size 4 --image-size 128
```

The default is **one epoch** (quick demonstration); explicitly use `--epochs 10` for a longer run. Training logs one record per epoch, saves `outputs/history.json` and the latest checkpoint at `outputs/last.pt`. Validation is computed separately from training using the VOC validation split.

## Evaluate and visualize

```bash
python -m segmentation.evaluate \
  --voc-root /path/to/VOCdevkit/VOC2012 \
  --checkpoint outputs/last.pt

python -m segmentation.visualize \
  --voc-root /path/to/VOCdevkit/VOC2012 \
  --checkpoint outputs/last.pt --count 6
```

Metrics: dataset-level **pixel accuracy**, **mIoU**, **mean Dice**, and number of evaluated (union-present) classes; ignore pixels (`255`) are excluded. Validation scores have not been generated for this release. The original auxiliary script included Hausdorff/recall/F1 calculations, but their batch-level implementation was not retained due to shape/ignore-pixel issues. See [`docs/changes.md`](docs/changes.md).

## Original experiment vs 2026 portfolio cleanup

| Area | Undergraduate source | Portfolio cleanup |
| --- | --- | --- |
| Backbone | ImageNet-pretrained ResNet-18 | Same default backbone |
| Head | 1×1 conv → upsample by fixed factor 32 | 1×1 conv → resize to exact input size |
| Data | Scan VOC images via local absolute path | CLI dataset root + official train/val splits |
| Masks | Nearest-neighbor resize, label 255 ignored by loss | Retain palette-index labels; nearest-neighbor resize |
| Evaluation | Separate auxiliary metrics script | Dataset-level confusion matrix, ignore pixels masked |
| Reproducibility | Hardcoded local paths | Platform-agnostic CLI and dependency list |

The original report described both one-epoch testing and a 10-epoch analysis. The extracted `segment.py` explicitly executes **one epoch**. As the original training history/checkpoint has not been verified, no numerical claim from the report is presented as a reproducible benchmark here.

## Authorship, credits, and limitations

- Undergraduate study/coursework by Yongjun Lee at Tianjin University; repository documentation and implementation structure were reorganized afterward for portfolio presentation.
- The source package included auxiliary `split.py` and `visualize.py` scripts with course-provided or other-person paths. They are **not republished verbatim**; the new loader and metrics/visualization utilities were written for this distribution. Do not attribute all original auxiliary code to the student.
- ResNet-18 architecture / ImageNet pretrained weights: [torchvision](https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html).
- PASCAL VOC 2012 dataset: [VOC challenge](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/).
- There is no verified held-out test performance, multiple-run variance, or model checkpoint in this release. Outputs may show poor predictions and should not be interpreted as production quality.
- **No license is granted for this repository's source code at publication time.** Add a license after checking original course material and any reuse rights.

## Further work

Try consistent image normalization, higher resolution, data augmentation, training longer, and comparing against an official segmentation baseline using a controlled validation protocol. Document actual scores and hardware only after running the experiments.
