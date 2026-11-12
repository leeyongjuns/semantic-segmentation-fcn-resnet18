# Cleanup and audit notes

## Source selection

Extracted from `天津大学.zip` → the undergraduate `作业2_语义分割_...` directory. The relevant source contained `segment.py`, `split.py`, `visualize.py`, exported `result_*.png`, a report, and unrelated interpolation/administrative files.

## Excluded from public package

- Student-identifying original report and evaluation/administrative documents (contain student number / name / institutional information).
- Original `split.py` and `visualize.py`: contained a local path associated with another named user and may be course-provided auxiliary material; no redistribution of those originals.
- Unrelated interpolation experiment, archive metadata, macOS resource forks, hidden cache files, any dataset or model weights.

## Code modernization

1. Replaced a personal absolute data path with `--voc-root`.
2. Read the official PASCAL VOC `train.txt` and `val.txt` segmentation splits to avoid using training examples to report evaluation metrics.
3. Used exact-target-size interpolation for logits, replacing fixed-scale 32× interpolation which assumes particular input dimensions.
4. Preserved indexed mask labels and `255` ignore pixels during nearest-neighbor resize.
5. Added device detection (CUDA, Apple Silicon MPS when supported, or CPU), deterministic basic seeding, CLI entry points, JSON logging, and checkpoint loading.
6. Replaced the original auxiliary metrics routine with dataset-aggregated confusion-matrix pixel accuracy, mIoU and Dice; this fixes the original flatten-then-reshape Hausdorff routine and excludes void pixels.
7. Retained seven original exported PNG figures but did not include unverified report metrics, private assessment, or model checkpoints.

## Testing limitations

- `python -m compileall`, Python AST parsing, package file checks, privacy pattern scanning, and static code checks are possible without original VOC dataset.
- Complete GPU/CPU training cannot be claimed unless the correct PyTorch environment, dataset, model weights and sufficient compute are available.
- No live performance metrics or numerical comparisons are asserted.

## Prepublication checklist

- [ ] Confirm student ownership and sharing permissions for original assignment and example images.
- [ ] Run training and evaluation on a clean environment; record hardware, package versions, training seed, and output metrics.
- [ ] Review predictions for personally identifying people before publication (sample images originate from a public research dataset but redistribution terms should be checked).
- [ ] Check that the repo does not contain university-owned restricted or instructor-only content.
- [ ] Decide on the appropriate open-source license after checking rights.
