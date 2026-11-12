"""Dataset-level confusion-matrix metrics; ignores VOC void pixels (255)."""
import torch

class SegmentationMetrics:
    def __init__(self, num_classes=21, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    @torch.no_grad()
    def update(self, predictions, targets):
        pred = predictions.detach().reshape(-1).to("cpu").long()
        tgt = targets.detach().reshape(-1).to("cpu").long()
        valid = (tgt != self.ignore_index) & (tgt >= 0) & (tgt < self.num_classes)
        valid &= (pred >= 0) & (pred < self.num_classes)
        index = tgt[valid] * self.num_classes + pred[valid]
        self.confusion += torch.bincount(index, minlength=self.num_classes**2).reshape(self.num_classes, self.num_classes)

    def compute(self):
        mat = self.confusion.float()
        tp = torch.diag(mat)
        support = mat.sum(1)
        predicted = mat.sum(0)
        union = support + predicted - tp
        present = union > 0
        iou = torch.where(present, tp / union.clamp(min=1), torch.zeros_like(tp))
        dice_denom = support + predicted
        dice = torch.where(dice_denom > 0, 2 * tp / dice_denom.clamp(min=1), torch.zeros_like(tp))
        total = mat.sum().item()
        return {
            "pixel_accuracy": (tp.sum().item()/total) if total else None,
            "mean_iou": iou[present].mean().item() if present.any().item() else None,
            "mean_dice": dice[dice_denom > 0].mean().item() if (dice_denom > 0).any().item() else None,
            "valid_pixels": int(total),
            "evaluated_classes": int(present.sum().item()),
        }
