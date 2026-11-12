"""A compact FCN-style classifier using a ResNet-18 feature extractor."""
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

class FCNResNet18(nn.Module):
    def __init__(self, num_classes=21, pretrained=True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        self.classifier = nn.Conv2d(512, num_classes, kernel_size=1)

    def forward(self, images):
        spatial_size = images.shape[-2:]
        logits = self.classifier(self.features(images))
        return F.interpolate(logits, size=spatial_size, mode="bilinear", align_corners=False)
