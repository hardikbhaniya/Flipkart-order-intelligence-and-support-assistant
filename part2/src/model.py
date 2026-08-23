"""
Part 2 Task 3: Transfer-learning model definition.

Wraps a pretrained ResNet-18 backbone (frozen early/middle layers) with a new
classifier head sized for Fashion-MNIST's 10 classes. This single class is used
consistently for feature-extraction, fine-tuning, evaluation, and inference so
the saved artifact always has one predictable architecture.

Input size documented: 224x224x3 (standard ResNet-18 ImageNet input size).
Normalization: ImageNet mean/std, since the backbone was pretrained on ImageNet.
"""
import torch
import torch.nn as nn
import torchvision.models as models

NUM_CLASSES = 10
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INPUT_SIZE = 224


class FlipkartProductClassifier(nn.Module):
    """ResNet-18 backbone (feature extractor) + a small linear classifier head."""

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        # Strip off the original 1000-class ImageNet FC layer; keep everything
        # up to (and including) the global average pool, which outputs a
        # 512-dim feature vector per image.
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.feature_dim = 512
        self.head = nn.Linear(self.feature_dim, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Run only the frozen backbone forward pass -> 512-dim feature vector."""
        with torch.no_grad():
            feats = self.backbone(x)
        return torch.flatten(feats, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        feats = torch.flatten(feats, 1)
        return self.head(feats)

    def forward_from_features(self, feats: torch.Tensor) -> torch.Tensor:
        """Run only the head on already-extracted 512-dim feature vectors
        (used during the fast, cached-feature training stage)."""
        return self.head(feats)

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_late_layers(self):
        """Gradual-unfreezing strategy (Task 4): unfreeze only the last
        residual block (layer4) of ResNet-18, keep everything earlier frozen."""
        # self.backbone children order (post-strip): conv1, bn1, relu, maxpool,
        # layer1, layer2, layer3, layer4, avgpool
        layer4 = self.backbone[7]
        for p in layer4.parameters():
            p.requires_grad = True


def build_model(pretrained: bool = True) -> FlipkartProductClassifier:
    model = FlipkartProductClassifier(pretrained=pretrained)
    model.freeze_backbone()
    return model
