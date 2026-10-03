"""Fire/smoke prediction head on Meta's pretrained DINOv3 ConvNeXt-Tiny."""

import torch
import torch.nn as nn
import torch.nn.functional as F


DINOV3_MODEL_ID = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"


class DINOv3FireSmokeNet(nn.Module):
    """Transfer-learn a compact DINOv3 visual encoder for this dataset."""

    def __init__(self, num_classes: int = 3, pretrained: bool = True):
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as error:
            raise ImportError(
                "The DINOv3 option requires transformers>=4.56.0. Install project requirements first."
            ) from error

        if not pretrained:
            raise ValueError("DINOv3 construction requires its pretrained initialization.")
        self.backbone = AutoModel.from_pretrained(DINOV3_MODEL_ID)

        config = self.backbone.config
        feature_dim = getattr(config, "hidden_sizes", [None])[-1]
        if feature_dim is None:
            feature_dim = getattr(config, "hidden_size", None)
        if feature_dim is None:
            raise ValueError("Unable to determine the DINOv3 feature dimension from its config.")

        self.neck = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(0.25),
            nn.Linear(feature_dim, 512),
            nn.GELU(),
            nn.Dropout(0.20),
        )
        self.classifier = nn.Linear(512, num_classes)
        self.box_regressor = nn.Linear(512, 4)

    def freeze_backbone(self, frozen: bool = True):
        for parameter in self.backbone.parameters():
            parameter.requires_grad = not frozen

    def forward(self, images):
        outputs = self.backbone(pixel_values=images)
        features = outputs.last_hidden_state
        if features.ndim == 4:
            pooled = F.adaptive_avg_pool2d(features, 1).flatten(1)
        elif features.ndim == 3:
            patch_features = features[:, 1:] if features.shape[1] > 1 else features
            pooled = patch_features.mean(dim=1)
        else:
            raise RuntimeError(f"Unexpected DINOv3 feature shape: {tuple(features.shape)}")

        shared = self.neck(pooled)
        return {
            "cls_logits": self.classifier(shared),
            "bbox": torch.sigmoid(self.box_regressor(shared)),
        }
