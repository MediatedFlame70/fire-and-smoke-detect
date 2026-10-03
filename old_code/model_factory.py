"""Model construction shared by training, metrics, and inference."""

from cn2vf_net import CN2VFNet


def build_model(model_type: str = "cn2vf", pretrained: bool = True):
    if model_type == "cn2vf":
        return CN2VFNet(num_classes=3)
    if model_type == "dinov3-convnext-tiny":
        from dinov3_model import DINOv3FireSmokeNet

        return DINOv3FireSmokeNet(num_classes=3, pretrained=pretrained)
    raise ValueError(f"Unknown model type: {model_type}")
