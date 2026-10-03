import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    """Per-sample stochastic depth for regularising residual transformer blocks."""

    def __init__(self, probability=0.0):
        super().__init__()
        self.probability = float(probability)

    def forward(self, x):
        if not self.training or self.probability <= 0.0:
            return x
        keep_probability = 1.0 - self.probability
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_probability)
        return x * mask / keep_probability


class GeMPool2d(nn.Module):
    """Learnable generalized-mean pooling for salient feature aggregation."""

    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        # Parameterize p so it remains greater than one throughout training.
        self.raw_p = nn.Parameter(torch.tensor(math.log(math.expm1(p - 1.0))))
        self.eps = eps

    def forward(self, x):
        p = F.softplus(self.raw_p) + 1.0
        x = F.softplus(x).clamp_min(self.eps)
        return F.avg_pool2d(x.pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class ConvBNAct(nn.Module):
    """Lightweight Conv-BN-Activation block."""

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, groups=1, act_layer=nn.GELU):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            act_layer(),
        )

    def forward(self, x):
        return self.block(x)


class InvertedResidual(nn.Module):
    """
    MobileNetV3-style inverted residual block (simplified for low latency).
    """

    def __init__(self, in_ch, out_ch, stride, expand_ratio=4.0, act_layer=nn.GELU):
        super().__init__()
        hidden_dim = int(in_ch * expand_ratio)
        self.use_residual = stride == 1 and in_ch == out_ch

        layers = []
        if hidden_dim != in_ch:
            layers.append(ConvBNAct(in_ch, hidden_dim, kernel_size=1, stride=1, act_layer=act_layer))

        # Depthwise convolution
        layers.append(
            ConvBNAct(hidden_dim, hidden_dim, kernel_size=3, stride=stride, groups=hidden_dim, act_layer=act_layer)
        )

        # Project back to output channels
        layers.extend(
            [
                nn.Conv2d(hidden_dim, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
            ]
        )

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        if self.use_residual:
            out = x + out
        return out


class PatchEmbed(nn.Module):
    """
    Converts 2D CNN feature maps to 1D token sequences for Transformer stages.
    """

    def __init__(self, in_ch, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=1, stride=1)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)
        # x: [B, embed_dim, H, W]
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        # x: [B, N, C] where N = H * W
        x = self.norm(x)
        return x, h, w


class MHSABlock(nn.Module):
    """
    Transformer block with MHSA + FFN + LayerNorm.
    """

    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.1, drop_path=0.05):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=drop, batch_first=True)

        hidden_dim = int(dim * mlp_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        # x: [B, N, C]
        attn_input = self.norm1(x)
        attn_out, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        x = x + self.drop_path(attn_out)
        # x: [B, N, C]

        ffn_input = self.norm2(x)
        x = x + self.drop_path(self.ffn(ffn_input))
        # x: [B, N, C]
        return x


class TransformerStage(nn.Module):
    """
    A stack of Transformer blocks representing one ViT stage.
    """

    def __init__(self, dim, depth, num_heads, mlp_ratio=4.0, drop=0.1, drop_path=0.1):
        super().__init__()
        drop_path_rates = torch.linspace(0, drop_path, depth).tolist()
        self.blocks = nn.ModuleList(
            [MHSABlock(dim, num_heads, mlp_ratio=mlp_ratio, drop=drop, drop_path=rate)
             for rate in drop_path_rates]
        )

    def forward(self, x):
        # x: [B, N, C]
        for blk in self.blocks:
            x = blk(x)
        # x: [B, N, C]
        return x


class TokenDownsample(nn.Module):
    """
    Efficient 2x spatial token downsampling between Transformer stages.
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x, h, w):
        # x: [B, N, C], N = H * W
        b, n, c = x.shape
        assert n == h * w, "Token count must match H*W"

        x2d = x.transpose(1, 2).reshape(b, c, h, w)
        # x2d: [B, C, H, W]
        x2d = F.avg_pool2d(x2d, kernel_size=2, stride=2)
        # x2d: [B, C, H/2, W/2]

        h2, w2 = x2d.shape[2], x2d.shape[3]
        x_ds = x2d.flatten(2).transpose(1, 2)
        # x_ds: [B, (H/2)*(W/2), C]
        x_ds = self.proj(x_ds)
        # x_ds: [B, (H/2)*(W/2), out_dim]
        return x_ds, h2, w2


class MultiScaleFusion(nn.Module):
    """Fuse two CNN and two transformer feature scales."""

    def __init__(self, c2_dim, c3_dim, t2_dim, t3_dim, fuse_dim=128):
        super().__init__()
        self.c2_proj = ConvBNAct(c2_dim, fuse_dim, kernel_size=1)
        self.c3_proj = ConvBNAct(c3_dim, fuse_dim, kernel_size=1)
        self.t2_proj = ConvBNAct(t2_dim, fuse_dim, kernel_size=1)
        self.t3_proj = ConvBNAct(t3_dim, fuse_dim, kernel_size=1)
        self.fuse = nn.Sequential(
            ConvBNAct(fuse_dim * 4, fuse_dim * 4, kernel_size=3, groups=fuse_dim * 4),
            ConvBNAct(fuse_dim * 4, fuse_dim, kernel_size=1),
        )

    def forward(self, c2, c3, t2_map, t3_map):
        target_size = c2.shape[-2:]
        features = [self.c2_proj(c2)]
        for projection, feature in ((self.c3_proj, c3), (self.t2_proj, t2_map), (self.t3_proj, t3_map)):
            projected = projection(feature)
            features.append(F.interpolate(projected, size=target_size, mode="bilinear", align_corners=False))
        return self.fuse(torch.cat(features, dim=1))


class SalientDetectionHead(nn.Module):
    """Decoupled classification/box heads with GeM salient-feature pooling."""

    def __init__(self, in_dim=128, num_classes=3):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(in_dim, in_dim, kernel_size=3, groups=in_dim),
            ConvBNAct(in_dim, in_dim, kernel_size=1),
        )
        self.gem_pool = GeMPool2d()
        self.cls_branch = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(in_dim // 2, num_classes),
        )
        self.box_branch = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(in_dim // 2, 4), nn.Sigmoid(),
        )

    def forward(self, x):
        features = self.gem_pool(self.stem(x)).flatten(1)
        return self.cls_branch(features), self.box_branch(features)


class CN2VFNet(nn.Module):
    """Four-stage CNN / three-stage transformer hybrid for fire-smoke images."""

    def __init__(self, num_classes=3):
        super().__init__()

        # Stem: [B, 3, 448, 448] -> [B, 16, 224, 224]
        self.stem = ConvBNAct(3, 16, kernel_size=3, stride=2)

        # CNN Stage 1: [B, 16, 224, 224] -> [B, 24, 112, 112]
        self.cnn_stage1 = nn.Sequential(
            InvertedResidual(16, 24, stride=2, expand_ratio=4.0),
            InvertedResidual(24, 24, stride=1, expand_ratio=4.0),
        )

        # CNN Stage 2: [B, 24, 112, 112] -> [B, 40, 56, 56]
        self.cnn_stage2 = nn.Sequential(
            InvertedResidual(24, 40, stride=2, expand_ratio=4.0),
            InvertedResidual(40, 40, stride=1, expand_ratio=4.0),
        )

        # CNN stages 3-4 add a compact high-level 14x14 feature map.
        self.cnn_stage3 = nn.Sequential(
            InvertedResidual(40, 80, stride=2, expand_ratio=4.0),
            InvertedResidual(80, 80, stride=1, expand_ratio=4.0),
        )
        self.cnn_stage4 = nn.Sequential(
            InvertedResidual(80, 128, stride=2, expand_ratio=4.0),
            InvertedResidual(128, 128, stride=1, expand_ratio=4.0),
        )

        # Transformer pyramids use learned 2-D-grid positional embeddings.
        self.patch_embed = PatchEmbed(in_ch=128, embed_dim=128)
        self.pos_embed1 = nn.Parameter(torch.zeros(1, 14 * 14, 128))
        self.pos_embed2 = nn.Parameter(torch.zeros(1, 7 * 7, 192))
        self.pos_embed3 = nn.Parameter(torch.zeros(1, 7 * 7, 256))
        for embedding in (self.pos_embed1, self.pos_embed2, self.pos_embed3):
            nn.init.trunc_normal_(embedding, std=0.02)

        self.vit_stage1 = TransformerStage(128, depth=2, num_heads=4, drop=0.1, drop_path=0.1)
        self.token_down1 = TokenDownsample(128, 192)
        self.vit_stage2 = TransformerStage(192, depth=2, num_heads=6, drop=0.1, drop_path=0.1)
        self.token_down2 = nn.Linear(192, 256)
        self.vit_stage3 = TransformerStage(256, depth=2, num_heads=8, drop=0.1, drop_path=0.1)

        self.fusion = MultiScaleFusion(40, 80, 192, 256, fuse_dim=128)
        self.head = SalientDetectionHead(128, num_classes)

    def forward(self, x):
        # x: [B, 3, 448, 448]

        x = self.stem(x)
        # x: [B, 16, 224, 224]

        c1 = self.cnn_stage1(x)
        # c1: [B, 24, 112, 112]

        c2 = self.cnn_stage2(c1)
        # c2: [B, 40, 56, 56]

        c3 = self.cnn_stage3(c2)
        c4 = self.cnn_stage4(c3)

        t1, h1, w1 = self.patch_embed(c4)
        if t1.shape[1] != self.pos_embed1.shape[1]:
            raise ValueError("CN2VF-Net positional embeddings currently require 448x448 input images.")
        t1 = self.vit_stage1(t1 + self.pos_embed1)
        t2, h2, w2 = self.token_down1(t1, h1, w1)
        t2 = self.vit_stage2(t2 + self.pos_embed2)
        t3 = self.vit_stage3(self.token_down2(t2) + self.pos_embed3)

        batch_size = x.shape[0]
        t2_map = t2.transpose(1, 2).reshape(batch_size, 192, h2, w2)
        t3_map = t3.transpose(1, 2).reshape(batch_size, 256, h2, w2)
        fused = self.fusion(c2, c3, t2_map, t3_map)

        cls_logits, bbox = self.head(fused)
        # cls_logits: [B, 3], bbox: [B, 4]

        return {
            "cls_logits": cls_logits,
            "bbox": bbox,
        }


class FocalLoss(nn.Module):
    """Class-weighted focal loss to emphasize hard and underrepresented classes."""

    def __init__(self, class_weights=None, gamma=2.0, label_smoothing=0.05):
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits, targets):
        per_item_ce = F.cross_entropy(
            logits, targets, reduction="none", label_smoothing=self.label_smoothing
        )
        focal_factor = (1.0 - torch.exp(-per_item_ce)).pow(self.gamma)
        loss = focal_factor * per_item_ce
        if self.class_weights is not None:
            loss = loss * self.class_weights[targets]
        return loss.mean()


def ciou_loss(pred_boxes, target_boxes, eps=1e-7):
    """Complete-IoU loss for normalized [cx, cy, width, height] boxes."""
    pred_wh = pred_boxes[:, 2:4].clamp_min(eps)
    target_wh = target_boxes[:, 2:4].clamp_min(eps)
    pred_min = pred_boxes[:, :2] - pred_wh / 2
    pred_max = pred_boxes[:, :2] + pred_wh / 2
    target_min = target_boxes[:, :2] - target_wh / 2
    target_max = target_boxes[:, :2] + target_wh / 2

    inter_wh = (torch.minimum(pred_max, target_max) - torch.maximum(pred_min, target_min)).clamp_min(0)
    inter_area = inter_wh[:, 0] * inter_wh[:, 1]
    pred_area = pred_wh[:, 0] * pred_wh[:, 1]
    target_area = target_wh[:, 0] * target_wh[:, 1]
    iou = inter_area / (pred_area + target_area - inter_area + eps)

    enclosing_min = torch.minimum(pred_min, target_min)
    enclosing_max = torch.maximum(pred_max, target_max)
    enclosing_wh = (enclosing_max - enclosing_min).clamp_min(eps)
    diagonal_sq = enclosing_wh.square().sum(dim=1) + eps
    center_dist_sq = (pred_boxes[:, :2] - target_boxes[:, :2]).square().sum(dim=1)

    v = (4.0 / math.pi**2) * (
        torch.atan(target_wh[:, 0] / target_wh[:, 1])
        - torch.atan(pred_wh[:, 0] / pred_wh[:, 1])
    ).square()
    with torch.no_grad():
        alpha = v / (1.0 - iou + v + eps)
    return (1.0 - iou + center_dist_sq / diagonal_sq + alpha * v).mean()


if __name__ == "__main__":
    # Quick shape sanity check
    model = CN2VFNet(num_classes=3)
    model.eval()

    with torch.no_grad():
        dummy = torch.randn(2, 3, 448, 448)
        out = model(dummy)
        print("cls_logits shape:", out["cls_logits"].shape)  # [2, 3]
        print("bbox shape:", out["bbox"].shape)  # [2, 4]
