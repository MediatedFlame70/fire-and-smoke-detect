"""
generate_architecture_diagrams.py
==================================
Renders two publication-grade visual architectural diagram images:
1. output/architecture_pipeline_diagram.png
   - The complete end-to-end CN2VF-Net architecture pipeline with all tensor dimensions and stage connections.
2. output/layer_processing_techniques_diagram.png
   - Detailed visual breakdown of every individual building block and processing technique.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

def create_architecture_pipeline_diagram(output_path="output/architecture_pipeline_diagram.png"):
    fig, ax = plt.subplots(figsize=(22, 12), dpi=300)
    ax.set_facecolor("#0F172A")
    fig.patch.set_facecolor("#0F172A")
    ax.axis("off")

    # Title
    ax.text(0.5, 0.96, "CN2VF-Net: End-to-End Hybrid Architecture & Tensor Data Flow",
            ha="center", va="center", color="#F8FAFC", fontsize=22, fontweight="bold")
    ax.text(0.5, 0.93, "4-Way Classification + Dual-Class CIoU Bounding Box Regression",
            ha="center", va="center", color="#94A3B8", fontsize=14)

    # Style helper
    def draw_box(x, y, w, h, title, subtitle, color, shape_text=""):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.02",
                                      facecolor=color, edgecolor="#38BDF8", linewidth=1.5, alpha=0.95)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h*0.65, title, ha="center", va="center", color="#FFFFFF", fontsize=11, fontweight="bold")
        ax.text(x + w/2, y + h*0.35, subtitle, ha="center", va="center", color="#E2E8F0", fontsize=9)
        if shape_text:
            shape_box = patches.FancyBboxPatch((x + w*0.05, y - 0.035), w*0.9, 0.028, boxstyle="round,pad=0.005",
                                               facecolor="#1E293B", edgecolor="#64748B", linewidth=1)
            ax.add_patch(shape_box)
            ax.text(x + w/2, y - 0.021, shape_text, ha="center", va="center", color="#38BDF8", fontsize=8.5, fontweight="bold")

    def draw_arrow(x1, y1, x2, y2, label=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#38BDF8", lw=2, mutation_scale=15))
        if label:
            ax.text((x1+x2)/2, (y1+y2)/2 + 0.015, label, ha="center", va="center", color="#FDE047", fontsize=8, fontweight="bold")

    # Group Boundaries (Background Panels)
    def draw_group_panel(x, y, w, h, label, bg_color="#1E293B"):
        panel = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
                                      facecolor=bg_color, edgecolor="#475569", linewidth=1.2, linestyle="--", alpha=0.7)
        ax.add_patch(panel)
        ax.text(x + 0.015, y + h - 0.025, label, ha="left", va="center", color="#38BDF8", fontsize=12, fontweight="bold")

    # Panels
    draw_group_panel(0.02, 0.52, 0.28, 0.36, "STAGE 1: INPUT & PREPROCESSING")
    draw_group_panel(0.32, 0.52, 0.32, 0.36, "STAGE 2: MOBILE INVERTED RESIDUAL CNN BACKBONE")
    draw_group_panel(0.66, 0.52, 0.32, 0.36, "STAGE 3: PYRAMIDAL VISION TRANSFORMER (ViT)")
    draw_group_panel(0.02, 0.06, 0.44, 0.40, "STAGE 4: MULTI-SCALE FEATURE FUSION & SE GATING")
    draw_group_panel(0.48, 0.06, 0.50, 0.40, "STAGE 5: GeM POOLING & DUAL-TASK OUTPUT HEADS")

    # STAGE 1: Input & Preprocessing
    draw_box(0.04, 0.70, 0.10, 0.12, "Input Image", "RGB Native Image", "#334155", "[H, W, 3]")
    draw_arrow(0.14, 0.76, 0.18, 0.76)
    draw_box(0.18, 0.70, 0.10, 0.12, "Letterbox", "Aspect-Safe Canvas", "#475569", "[448, 448, 3]")
    draw_arrow(0.23, 0.70, 0.23, 0.58)
    draw_box(0.18, 0.55, 0.10, 0.08, "Normalize", "ImageNet Mean/Std", "#0284C7", "[B, 3, 448, 448]")

    # STAGE 2: CNN Backbone
    draw_arrow(0.28, 0.59, 0.34, 0.76)
    draw_box(0.34, 0.70, 0.08, 0.12, "CNN Stem", "Conv3x3 (S=2) + GELU", "#0369A1", "[B, 16, 224, 224]")
    draw_arrow(0.42, 0.76, 0.45, 0.76)
    draw_box(0.45, 0.70, 0.08, 0.12, "Stage 1 (c1)", "2x InvertedResidual", "#0284C7", "[B, 24, 112, 112]")
    draw_arrow(0.53, 0.76, 0.55, 0.76)
    draw_box(0.55, 0.70, 0.08, 0.12, "Stage 2 (c2)", "2x InvertedResidual", "#0EA5E9", "[B, 40, 56, 56]")
    
    # Stage 3 & 4
    draw_arrow(0.59, 0.70, 0.40, 0.58)
    draw_box(0.35, 0.55, 0.10, 0.08, "Stage 3 (c3)", "2x InvRes (S=2)", "#38BDF8", "[B, 80, 28, 28]")
    draw_arrow(0.45, 0.59, 0.50, 0.59)
    draw_box(0.50, 0.55, 0.12, 0.08, "Stage 4 (c4)", "2x InvRes (S=2)", "#7DD3FC", "[B, 128, 14, 14]")

    # STAGE 3: ViT Stages
    draw_arrow(0.62, 0.59, 0.68, 0.76)
    draw_box(0.68, 0.70, 0.08, 0.12, "PatchEmbed", "Conv1x1 + LayerNorm", "#4338CA", "[B, 196, 128]")
    draw_arrow(0.76, 0.76, 0.78, 0.76)
    draw_box(0.78, 0.70, 0.08, 0.12, "ViT Stage 1", "2x MHSA (4 Heads)", "#6366F1", "[B, 196, 128]")
    draw_arrow(0.86, 0.76, 0.88, 0.76)
    draw_box(0.88, 0.70, 0.08, 0.12, "TokenDown 1", "AvgPool2d + Linear", "#818CF8", "[B, 49, 192]")

    draw_arrow(0.92, 0.70, 0.74, 0.58)
    draw_box(0.68, 0.55, 0.12, 0.08, "ViT Stage 2 (t2)", "2x MHSA (6 Heads)", "#A855F7", "[B, 49, 192]")
    draw_arrow(0.80, 0.59, 0.84, 0.59)
    draw_box(0.84, 0.55, 0.12, 0.08, "ViT Stage 3 (t3)", "Linear + 2x MHSA", "#C084FC", "[B, 49, 256]")

    # STAGE 4: Multi-Scale Feature Fusion
    # Route arrows from c2, c3, t2, t3 down to Fusion
    draw_arrow(0.59, 0.70, 0.08, 0.38, "c2")
    draw_arrow(0.40, 0.55, 0.14, 0.38, "c3")
    draw_arrow(0.74, 0.55, 0.20, 0.38, "t2")
    draw_arrow(0.90, 0.55, 0.26, 0.38, "t3")

    draw_box(0.04, 0.26, 0.24, 0.11, "Multi-Scale Projections", "Conv1x1 to 128 + Bilinear Interp (56x56)", "#0F766E", "[B, 512, 56, 56]")
    draw_arrow(0.16, 0.26, 0.16, 0.20)
    draw_box(0.04, 0.10, 0.18, 0.09, "SE Gating & DW-Conv", "AdaptiveAvgPool + Depthwise Conv", "#0D9488", "[B, 128, 56, 56]")
    draw_arrow(0.22, 0.14, 0.26, 0.14)
    draw_box(0.26, 0.10, 0.18, 0.09, "Fused Feature Map", "Concatenated & Refined", "#14B8A6", "[B, 128, 56, 56]")

    # STAGE 5: GeM Pooling & Output Heads
    draw_arrow(0.44, 0.14, 0.50, 0.32)
    draw_box(0.50, 0.26, 0.16, 0.11, "Head Stem & GeM Pool", "Learnable p-norm spatial collapse", "#B45309", "[B, 128]")

    # Branch 1: Classification
    draw_arrow(0.66, 0.34, 0.72, 0.34)
    draw_box(0.72, 0.30, 0.24, 0.08, "4-Way Classification Head", "Linear(128->64) + Linear(64->4) Logits", "#BE123C", "[B, 4] cls_logits")

    # Branch 2: Fire Box
    draw_arrow(0.66, 0.31, 0.72, 0.21)
    draw_box(0.72, 0.17, 0.24, 0.08, "Fire Bounding Box Head", "Linear(128->64) + Sigmoid() -> [cx,cy,w,h]", "#EA580C", "[B, 4] fire_bbox")

    # Branch 3: Smoke Box
    draw_arrow(0.66, 0.28, 0.72, 0.08)
    draw_box(0.72, 0.05, 0.24, 0.08, "Smoke Bounding Box Head", "Linear(128->64) + Sigmoid() -> [cx,cy,w,h]", "#475569", "[B, 4] smoke_bbox")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="#0F172A")
    plt.close()
    print(f"[+] Architecture pipeline diagram saved to: {output_path}")

def create_layer_processing_techniques_diagram(output_path="output/layer_processing_techniques_diagram.png"):
    fig = plt.figure(figsize=(22, 14), dpi=300)
    fig.patch.set_facecolor("#0F172A")
    gs = GridSpec(2, 3, figure=fig, hspace=0.32, wspace=0.25, left=0.04, right=0.96, top=0.92, bottom=0.05)

    fig.suptitle("CN2VF-Net: Deep Learning Layer Processing Techniques & Internal Mechanics",
                 fontsize=22, fontweight="bold", color="#F8FAFC", y=0.97)

    def style_subpanel(ax, title, subtitle):
        ax.set_facecolor("#1E293B")
        ax.set_title(title, fontsize=14, fontweight="bold", color="#38BDF8", pad=12)
        ax.text(0.5, 0.94, subtitle, transform=ax.transAxes, ha="center", va="center", color="#94A3B8", fontsize=9.5)
        ax.axis("off")

    # Panel 1: MobileNetV3 Inverted Residual Block
    ax1 = fig.add_subplot(gs[0, 0])
    style_subpanel(ax1, "1. Inverted Residual Block (MobileNetV3)", "Local Edge & Texture Extraction with Depthwise Convolutions")
    tech_text_1 = """• Step 1: 1x1 Pointwise Conv (Expansion)
  Expands channels by factor e = 4.0:
  X_exp = GELU(BN(Conv1x1(X_in)))

• Step 2: 3x3 Depthwise Conv (Spatial Filter)
  Applies spatial filtering per channel (Groups = C_hidden):
  X_dw = GELU(BN(DWConv3x3(X_exp, stride=S)))

• Step 3: 1x1 Pointwise Conv (Linear Projection)
  Compresses back to target channel count C_out:
  X_proj = BN(Conv1x1(X_dw))

• Step 4: Residual Addition (Skip Connection)
  If stride == 1 and C_in == C_out:
  Output = X_in + X_proj"""
    ax1.text(0.05, 0.45, tech_text_1, color="#E2E8F0", fontsize=10.5, family="monospace", va="center")

    # Panel 2: Vision Transformer MHSA Block
    ax2 = fig.add_subplot(gs[0, 1])
    style_subpanel(ax2, "2. Vision Transformer MHSA Block", "Multi-Head Self-Attention for Global Context Modeling")
    tech_text_2 = """• Step 1: LayerNorm & Linear Q, K, V Projections
  Q = X_norm W_Q,  K = X_norm W_K,  V = X_norm W_V
  Dimension per head: d_k = d / num_heads

• Step 2: Scaled Dot-Product Self-Attention
  Attention(Q, K, V) = softmax( (Q K^T) / √d_k ) V
  Captures long-range spatial dependencies across all tokens.

• Step 3: Stochastic Depth Residual Connection
  X_1 = X + DropPath(Attention(Q, K, V))

• Step 4: Feed-Forward Network (FFN)
  FFN(X) = Linear(GELU(Linear(X_norm, 4d)), d)
  Output = X_1 + DropPath(FFN(LayerNorm(X_1)))"""
    ax2.text(0.05, 0.45, tech_text_2, color="#E2E8F0", fontsize=10.5, family="monospace", va="center")

    # Panel 3: Token Downsampling Mechanism
    ax3 = fig.add_subplot(gs[0, 2])
    style_subpanel(ax3, "3. Pyramidal Token Downsampling", "Spatial Sequence Reduction & Dimension Upscaling")
    tech_text_3 = """• Step 1: 2D Spatial Grid Reshaping
  Reshapes 1D token sequence [B, N, C] back to 2D:
  X_2d = X.transpose(1, 2).reshape(B, C, H, W)
  (e.g., 196 tokens -> 14 x 14 grid)

• Step 2: 2x Spatial Average Pooling
  X_ds = AvgPool2d(X_2d, kernel_size=2, stride=2)
  Reduces 14 x 14 (196 tokens) -> 7 x 7 (49 tokens).

• Step 3: Sequence Flatten & Linear Embedding
  X_flat = X_ds.flatten(2).transpose(1, 2)
  Output = Linear(X_flat, in_dim=128 -> out_dim=192)
  Cuts self-attention computational cost by 4x."""
    ax3.text(0.05, 0.45, tech_text_3, color="#E2E8F0", fontsize=10.5, family="monospace", va="center")

    # Panel 4: Multi-Scale Fusion & SE Gating
    ax4 = fig.add_subplot(gs[1, 0])
    style_subpanel(ax4, "4. Multi-Scale Fusion & SE Gating", "Fuses 4 Hierarchical Scales (c2, c3, t2, t3)")
    tech_text_4 = """• Step 1: Multi-Scale 1x1 Projections & Interpolation
  Projects c2 (40ch), c3 (80ch), t2 (192ch), t3 (256ch) to 128ch.
  Bilinear upsamples all maps to uniform (56 x 56).

• Step 2: Channel Concatenation
  X_cat = Concat([c2_p, c3_p, t2_p, t3_p]) -> [B, 512, 56, 56]

• Step 3: Squeeze-and-Excitation (SE) Channel Attention
  Gate = Sigmoid(Conv1x1(GELU(Conv1x1(AdaptiveAvgPool(X_cat)))))
  Dynamically re-weights spatial CNN vs ViT channels.

• Step 4: Depthwise-Pointwise Fusion Conv
  Output = Conv1x1(DWConv3x3(X_cat * Gate)) -> [B, 128, 56, 56]"""
    ax4.text(0.05, 0.45, tech_text_4, color="#E2E8F0", fontsize=10.5, family="monospace", va="center")

    # Panel 5: Learnable GeM Pooling
    ax5 = fig.add_subplot(gs[1, 1])
    style_subpanel(ax5, "5. Generalized-Mean (GeM) Pooling", "Learnable p-norm Salient Feature Aggregation")
    tech_text_5 = """• Mathematical Formulation:
  f_c = ( 1 / (H · W) ∑_{i=1}^H ∑_{j=1}^W x_{c, i, j}^p )^{1/p}

• Parameterization:
  p = softplus(raw_p) + 1.0  (Ensures p >= 1.0, initialized p=3.0)

• Physical Significance:
  - If p -> 1.0: Becomes standard Average Pooling.
  - If p -> ∞:   Becomes Max Pooling.
  - At p ≈ 3.0:  Selectively amplifies localized high-intensity
    fire/smoke thermal hubs while suppressing background noise.

• Output: Collapses [B, 128, 56, 56] -> 1D Vector [B, 128]"""
    ax5.text(0.05, 0.45, tech_text_5, color="#E2E8F0", fontsize=10.5, family="monospace", va="center")

    # Panel 6: Dual-Task Output Heads & Loss
    ax6 = fig.add_subplot(gs[1, 2])
    style_subpanel(ax6, "6. Multi-Task Heads & Loss Objectives", "Simultaneous 4-Way Classification + Dual CIoU BBoxes")
    tech_text_6 = """• 1. 4-Way Classification Head:
  Logits = Linear(GELU(Linear(f_gem, 64)), 4)  [B, 4]
  Loss: Class-Weighted Focal Loss (gamma=2.0, label_smoothing=0.05)

• 2. Dual Bounding Box Regressors:
  Fire Box  = Sigmoid(Linear(GELU(Linear(f_gem, 64)), 4)) [B, 4]
  Smoke Box = Sigmoid(Linear(GELU(Linear(f_gem, 64)), 4)) [B, 4]
  Loss: Complete-IoU (CIoU) Loss on normalized [cx, cy, w, h].

• Total Multi-Task Loss:
  L_total = 1.0 · L_Focal + 1.0 · ( I_fire · L_CIoU_f + I_smoke · L_CIoU_s )"""
    ax6.text(0.05, 0.45, tech_text_6, color="#E2E8F0", fontsize=10.5, family="monospace", va="center")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="#0F172A")
    plt.close()
    print(f"[+] Layer processing techniques diagram saved to: {output_path}")

if __name__ == "__main__":
    create_architecture_pipeline_diagram()
    create_layer_processing_techniques_diagram()
