"""
Training script for EfficientViT-YOLOv8 and baseline YOLOv8 models
on the WeaponSenseV2 dataset.

All three models (EfficientViT-YOLOv8, YOLOv8s, YOLOv8m) use the same
Ultralytics YOLO() training flow, producing identical output format
(metrics, checkpoints, plots, etc.).

Usage (from project root):
    PYTHONPATH=src python scripts/train.py --model efficientvit
    PYTHONPATH=src python scripts/train.py --model yolov8s
    PYTHONPATH=src python scripts/train.py --model yolov8m
    PYTHONPATH=src python scripts/train.py --model all
"""

import argparse
import sys
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DATA_YAML = str(PROJECT_ROOT / "data" / "WeaponSenseV2" / "data.yaml")
RESULTS_DIR = PROJECT_ROOT / "results"
YAML_CONFIG = str(PROJECT_ROOT / "configs" / "efficientvit_yolov8.yaml")


# ─────────────────────────────────────────────────────────────────────────────
# Paper-matched training recipe (Berardini et al. MESA 2024 — WeaponSenseV2)
# ─────────────────────────────────────────────────────────────────────────────
# After auditing the three SoTA papers, the same recipe is used for every
# YOLOv8 variant (n/s/m/l/xl) to produce the published AP50 numbers:
#   - SGD, lr0=0.05, momentum=0.9, weight_decay=0.0005
#   - batch=32, 300 epochs, linear LR decay to lrf=0.01 (i.e. 0.05 → 0.0005)
#   - patience=100 epochs of no val-loss improvement
#   - Augmentation: mosaic + HSV + horizontal flip + translate + scale ONLY
#     (NO mixup, NO copy_paste — both kill convergence on this small,
#      high-frame-correlation video dataset)
#   - imgsz=640, COCO-pretrained init
#
# Why our prior runs underperformed:
#   1. lr0 was 5–50× too low (0.001/0.01 vs paper's 0.05)
#   2. mixup=0.15 and copy_paste=0.3 destroyed clean object views
#   3. cos_lr instead of linear (less critical, but matches paper)
#   4. patience=30–50 killed runs before convergence (paper uses 100)
#   5. batch=16 instead of 32 (effective LR halved by linear-scaling rule)
# ─────────────────────────────────────────────────────────────────────────────

PAPER_RECIPE = dict(
    # Optimizer — exact paper match
    optimizer="SGD",
    lr0=0.05,
    lrf=0.01,            # final LR = 0.05 * 0.01 = 0.0005 (paper)
    momentum=0.9,        # paper uses 0.9 (not Ultralytics' 0.937)
    weight_decay=0.0005,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    cos_lr=False,        # LINEAR decay per paper (was cosine)
    # Augmentation — paper-matched: mosaic + HSV + flip + translate + scale
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.0,           # OFF — not in paper recipe
    copy_paste=0.0,      # OFF — not in paper recipe
    translate=0.1,
    scale=0.5,
    close_mosaic=10,
    # Loss weights (Ultralytics defaults — paper uses these too)
    box=7.5,
    cls=0.5,
    dfl=1.5,
    # Schedule
    patience=100,        # paper-matched (was 30–50, killed convergence)
    save=True,
    save_period=50,
    verbose=True,
)


def train_yolov8_baseline(variant="yolov8s", epochs=300, batch=32, imgsz=640, device="0"):
    """Train standard YOLOv8 baseline (s or m variant).

    Uses PAPER_RECIPE — exact match to Berardini et al. MESA 2024 on
    WeaponSenseV2: SGD lr0=0.05, batch=32, linear decay, patience=100,
    mosaic+HSV+flip+translate+scale only (no mixup, no copy_paste).
    """
    print(f"\n{'='*60}")
    print(f"Training YOLOv8 baseline: {variant}")
    print(f"Recipe: PAPER (lr0=0.05, batch=32, linear, no mixup/copy_paste)")
    print(f"{'='*60}\n")

    model = YOLO(f"{variant}.pt")
    results = model.train(
        data=DATA_YAML,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        project=str(RESULTS_DIR),
        name=variant,
        exist_ok=True,
        **PAPER_RECIPE,
    )
    return results


def train_efficientvit_yolov8(epochs=300, batch=32, imgsz=640, device="0"):
    """Train EfficientViT-YOLOv8 with wide neck (YOLOv8s-scale).

    Architecture:
    - Backbone: EfficientViT-B1 (LiteMLA O(n) attention), ImageNet pretrained
    - Neck: Wide PANet (128/256/512 channels) — matches YOLOv8s capacity
    - Head: YOLOv8 Detect

    Pretraining strategy:
    - Backbone: ImageNet weights (transfers well to detection feature extraction)
    - Neck+Head: Random init (COCO weights are CSPDarknet-specific, hurt convergence)

    Training recipe:
    Uses the same PAPER_RECIPE as baselines (fair comparison) with two
    EfficientViT-specific overrides:
      - warmup_epochs=10 (vs 3) — longer warmup protects the pretrained
        transformer backbone from gradient shocks while the random-init
        neck+head are still producing chaotic gradients.
      - amp=False — LiteMLA's softmax-free linear attention has fp16
        overflow risk in the K^T @ V intermediate.
    """
    print(f"\n{'='*60}")
    print(f"Training EfficientViT-YOLOv8 (wide-neck, ImageNet backbone)")
    print(f"Recipe: PAPER + 10ep warmup (protects pretrained ViT backbone)")
    print(f"{'='*60}\n")

    # Register custom modules with Ultralytics parse_model
    from models.efficientvit_modules import register_efficientvit_modules
    register_efficientvit_modules()

    # Build model from YAML
    model = YOLO(YAML_CONFIG, task="detect")

    # Load ONLY backbone pretrained weights (ImageNet).
    # Neck+head trains from scratch — proven critical: COCO neck weights
    # are tuned for CSPDarknet feature distributions and trap the optimizer.
    from models.pretrained_init import load_all_pretrained
    load_all_pretrained(model.model, load_neck=False)

    # Print model info
    total_params = sum(p.numel() for p in model.model.parameters())
    print(f"EfficientViT-YOLOv8: {total_params/1e6:.2f}M parameters")

    # Paper recipe + EfficientViT-specific warmup override
    hyp = {**PAPER_RECIPE, "warmup_epochs": 10}

    results = model.train(
        data=DATA_YAML,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        project=str(RESULTS_DIR),
        name="efficientvit_yolov8",
        exist_ok=True,
        pretrained=False,
        amp=False,  # fp32 — LiteMLA intermediates overflow fp16
        **hyp,
    )
    return results


def main():
    parser = argparse.ArgumentParser(description="Train weapon detection models")
    parser.add_argument("--model", type=str, default="efficientvit",
                        choices=["efficientvit", "yolov8s", "yolov8m", "all"])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=32)  # paper-matched (was 16)
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Training resolution. Try 896 for small-object boost (~+3-6 AP50, 2x slower)")
    parser.add_argument("--multi-scale", action="store_true",
                        help="Multi-scale training (0.5-1.5x random rescaling). Free +1-2 AP50 on small objects")
    parser.add_argument("--device", type=str, default="0")
    args = parser.parse_args()

    # Tier-2 opt-in overrides
    if args.multi_scale:
        PAPER_RECIPE["multi_scale"] = True
        print(f"[Train] multi_scale=True enabled (random 0.5x-1.5x image rescaling)")
    if args.imgsz != 640:
        print(f"[Train] Training at imgsz={args.imgsz} (default 640) — small-object boost")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.model in ("yolov8s", "all"):
        train_yolov8_baseline("yolov8s", args.epochs, args.batch, args.imgsz, args.device)

    if args.model in ("yolov8m", "all"):
        train_yolov8_baseline("yolov8m", args.epochs, args.batch, args.imgsz, args.device)

    if args.model in ("efficientvit", "all"):
        train_efficientvit_yolov8(args.epochs, args.batch, args.imgsz, args.device)


if __name__ == "__main__":
    main()
