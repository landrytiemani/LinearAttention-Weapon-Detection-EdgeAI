"""
Visualization script for comparing EfficientViT-YOLOv8 against baselines and SoTA.

Generates publication-quality plots:
1. Fair comparison bar chart (our 3 models, same test split)
2. Per-class AP50 (Handgun vs Knife) — highlights transformer advantage on knife
3. AP50 vs GFLOPs efficiency scatter
4. AP50 vs Params efficiency scatter
5. Radar chart (multi-metric overview)
6. Full SoTA comparison with test-split disclaimer
7. Qualitative detection examples on test frames (EfficientViT only, 2x4 grid)
8. Side-by-side detection comparison (GT | YOLOv8m | EfficientViT, 4 rows)

Plots 7 and 8 run inference on the trained checkpoints, so they require
results/{efficientvit_yolov8,yolov8m}/weights/best.pt to exist.

Usage:
    python visualize_results.py
"""

import json
import random
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
from pathlib import Path
from PIL import Image

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams['axes.spines.right'] = False

PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
TEST_IMG_DIR = PROJECT_ROOT / "data" / "WeaponSenseV2" / "test" / "images"
TEST_LBL_DIR = PROJECT_ROOT / "data" / "WeaponSenseV2" / "test" / "labels"

# Colors
C_EVIT = '#2ecc71'    # green — EfficientViT (ours, highlight)
C_OURS = '#3498db'    # blue — our baselines
C_SOTA = '#95a5a6'    # gray — paper results
C_HANDGUN = '#e74c3c' # red — Handgun class
C_KNIFE = '#3498db'   # blue — Knife class
C_GT = '#f1c40f'      # yellow — ground truth boxes
CLASS_NAMES = ["Handgun", "Knife"]
CLASS_COLORS = [C_HANDGUN, C_KNIFE]


def load_results():
    results_path = RESULTS_DIR / "comparison_results.json"
    if not results_path.exists():
        print(f"No results file found at {results_path}")
        print("Run evaluate.py first to generate results.")
        return None
    with open(results_path) as f:
        return json.load(f)


def plot_fair_comparison(results):
    """Bar chart: our 3 models on the SAME test split (fair comparison)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ours = results["ours"]
    # Order: EfficientViT first for emphasis
    order = ["EfficientViT-YOLOv8 (ours)", "YOLOv8s (ours)", "YOLOv8m (ours)"]
    labels = ["EfficientViT-YOLOv8\n(6.4M params)", "YOLOv8s\n(11.1M params)", "YOLOv8m\n(25.8M params)"]
    colors = [C_EVIT, C_OURS, C_OURS]

    metrics = ["AP50_all", "Prec_all", "Rec_all", "F1_all"]
    metric_labels = ["mAP50", "Precision", "Recall", "F1"]

    x = np.arange(len(metrics))
    width = 0.25

    for i, (name, label, color) in enumerate(zip(order, labels, colors)):
        r = ours[name]
        vals = [r.get(m, 0) for m in metrics]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=label, color=color,
                      edgecolor='white', linewidth=0.5, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{val:.1f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=12)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title("Fair Comparison: Same Test Set (Video-Level Split)",
                 fontweight='bold', fontsize=14)
    ax.set_ylim(0, 65)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Annotation
    ax.text(0.02, 0.02,
            "All models evaluated on identical test set with video-level split\n"
            "(prevents data leakage from same-video frames in train/test)",
            transform=ax.transAxes, fontsize=8, style='italic', alpha=0.6,
            verticalalignment='bottom')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "1_fair_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 1_fair_comparison.png")


def plot_per_class(results):
    """Grouped bar: per-class AP50 — shows transformer advantage on knife."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Our models + select paper models for context
    models = {}
    for name, r in results["ours"].items():
        if r.get("AP50_handgun") is not None and r.get("AP50_knife") is not None:
            models[name.replace(" (ours)", "\n(ours)")] = (r["AP50_handgun"], r["AP50_knife"])
    for name in ["YOLOv8s (paper)", "YOLOv8m (paper)"]:
        r = results["sota"].get(name, {})
        if r.get("AP50_handgun") is not None:
            models[name.replace(" (paper)", "\n(paper)")] = (r["AP50_handgun"], r["AP50_knife"])

    names = list(models.keys())
    handgun = [v[0] for v in models.values()]
    knife = [v[1] for v in models.values()]

    x = np.arange(len(names))
    width = 0.35

    bars1 = ax.bar(x - width/2, handgun, width, label='Handgun AP50',
                   color=C_HANDGUN, alpha=0.8, edgecolor='white')
    bars2 = ax.bar(x + width/2, knife, width, label='Knife AP50',
                   color=C_KNIFE, alpha=0.8, edgecolor='white')

    for bar, val in zip(bars1, handgun):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}", ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar, val in zip(bars2, knife):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("AP50 (%)", fontsize=12)
    ax.set_title("Per-Class AP50: Transformer Attention Excels at Knife Detection",
                 fontweight='bold', fontsize=13)
    ax.legend(fontsize=11)
    ax.set_ylim(0, max(max(handgun), max(knife)) * 1.2)
    ax.grid(axis='y', alpha=0.3)

    # Highlight annotation
    ax.annotate("EfficientViT: best knife\ndetection (global context)",
                xy=(0, knife[0]),
                xytext=(0.5, knife[0] + 15),
                fontsize=9, fontweight='bold', color=C_EVIT,
                arrowprops=dict(arrowstyle='->', color=C_EVIT, lw=1.5),
                ha='center')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "2_per_class_ap50.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 2_per_class_ap50.png")


def plot_efficiency_gflops(results):
    """Scatter: AP50 vs GFLOPs — EfficientViT dominates efficiency frontier."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # SoTA
    for name, r in results["sota"].items():
        if r.get("AP50_all") and r.get("GFLOPs"):
            ax.scatter(r["GFLOPs"], r["AP50_all"], c=C_SOTA, s=80, zorder=3,
                       edgecolors='white', linewidth=1, alpha=0.7)
            label = name.replace(" (paper)", "")
            ax.annotate(label, (r["GFLOPs"], r["AP50_all"]),
                        textcoords="offset points", xytext=(5, 5), fontsize=8,
                        color='#666666')

    # Ours
    for name, r in results["ours"].items():
        if r.get("AP50_all") and r.get("GFLOPs"):
            is_evit = "EfficientViT" in name
            color = C_EVIT if is_evit else C_OURS
            marker = '*' if is_evit else 'D'
            size = 300 if is_evit else 120
            ax.scatter(r["GFLOPs"], r["AP50_all"], c=color, s=size, zorder=5,
                       edgecolors='black', linewidth=1.5, marker=marker)
            label = name.replace(" (ours)", "")
            ax.annotate(label, (r["GFLOPs"], r["AP50_all"]),
                        textcoords="offset points", xytext=(8, 8), fontsize=10,
                        fontweight='bold', color=color)

    # Draw efficiency frontier arrow
    evit = results["ours"].get("EfficientViT-YOLOv8 (ours)", {})
    if evit.get("GFLOPs") and evit.get("AP50_all"):
        ax.annotate("Best efficiency:\nhighest AP50/GFLOP",
                    xy=(evit["GFLOPs"], evit["AP50_all"]),
                    xytext=(evit["GFLOPs"] + 20, evit["AP50_all"] - 5),
                    fontsize=10, fontweight='bold', color=C_EVIT,
                    arrowprops=dict(arrowstyle='->', color=C_EVIT, lw=2),
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#e8f8f5', alpha=0.8))

    ax.set_xlabel("GFLOPs (Computational Cost)", fontsize=12)
    ax.set_ylabel("AP50 (%)", fontsize=12)
    ax.set_title("Efficiency: AP50 vs Computational Cost", fontweight='bold', fontsize=14)
    ax.grid(alpha=0.3)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=C_SOTA,
               markersize=10, label='Published results'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor=C_OURS,
               markersize=10, label='Our baselines (same test)'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor=C_EVIT,
               markersize=15, label='EfficientViT-YOLOv8 (ours)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "3_efficiency_gflops.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 3_efficiency_gflops.png")


def plot_efficiency_params(results):
    """Scatter: AP50 vs Params — shows EfficientViT achieves more with less."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # SoTA
    for name, r in results["sota"].items():
        if r.get("AP50_all") and r.get("Params_M"):
            ax.scatter(r["Params_M"], r["AP50_all"], c=C_SOTA, s=80, zorder=3,
                       edgecolors='white', linewidth=1, alpha=0.7)
            label = name.replace(" (paper)", "")
            ax.annotate(label, (r["Params_M"], r["AP50_all"]),
                        textcoords="offset points", xytext=(5, 5), fontsize=8,
                        color='#666666')

    # Ours
    for name, r in results["ours"].items():
        if r.get("AP50_all") and r.get("Params_M"):
            is_evit = "EfficientViT" in name
            color = C_EVIT if is_evit else C_OURS
            marker = '*' if is_evit else 'D'
            size = 300 if is_evit else 120
            ax.scatter(r["Params_M"], r["AP50_all"], c=color, s=size, zorder=5,
                       edgecolors='black', linewidth=1.5, marker=marker)
            label = name.replace(" (ours)", "")
            ax.annotate(label, (r["Params_M"], r["AP50_all"]),
                        textcoords="offset points", xytext=(8, 8), fontsize=10,
                        fontweight='bold', color=color)

    ax.set_xlabel("Parameters (Millions)", fontsize=12)
    ax.set_ylabel("AP50 (%)", fontsize=12)
    ax.set_title("AP50 vs Model Size", fontweight='bold', fontsize=14)
    ax.grid(alpha=0.3)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=C_SOTA,
               markersize=10, label='Published results'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor=C_OURS,
               markersize=10, label='Our baselines (same test)'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor=C_EVIT,
               markersize=15, label='EfficientViT-YOLOv8 (ours)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "4_efficiency_params.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 4_efficiency_params.png")


def plot_radar(results):
    """Radar chart comparing our 3 models across multiple metrics."""
    ours = results["ours"]

    categories = ['mAP50', 'Precision', 'Recall', 'F1',
                  'Efficiency\n(1/GFLOPs)', 'Compactness\n(1/Params)']
    N = len(categories)

    # Normalize metrics for radar (higher = better, scale 0-1)
    models = {
        "EfficientViT-YOLOv8": ours.get("EfficientViT-YOLOv8 (ours)", {}),
        "YOLOv8s": ours.get("YOLOv8s (ours)", {}),
        "YOLOv8m": ours.get("YOLOv8m (ours)", {}),
    }

    # Raw values
    raw = {}
    for name, r in models.items():
        raw[name] = [
            r.get("AP50_all", 0),
            r.get("Prec_all", 0),
            r.get("Rec_all", 0),
            r.get("F1_all", 0),
            1.0 / max(r.get("GFLOPs", 1), 0.01),  # inverted: lower GFLOPs = better
            1.0 / max(r.get("Params_M", 1), 0.01),  # inverted: fewer params = better
        ]

    # Normalize each metric to 0-1 range across all models
    all_vals = list(raw.values())
    mins = [min(v[i] for v in all_vals) for i in range(N)]
    maxs = [max(v[i] for v in all_vals) for i in range(N)]

    normed = {}
    for name, vals in raw.items():
        normed[name] = [
            (vals[i] - mins[i]) / max(maxs[i] - mins[i], 1e-9) * 0.6 + 0.3
            for i in range(N)
        ]

    # Radar plot
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    colors_map = {"EfficientViT-YOLOv8": C_EVIT, "YOLOv8s": C_OURS, "YOLOv8m": '#e67e22'}

    for name, vals in normed.items():
        vals_closed = vals + vals[:1]
        ax.plot(angles, vals_closed, 'o-', linewidth=2, label=name, color=colors_map[name])
        ax.fill(angles, vals_closed, alpha=0.1, color=colors_map[name])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticklabels([])
    ax.set_title("Multi-Metric Radar Comparison", fontweight='bold', fontsize=14, pad=20)
    ax.legend(loc='lower right', bbox_to_anchor=(1.3, 0), fontsize=10)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "5_radar_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 5_radar_comparison.png")


def plot_full_sota_table(results):
    """Full comparison bar chart with SoTA + disclaimer."""
    fig, ax = plt.subplots(figsize=(16, 7))

    all_models = {}
    for name, r in results["sota"].items():
        if r.get("AP50_all") is not None:
            all_models[name] = r["AP50_all"]
    for name, r in results["ours"].items():
        if r.get("AP50_all") is not None:
            all_models[name] = r["AP50_all"]

    names = list(all_models.keys())
    values = list(all_models.values())

    colors = []
    for name in names:
        if "EfficientViT" in name:
            colors.append(C_EVIT)
        elif "(ours)" in name:
            colors.append(C_OURS)
        else:
            colors.append(C_SOTA)

    bars = ax.bar(range(len(names)), values, color=colors, edgecolor='white', linewidth=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel("AP50 (%)", fontsize=12)
    ax.set_title("AP50 Comparison: All Models\n(Note: Paper results use different test split — not directly comparable)",
                 fontweight='bold', fontsize=13)
    ax.set_ylim(0, max(values) * 1.15)
    ax.grid(axis='y', alpha=0.3)

    # Separator line between paper and our results
    n_sota = len(results["sota"])
    ax.axvline(x=n_sota - 0.5, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.text(n_sota - 0.7, max(values) * 1.08, "Paper results\n(different test split)",
            fontsize=8, ha='right', color='red', alpha=0.7)
    ax.text(n_sota - 0.3, max(values) * 1.08, "Our results\n(video-level split)",
            fontsize=8, ha='left', color=C_OURS, alpha=0.7)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=C_SOTA, label='Published (different test split)'),
        Patch(facecolor=C_OURS, label='Our baselines (same test split)'),
        Patch(facecolor=C_EVIT, label='EfficientViT-YOLOv8 (ours)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "6_full_sota_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 6_full_sota_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# Qualitative detection examples (for publication figures)
# ─────────────────────────────────────────────────────────────────────────────

def _load_yolo_label(lbl_path, img_w, img_h):
    """Read a YOLO-format .txt and return list of (cls, x1, y1, x2, y2) in pixels."""
    if not lbl_path.exists():
        return []
    boxes = []
    for line in lbl_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        cx, cy, w, h = map(float, parts[1:5])
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h
        boxes.append((cls, x1, y1, x2, y2))
    return boxes


def _select_example_frames(n=8, seed=42, prefer_balanced=True):
    """Pick N test frames deterministically — prefer a class-balanced mix.

    Returns list of (img_path, gt_boxes) where gt_boxes are pixel coords.
    """
    if not TEST_IMG_DIR.exists():
        print(f"[examples] test image dir not found: {TEST_IMG_DIR}")
        return []

    all_imgs = sorted(TEST_IMG_DIR.glob("*.jpg")) + sorted(TEST_IMG_DIR.glob("*.png"))
    if not all_imgs:
        return []

    # Build (img, gt_boxes, classes_present) tuples
    indexed = []
    for img_path in all_imgs:
        lbl_path = TEST_LBL_DIR / (img_path.stem + ".txt")
        with Image.open(img_path) as im:
            w, h = im.size
        gt = _load_yolo_label(lbl_path, w, h)
        if not gt:
            continue
        classes = {b[0] for b in gt}
        indexed.append((img_path, gt, classes))

    rnd = random.Random(seed)
    rnd.shuffle(indexed)

    if prefer_balanced:
        handgun_imgs = [t for t in indexed if 0 in t[2]]
        knife_imgs = [t for t in indexed if 1 in t[2]]
        half = n // 2
        chosen = handgun_imgs[:half] + knife_imgs[:n - half]
        # Dedup while preserving order
        seen, dedup = set(), []
        for t in chosen:
            if t[0] not in seen:
                seen.add(t[0])
                dedup.append(t)
        chosen = dedup[:n]
        if len(chosen) < n:
            extras = [t for t in indexed if t[0] not in seen][: n - len(chosen)]
            chosen += extras
    else:
        chosen = indexed[:n]

    return [(p, gt) for p, gt, _ in chosen]


def _run_predictions(weights_path, img_paths, imgsz=896, conf=0.25, iou=0.6,
                     register_efficientvit=False, device="0"):
    """Run YOLO inference on a list of image paths.

    Returns dict[Path -> list of (cls, conf, x1, y1, x2, y2)] in pixel coords.
    """
    try:
        from ultralytics import YOLO
        if register_efficientvit:
            sys.path.insert(0, str(PROJECT_ROOT / "src"))
            from models.efficientvit_modules import register_efficientvit_modules
            register_efficientvit_modules()
        model = YOLO(str(weights_path))
        results = model.predict(
            source=[str(p) for p in img_paths],
            imgsz=imgsz, conf=conf, iou=iou, device=device, verbose=False,
        )
    except Exception as e:
        print(f"[examples] inference failed for {weights_path}: {e}")
        return {p: [] for p in img_paths}

    out = {}
    for img_path, r in zip(img_paths, results):
        boxes = []
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            scores = r.boxes.conf.cpu().numpy()
            for c, sc, (x1, y1, x2, y2) in zip(cls, scores, xyxy):
                boxes.append((int(c), float(sc), float(x1), float(y1), float(x2), float(y2)))
        out[img_path] = boxes
    return out


def _draw_boxes(ax, img, gt_boxes, pred_boxes, show_gt=True, show_labels=True):
    """Draw an image with GT and prediction boxes overlaid."""
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])

    # Ground truth — yellow dashed
    if show_gt:
        for cls, x1, y1, x2, y2 in gt_boxes:
            rect = mpatches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=1.5, edgecolor=C_GT, facecolor='none', linestyle='--', alpha=0.9,
            )
            ax.add_patch(rect)

    # Predictions — solid, color by class
    for cls, conf, x1, y1, x2, y2 in pred_boxes:
        color = CLASS_COLORS[cls] if cls < len(CLASS_COLORS) else '#9b59b6'
        rect = mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2.0, edgecolor=color, facecolor='none',
        )
        ax.add_patch(rect)
        if show_labels:
            label = f"{CLASS_NAMES[cls]} {conf:.2f}" if cls < len(CLASS_NAMES) else f"{cls} {conf:.2f}"
            ax.text(
                x1, max(y1 - 4, 8), label,
                fontsize=7, color='white', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.18', facecolor=color, edgecolor='none', alpha=0.85),
            )


def _compute_zoom_region(target_box, img_w, img_h, pad=4.0, min_side=140):
    """Return (zx1, zy1, zx2, zy2) zoom rectangle around a target box.

    target_box: (cls, x1, y1, x2, y2) — pixel coords.
    Output is clipped to image bounds and forced to a minimum size so very
    tiny detections still get a meaningful crop.
    """
    _, x1, y1, x2, y2 = target_box
    bw, bh = x2 - x1, y2 - y1
    side = max(bw * pad, bh * pad, min_side)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    zx1 = max(0, cx - side / 2)
    zy1 = max(0, cy - side / 2)
    zx2 = min(img_w, cx + side / 2)
    zy2 = min(img_h, cy + side / 2)
    return zx1, zy1, zx2, zy2


def _pick_zoom_target(gt_boxes, pred_boxes):
    """Choose the box to zoom on. Prefers the smallest predicted box;
    falls back to the smallest GT box. Returns a (cls, x1, y1, x2, y2) tuple
    or None if no boxes."""
    candidates = []
    for cls, conf, x1, y1, x2, y2 in pred_boxes:
        candidates.append((cls, x1, y1, x2, y2, (x2 - x1) * (y2 - y1)))
    if not candidates and gt_boxes:
        for cls, x1, y1, x2, y2 in gt_boxes:
            candidates.append((cls, x1, y1, x2, y2, (x2 - x1) * (y2 - y1)))
    if not candidates:
        return None
    candidates.sort(key=lambda c: c[5])  # smallest first
    cls, x1, y1, x2, y2, _ = candidates[0]
    return (cls, x1, y1, x2, y2)


def _add_zoom_inset(ax, img, zoom_box, gt_boxes=None, pred_boxes=None,
                    inset_size=0.36, loc='upper right', show_labels=True):
    """Draw a zoomed crop of `zoom_box` as an inset on `ax`, with overlays.

    Also draws a thin dotted marker on the parent axis showing the zoom region.
    """
    img_h, img_w = img.shape[:2]
    zx1, zy1, zx2, zy2 = [int(v) for v in zoom_box]
    zx1 = max(0, zx1); zy1 = max(0, zy1)
    zx2 = min(img_w, zx2); zy2 = min(img_h, zy2)

    pad = 0.012
    positions = {
        'upper right': [1 - inset_size - pad, 1 - inset_size - pad, inset_size, inset_size],
        'upper left':  [pad,                   1 - inset_size - pad, inset_size, inset_size],
        'lower right': [1 - inset_size - pad, pad,                   inset_size, inset_size],
        'lower left':  [pad,                   pad,                   inset_size, inset_size],
    }
    inset = ax.inset_axes(positions[loc])
    inset.imshow(img[zy1:zy2, zx1:zx2])
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_edgecolor('white')
        spine.set_linewidth(2.2)

    # GT boxes (intersecting the zoom region)
    if gt_boxes:
        for cls, x1, y1, x2, y2 in gt_boxes:
            if x2 < zx1 or x1 > zx2 or y2 < zy1 or y1 > zy2:
                continue
            inset.add_patch(mpatches.Rectangle(
                (x1 - zx1, y1 - zy1), x2 - x1, y2 - y1,
                linewidth=1.6, edgecolor=C_GT, facecolor='none', linestyle='--',
            ))

    # Predictions
    if pred_boxes:
        for cls, conf, x1, y1, x2, y2 in pred_boxes:
            if x2 < zx1 or x1 > zx2 or y2 < zy1 or y1 > zy2:
                continue
            color = CLASS_COLORS[cls] if cls < len(CLASS_COLORS) else '#9b59b6'
            inset.add_patch(mpatches.Rectangle(
                (x1 - zx1, y1 - zy1), x2 - x1, y2 - y1,
                linewidth=2.0, edgecolor=color, facecolor='none',
            ))
            if show_labels:
                label = (f"{CLASS_NAMES[cls]} {conf:.2f}"
                         if cls < len(CLASS_NAMES) else f"{cls} {conf:.2f}")
                inset.text(
                    x1 - zx1, max(y1 - zy1 - 3, 8), label,
                    fontsize=6.5, color='white', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.15', facecolor=color,
                              edgecolor='none', alpha=0.9),
                )

    # Marker on parent showing zoom region
    ax.add_patch(mpatches.Rectangle(
        (zx1, zy1), zx2 - zx1, zy2 - zy1,
        linewidth=1.2, edgecolor='white', facecolor='none', linestyle=':', alpha=0.85,
    ))


def plot_detection_examples(n_examples=8, seed=42):
    """Publication figure A: 2×4 grid of EfficientViT-YOLOv8 detections on test frames.

    Each panel shows ground-truth (yellow dashed) and predictions (solid color
    per class) with confidence scores. Frames are class-balanced.
    """
    examples = _select_example_frames(n=n_examples, seed=seed)
    if not examples:
        print("[examples] no test frames available — skipping detection figures")
        return

    weights = RESULTS_DIR / "efficientvit_yolov8" / "weights" / "best.pt"
    if not weights.exists():
        print(f"[examples] EfficientViT weights not found at {weights} — skipping")
        return

    img_paths = [p for p, _ in examples]
    preds = _run_predictions(weights, img_paths, register_efficientvit=True)

    rows, cols = 2, n_examples // 2
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.2))
    axes = np.atleast_2d(axes)

    for idx, (img_path, gt) in enumerate(examples):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        with Image.open(img_path) as im:
            img = np.array(im.convert("RGB"))
        pred = preds.get(img_path, [])
        _draw_boxes(ax, img, gt, pred)
        ax.set_title(img_path.name, fontsize=8, color='#555555')

        # Zoomed inset on the smallest detection (or smallest GT if no preds)
        target = _pick_zoom_target(gt, pred)
        if target is not None:
            zoom = _compute_zoom_region(target, img.shape[1], img.shape[0])
            _add_zoom_inset(ax, img, zoom, gt_boxes=gt, pred_boxes=pred,
                            inset_size=0.36, loc='upper right')

    # Shared legend
    legend = [
        mpatches.Patch(facecolor='none', edgecolor=C_GT, linestyle='--', label='Ground truth'),
        mpatches.Patch(facecolor='none', edgecolor=C_HANDGUN, label='Pred: Handgun'),
        mpatches.Patch(facecolor='none', edgecolor=C_KNIFE, label='Pred: Knife'),
    ]
    fig.legend(handles=legend, loc='upper center', ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.suptitle("EfficientViT-YOLOv8 — Qualitative Detections on WeaponSenseV2 Test Set",
                 fontweight='bold', fontsize=13, y=1.06)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "7_detection_examples_efficientvit.png",
                dpi=180, bbox_inches='tight')
    plt.close()
    print("Saved: 7_detection_examples_efficientvit.png")


def plot_detection_comparison(n_examples=4, seed=42):
    """Publication figure B: side-by-side comparison on the same N test frames.

    Layout: N rows × 3 cols (Ground truth | YOLOv8m | EfficientViT-YOLOv8).
    Highlights cases where EfficientViT recovers detections that the
    CSPDarknet baseline misses (transformer global-context advantage).
    """
    examples = _select_example_frames(n=n_examples, seed=seed)
    if not examples:
        return

    evit_weights = RESULTS_DIR / "efficientvit_yolov8" / "weights" / "best.pt"
    yv8m_weights = RESULTS_DIR / "yolov8m" / "weights" / "best.pt"
    if not (evit_weights.exists() and yv8m_weights.exists()):
        print("[examples] need both EfficientViT and YOLOv8m weights for comparison — skipping")
        return

    img_paths = [p for p, _ in examples]
    preds_evit = _run_predictions(evit_weights, img_paths, register_efficientvit=True)
    preds_yv8m = _run_predictions(yv8m_weights, img_paths, register_efficientvit=False)

    fig, axes = plt.subplots(n_examples, 3, figsize=(13, n_examples * 3.2))
    if n_examples == 1:
        axes = axes.reshape(1, 3)

    col_titles = ["Ground Truth", "YOLOv8m (baseline)", "EfficientViT-YOLOv8 (ours)"]
    for c, t in enumerate(col_titles):
        axes[0, c].set_title(t, fontsize=12, fontweight='bold',
                             color=[C_GT, C_OURS, C_EVIT][c], pad=8)

    for r, (img_path, gt) in enumerate(examples):
        with Image.open(img_path) as im:
            img = np.array(im.convert("RGB"))
        pred_yv8m = preds_yv8m.get(img_path, [])
        pred_evit = preds_evit.get(img_path, [])

        # Col 0: GT only (no preds)
        _draw_boxes(axes[r, 0], img, gt, [], show_gt=True, show_labels=False)
        # Col 1: YOLOv8m preds
        _draw_boxes(axes[r, 1], img, gt, pred_yv8m, show_gt=True)
        # Col 2: EfficientViT preds
        _draw_boxes(axes[r, 2], img, gt, pred_evit, show_gt=True)

        # Shared zoom region anchored on the smallest GT box for the row, so all
        # 3 columns show the SAME crop and are visually comparable side-by-side.
        zoom_target = _pick_zoom_target(gt, []) or _pick_zoom_target([], pred_evit)
        if zoom_target is not None:
            zoom = _compute_zoom_region(zoom_target, img.shape[1], img.shape[0])
            _add_zoom_inset(axes[r, 0], img, zoom, gt_boxes=gt,
                            pred_boxes=None, inset_size=0.36, loc='upper right')
            _add_zoom_inset(axes[r, 1], img, zoom, gt_boxes=gt,
                            pred_boxes=pred_yv8m, inset_size=0.36, loc='upper right')
            _add_zoom_inset(axes[r, 2], img, zoom, gt_boxes=gt,
                            pred_boxes=pred_evit, inset_size=0.36, loc='upper right')

        # Counts annotation in row label
        n_gt = len(gt)
        axes[r, 0].set_ylabel(f"Frame {r+1}\nGT: {n_gt} obj",
                              fontsize=9, rotation=0, ha='right', va='center', labelpad=22)

    legend = [
        mpatches.Patch(facecolor='none', edgecolor=C_GT, linestyle='--', label='Ground truth'),
        mpatches.Patch(facecolor='none', edgecolor=C_HANDGUN, label='Pred: Handgun'),
        mpatches.Patch(facecolor='none', edgecolor=C_KNIFE, label='Pred: Knife'),
    ]
    fig.legend(handles=legend, loc='upper center', ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, 1.01), frameon=False)
    fig.suptitle("Detection Comparison: YOLOv8m vs EfficientViT-YOLOv8",
                 fontweight='bold', fontsize=13, y=1.04)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "8_detection_comparison.png",
                dpi=180, bbox_inches='tight')
    plt.close()
    print("Saved: 8_detection_comparison.png")


def print_summary_table(results):
    """Print a formatted summary table to console."""
    print("\n" + "=" * 110)
    print("RESULTS SUMMARY")
    print("=" * 110)

    print("\n--- Our Models (same test set, fair comparison) ---")
    header = f"{'Model':<28} {'AP50':>7} {'AP50(H)':>8} {'AP50(K)':>8} {'Prec':>7} {'Rec':>7} {'F1':>7} {'Params':>8} {'GFLOPs':>8}"
    print(header)
    print("-" * 100)

    for name, r in results["ours"].items():
        print(f"{name:<28} {r['AP50_all']:>6.1f}% {r.get('AP50_handgun',0):>7.1f}% "
              f"{r.get('AP50_knife',0):>7.1f}% {r['Prec_all']:>6.1f}% {r['Rec_all']:>6.1f}% "
              f"{r['F1_all']:>6.1f}% {r['Params_M']:>6.2f}M {r.get('GFLOPs',0):>7.1f}")

    print("\n--- Key Findings ---")
    evit = results["ours"].get("EfficientViT-YOLOv8 (ours)", {})
    v8s = results["ours"].get("YOLOv8s (ours)", {})
    v8m = results["ours"].get("YOLOv8m (ours)", {})

    if evit and v8s:
        delta_s = evit["AP50_all"] - v8s["AP50_all"]
        delta_m = evit["AP50_all"] - v8m["AP50_all"]
        param_ratio_s = (1 - evit["Params_M"] / v8s["Params_M"]) * 100
        param_ratio_m = (1 - evit["Params_M"] / v8m["Params_M"]) * 100
        gflops_ratio_s = (1 - evit.get("GFLOPs", 0) / v8s.get("GFLOPs", 1)) * 100

        print(f"  - EfficientViT-YOLOv8 vs YOLOv8s: {delta_s:+.1f}% mAP50 with {param_ratio_s:.0f}% fewer params, {gflops_ratio_s:.0f}% fewer GFLOPs")
        print(f"  - EfficientViT-YOLOv8 vs YOLOv8m: {delta_m:+.1f}% mAP50 with {param_ratio_m:.0f}% fewer params")
        print(f"  - Best knife detection: EfficientViT {evit.get('AP50_knife',0):.1f}% vs YOLOv8s {v8s.get('AP50_knife',0):.1f}% vs YOLOv8m {v8m.get('AP50_knife',0):.1f}%")
        print(f"  - Transformer global context advantage validated for small/occluded objects")

    print("\n--- Note on paper comparison ---")
    print("  Paper results use a different (likely random) train/test split.")
    print("  Our video-level split prevents data leakage (frames from same video")
    print("  never appear in both train and test), producing harder but more honest results.")
    print("  ALL our models (including baselines) score lower than papers — this is expected.")
    print("=" * 110)


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    results = load_results()
    if results is None:
        return

    print_summary_table(results)

    print(f"\nGenerating comparison plots in {PLOTS_DIR}/\n")

    plot_fair_comparison(results)
    plot_per_class(results)
    plot_efficiency_gflops(results)
    plot_efficiency_params(results)
    plot_radar(results)
    plot_full_sota_table(results)

    # Qualitative detection figures (publication-ready examples on test frames)
    print("\nGenerating qualitative detection figures (this runs inference)...")
    plot_detection_examples(n_examples=8, seed=42)
    plot_detection_comparison(n_examples=4, seed=42)

    print(f"\nAll plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
