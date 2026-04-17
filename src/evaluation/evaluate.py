"""
Evaluation script for weapon detection models.

Evaluates trained models on the test set and compares against SoTA results.
Computes: AP50, AP50:95, Precision, Recall, F1, FPS, Params, GFLOPs.

Usage:
    python evaluate.py
    python evaluate.py --model efficientvit
    python evaluate.py --model yolov8s
    python evaluate.py --model yolov8m
"""

import argparse
import json
import sys
import time
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent  # src/evaluation/ -> project root
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DATA_YAML = str(PROJECT_ROOT / "data" / "WeaponSenseV2" / "data.yaml")
RESULTS_DIR = PROJECT_ROOT / "results"

# SoTA results from the papers (WeaponSenseV2 benchmark + YOLOSR)
SOTA_RESULTS = {
    "YOLOv8n (paper)": {
        "AP50_handgun": 65.60, "AP50_knife": 21.90, "AP50_all": 43.70,
        "Prec_all": 66.30, "Rec_all": 37.40,
        "Params_M": 3.01, "GFLOPs": 8.10, "FPS_jetson": 21.7,
    },
    "YOLOv8s (paper)": {
        "AP50_handgun": 66.90, "AP50_knife": 21.30, "AP50_all": 44.10,
        "Prec_all": 55.70, "Rec_all": 40.10,
        "Params_M": 11.13, "GFLOPs": 28.40, "FPS_jetson": 9.90,
    },
    "YOLOv8m (paper)": {
        "AP50_handgun": 73.20, "AP50_knife": 29.10, "AP50_all": 51.20,
        "Prec_all": 45.70, "Rec_all": 49.70,
        "Params_M": 25.86, "GFLOPs": 79.10, "FPS_jetson": 4.40,
    },
    "YOLOv8l (paper)": {
        "AP50_handgun": 74.60, "AP50_knife": 41.10, "AP50_all": 57.80,
        "Prec_all": 68.80, "Rec_all": 50.70,
        "Params_M": 43.63, "GFLOPs": 165.40, "FPS_jetson": 2.50,
    },
    "YOLOv8xl (paper)": {
        "AP50_handgun": 78.00, "AP50_knife": 39.80, "AP50_all": 58.90,
        "Prec_all": 73.20, "Rec_all": 51.20,
        "Params_M": 68.23, "GFLOPs": 257.40, "FPS_jetson": 1.60,
    },
    "YOLOSR (paper)": {
        "AP50_handgun": 66.00, "AP50_knife": 44.20, "AP50_all": 55.10,
        "Prec_all": None, "Rec_all": None,
        "Params_M": None, "GFLOPs": 28.80, "FPS_jetson": None,
    },
    "YOLOv8mSR (paper)": {
        "AP50_handgun": 74.00, "AP50_knife": 46.00, "AP50_all": 60.00,
        "Prec_all": None, "Rec_all": None,
        "Params_M": None, "GFLOPs": 79.10, "FPS_jetson": None,
    },
}


def measure_fps(model, device, imgsz=640, n_warmup=50, n_measure=200):
    """Measure inference FPS with proper CUDA synchronization."""
    model.eval()
    dummy = torch.randn(1, 3, imgsz, imgsz).to(device)

    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_measure):
            _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    fps = n_measure / elapsed
    ms_per_img = (elapsed / n_measure) * 1000
    return fps, ms_per_img


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    return total


def compute_gflops(model, device, imgsz=640):
    try:
        from thop import profile
        dummy = torch.randn(1, 3, imgsz, imgsz).to(device)
        model.eval()
        flops, _ = profile(model, inputs=(dummy,), verbose=False)
        return flops / 1e9
    except Exception as e:
        print(f"GFLOPs computation failed: {e}")
        return None


def evaluate_yolov8_baseline(variant, device="0", imgsz=896, tta=True, iou=0.6):
    """Evaluate trained YOLOv8 baseline on test set.

    Default eval recipe (Tier-1 free gains over training-time settings):
      - imgsz=896 (>>640 train) — small objects (median ~0.1% area, ~10-16 px)
        get more feature-map signal at higher inference resolution
      - augment=True — Ultralytics test-time augmentation (multi-scale + flip)
      - iou=0.6 — looser NMS for crowded small-object scenes
    """
    from ultralytics import YOLO

    weights_path = RESULTS_DIR / variant / "weights" / "best.pt"
    if not weights_path.exists():
        for p in RESULTS_DIR.glob(f"{variant}*/weights/best.pt"):
            weights_path = p
            break

    if not weights_path.exists():
        print(f"WARNING: No weights found for {variant} at {weights_path}")
        return None

    print(f"\nEvaluating {variant} from {weights_path}")
    print(f"  imgsz={imgsz}, augment(TTA)={tta}, iou={iou}")
    model = YOLO(str(weights_path))

    # Evaluate on test set
    metrics = model.val(
        data=DATA_YAML,
        split="test",
        device=device,
        batch=1,
        imgsz=imgsz,
        augment=tta,
        iou=iou,
        verbose=True,
    )

    # Measure FPS
    dev = torch.device(f"cuda:{device}" if torch.cuda.is_available() and device != "cpu" else "cpu")
    fps, ms_per_img = measure_fps(model.model.to(dev), dev, imgsz=imgsz)

    # Model stats
    total_params = count_parameters(model.model)
    gflops = compute_gflops(model.model, dev, imgsz=imgsz)

    results = {
        "AP50_handgun": float(metrics.box.ap50[0]) * 100 if len(metrics.box.ap50) > 0 else None,
        "AP50_knife": float(metrics.box.ap50[1]) * 100 if len(metrics.box.ap50) > 1 else None,
        "AP50_all": float(metrics.box.map50) * 100,
        "AP50_95_all": float(metrics.box.map) * 100,
        "Prec_all": float(metrics.box.mp) * 100,
        "Rec_all": float(metrics.box.mr) * 100,
        "F1_all": 2 * (metrics.box.mp * metrics.box.mr) / max(metrics.box.mp + metrics.box.mr, 1e-6) * 100,
        "Params_M": total_params / 1e6,
        "GFLOPs": gflops,
        "FPS": fps,
        "ms_per_img": ms_per_img,
    }

    return results


def evaluate_efficientvit(device="0", imgsz=896, tta=True, iou=0.6):
    """Evaluate trained EfficientViT-YOLOv8 on test set.

    Uses the same Ultralytics YOLO() flow as baselines — the model was
    trained via YOLO().train() and saved as a standard checkpoint.
    Just needs custom module registration before loading.

    Eval recipe matches baselines: imgsz=896, TTA on, iou=0.6 (see baseline
    docstring for rationale).
    """
    from ultralytics import YOLO
    from models.efficientvit_modules import register_efficientvit_modules
    register_efficientvit_modules()

    weights_path = RESULTS_DIR / "efficientvit_yolov8" / "weights" / "best.pt"
    if not weights_path.exists():
        for p in RESULTS_DIR.glob("efficientvit_yolov8*/weights/best.pt"):
            weights_path = p
            break

    if not weights_path.exists():
        print(f"WARNING: No weights found at {weights_path}")
        return None

    print(f"\nEvaluating EfficientViT-YOLOv8 from {weights_path}")
    print(f"  imgsz={imgsz}, augment(TTA)={tta}, iou={iou}")
    model = YOLO(str(weights_path))

    # Evaluate on test set
    metrics = model.val(
        data=DATA_YAML,
        split="test",
        device=device,
        batch=1,
        imgsz=imgsz,
        augment=tta,
        iou=iou,
        verbose=True,
    )

    # Measure FPS
    dev = torch.device(f"cuda:{device}" if torch.cuda.is_available() and device != "cpu" else "cpu")
    fps, ms_per_img = measure_fps(model.model.to(dev), dev, imgsz=imgsz)

    # Model stats
    total_params = count_parameters(model.model)
    gflops = compute_gflops(model.model, dev, imgsz=imgsz)

    results = {
        "AP50_handgun": float(metrics.box.ap50[0]) * 100 if len(metrics.box.ap50) > 0 else None,
        "AP50_knife": float(metrics.box.ap50[1]) * 100 if len(metrics.box.ap50) > 1 else None,
        "AP50_all": float(metrics.box.map50) * 100,
        "AP50_95_all": float(metrics.box.map) * 100,
        "Prec_all": float(metrics.box.mp) * 100,
        "Rec_all": float(metrics.box.mr) * 100,
        "F1_all": 2 * (metrics.box.mp * metrics.box.mr) / max(metrics.box.mp + metrics.box.mr, 1e-6) * 100,
        "Params_M": total_params / 1e6,
        "GFLOPs": gflops,
        "FPS": fps,
        "ms_per_img": ms_per_img,
    }

    return results


def print_comparison_table(our_results):
    """Print formatted comparison table."""
    print("\n" + "=" * 100)
    print("COMPARISON TABLE: EfficientViT-YOLOv8 vs State-of-the-Art")
    print("=" * 100)

    header = f"{'Model':<28} {'AP50(H)':>8} {'AP50(K)':>8} {'AP50':>7} {'Prec':>7} {'Rec':>7} {'Params':>8} {'GFLOPs':>8} {'FPS':>8}"
    print(header)
    print("-" * 100)

    def fmt(val, suffix=""):
        return f"{val:.1f}{suffix}" if val is not None else "-"

    for name, r in SOTA_RESULTS.items():
        print(f"{name:<28} {fmt(r.get('AP50_handgun')):>8} {fmt(r.get('AP50_knife')):>8} "
              f"{fmt(r.get('AP50_all')):>7} {fmt(r.get('Prec_all')):>7} {fmt(r.get('Rec_all')):>7} "
              f"{fmt(r.get('Params_M'), 'M') if r.get('Params_M') else '-':>8} "
              f"{fmt(r.get('GFLOPs')):>8} "
              f"{fmt(r.get('FPS_jetson'), '*') if r.get('FPS_jetson') else '-':>8}")

    print("-" * 100)

    for name, r in our_results.items():
        if r is None:
            continue
        print(f"{name:<28} {fmt(r.get('AP50_handgun')):>8} {fmt(r.get('AP50_knife')):>8} "
              f"{fmt(r.get('AP50_all')):>7} {fmt(r.get('Prec_all')):>7} {fmt(r.get('Rec_all')):>7} "
              f"{fmt(r.get('Params_M'), 'M'):>8} {fmt(r.get('GFLOPs')):>8} {fmt(r.get('FPS')):>8}")

    print("-" * 100)
    print("* FPS measured on NVIDIA Jetson Nano (from papers)")
    print(f"  Our FPS measured on {'GPU' if torch.cuda.is_available() else 'CPU'}")


def save_results(our_results):
    save_path = RESULTS_DIR / "comparison_results.json"
    all_results = {"sota": SOTA_RESULTS, "ours": {k: v for k, v in our_results.items() if v}}
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate weapon detection models")
    parser.add_argument("--model", type=str, default="all",
                        choices=["efficientvit", "yolov8s", "yolov8m", "all"])
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--imgsz", type=int, default=896,
                        help="Inference resolution (default 896 for small-object boost; 640 to match training)")
    parser.add_argument("--tta", action="store_true", default=True,
                        help="Test-time augmentation (multi-scale + flip ensemble)")
    parser.add_argument("--no-tta", dest="tta", action="store_false")
    parser.add_argument("--iou", type=float, default=0.6,
                        help="NMS IoU threshold (lower = looser; 0.6 helps crowded small-obj scenes)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    our_results = {}
    eval_kwargs = dict(device=args.device, imgsz=args.imgsz, tta=args.tta, iou=args.iou)

    if args.model in ("yolov8s", "all"):
        our_results["YOLOv8s (ours)"] = evaluate_yolov8_baseline("yolov8s", **eval_kwargs)

    if args.model in ("yolov8m", "all"):
        our_results["YOLOv8m (ours)"] = evaluate_yolov8_baseline("yolov8m", **eval_kwargs)

    if args.model in ("efficientvit", "all"):
        our_results["EfficientViT-YOLOv8 (ours)"] = evaluate_efficientvit(**eval_kwargs)

    print_comparison_table(our_results)
    save_results(our_results)


if __name__ == "__main__":
    main()
