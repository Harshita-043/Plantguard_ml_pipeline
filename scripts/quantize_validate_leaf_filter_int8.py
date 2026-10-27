from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import quantize_dynamic, QuantType
from PIL import Image
from torchvision import transforms
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "artifacts/dino_leaf_filter/dino_leaf_filter.pt"
DATA = ROOT / "data/processed/leaf_filter/training_dataset/test"
FP32 = ROOT / "artifacts/optimized/dino_leaf_filter/onnx/dino_leaf_filter_fp32_224.onnx"
INT8 = ROOT / "artifacts/optimized/dino_leaf_filter/onnx_int8/dino_leaf_filter_dynamic_int8.onnx"
OUT = ROOT / "reports/device_validation/dino_leaf_filter/int8_full_validation.json"

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASSES = ["negative", "positive"]


class DinoBinaryFilter(nn.Module):
    def __init__(self, backbone_name="dinov2_vits14"):
        super().__init__()
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", backbone_name
        )
        dim = self.backbone.embed_dim
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))

    def forward(self, x):
        return self.head(self.backbone(x)).squeeze(1)


def sigmoid(value):
    value = np.asarray(value, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(value, -80, 80)))


def main():
    for path in (CHECKPOINT, DATA, FP32):
        if not path.exists():
            raise FileNotFoundError(f"Missing path: {path}")

    # Create a new quantized artifact. Never overwrite FP32.
    INT8.parent.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)

    print("Quantizing dynamic MatMul/Gemm weights to INT8...")
    quantize_dynamic(
        model_input=str(FP32),
        model_output=str(INT8),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
        per_channel=True,
    )
    onnx.checker.check_model(onnx.load(str(INT8)))
    print("INT8 export:", INT8)
    print("FP32 size:", FP32.stat().st_size)
    print("INT8 size:", INT8.stat().st_size)

    # Reconstruct the unchanged original model for comparison.
    bundle = torch.load(
        CHECKPOINT, map_location="cpu", weights_only=False
    )
    classes = bundle.get("classes", CLASSES)
    if classes != CLASSES:
        raise ValueError(f"Unexpected class ordering: {classes}")

    model = DinoBinaryFilter(
        bundle.get("backbone_name", "dinov2_vits14")
    )
    model.load_state_dict(bundle["state_dict"], strict=True)
    model.eval().cpu()

    prep = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
        ),
    ])

    fp = ort.InferenceSession(
        str(FP32), providers=["CPUExecutionProvider"]
    )
    q = ort.InferenceSession(
        str(INT8), providers=["CPUExecutionProvider"]
    )

    rows = []
    for folder in sorted(p for p in DATA.iterdir() if p.is_dir()):
        if folder.name not in classes:
            print("Skipping unknown class folder:", folder.name)
            continue

        true_index = classes.index(folder.name)
        images = sorted(
            p for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in EXTENSIONS
        )

        for path in images:
            with Image.open(path) as im:
                x = prep(im.convert("RGB")).unsqueeze(0).numpy()

            # Warm-up both runtimes on this input; exclude warm-up time.
            fp.run(["logits"], {"pixel_values": x})
            q.run(["logits"], {"pixel_values": x})

            t0 = time.perf_counter()
            lf = fp.run(["logits"], {"pixel_values": x})[0]
            fp_seconds = time.perf_counter() - t0

            t0 = time.perf_counter()
            lq = q.run(["logits"], {"pixel_values": x})[0]
            q_seconds = time.perf_counter() - t0

            pf = sigmoid(lf)
            pq = sigmoid(lq)
            pred_fp = int(pf.reshape(-1)[0] >= 0.5)
            pred_q = int(pq.reshape(-1)[0] >= 0.5)

            rows.append({
                "image": str(path.relative_to(DATA)),
                "true_class": folder.name,
                "fp32_correct": pred_fp == true_index,
                "int8_correct": pred_q == true_index,
                "decision_agreement": pred_fp == pred_q,
                "fp32_probability": float(pf.reshape(-1)[0]),
                "int8_probability": float(pq.reshape(-1)[0]),
                "absolute_probability_difference": float(
                    abs(pf.reshape(-1)[0] - pq.reshape(-1)[0])
                ),
                "fp32_seconds": fp_seconds,
                "int8_seconds": q_seconds,
            })

        print(f"Finished class {folder.name}: {len(images)} images")

    if not rows:
        raise RuntimeError("No test images were found.")

    per_class = {}
    for cls in classes:
        subset = [r for r in rows if r["true_class"] == cls]
        if subset:
            per_class[cls] = {
                "samples": len(subset),
                "fp32_accuracy": float(np.mean([
                    r["fp32_correct"] for r in subset
                ])),
                "int8_accuracy": float(np.mean([
                    r["int8_correct"] for r in subset
                ])),
                "prediction_agreement": float(np.mean([
                    r["decision_agreement"] for r in subset
                ])),
            }

    fp_times = [r["fp32_seconds"] for r in rows]
    q_times = [r["int8_seconds"] for r in rows]
    diffs = [r["absolute_probability_difference"] for r in rows]

    report = {
        "checkpoint": str(CHECKPOINT),
        "fp32_path": str(FP32),
        "int8_path": str(INT8),
        "images_tested": len(rows),
        "fp32_size_bytes": FP32.stat().st_size,
        "int8_size_bytes": INT8.stat().st_size,
        "size_reduction_percent": float(
            (1 - INT8.stat().st_size / FP32.stat().st_size) * 100
        ),
        "fp32_accuracy": float(np.mean([
            r["fp32_correct"] for r in rows
        ])),
        "int8_accuracy": float(np.mean([
            r["int8_correct"] for r in rows
        ])),
        "prediction_agreement": float(np.mean([
            r["decision_agreement"] for r in rows
        ])),
        "mean_absolute_probability_difference": float(np.mean(diffs)),
        "max_absolute_probability_difference": float(max(diffs)),
        "fp32_median_seconds": float(np.median(fp_times)),
        "int8_median_seconds": float(np.median(q_times)),
        "per_class": per_class,
        "per_image": rows,
    }

    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nSUMMARY")
    print(json.dumps(
        {k: v for k, v in report.items()
         if k not in ("per_image", "per_class")} | {"per_class": per_class},
        indent=2
    ))
    print("Full report saved:", OUT)

    print(
        "INT8 decision agreement:",
        "PASSED" if report["prediction_agreement"] == 1.0 else "DIFFERENCES FOUND"
    )


if __name__ == "__main__":
    main()
