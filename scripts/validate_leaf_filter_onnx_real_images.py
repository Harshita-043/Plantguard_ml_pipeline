from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from torchvision import transforms
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "artifacts/dino_leaf_filter/dino_leaf_filter.pt"
DATA = ROOT / "data/processed/leaf_filter/training_dataset/test"
ONNX_PATH = ROOT / "artifacts/optimized/dino_leaf_filter/onnx/dino_leaf_filter_fp32_224.onnx"
OUT = ROOT / "reports/device_validation/dino_leaf_filter/real_image_validation.json"

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class DinoBinaryFilter(nn.Module):
    def __init__(self, backbone_name="dinov2_vits14"):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", backbone_name)
        dim = self.backbone.embed_dim
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))

    def forward(self, x):
        return self.head(self.backbone(x)).squeeze(1)


def main():
    for path in (CHECKPOINT, DATA, ONNX_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Missing path: {path}")

    bundle = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    classes = bundle.get("classes", ["negative", "positive"])
    if classes != ["negative", "positive"]:
        raise ValueError(f"Unexpected class order: {classes}")

    model = DinoBinaryFilter(bundle.get("backbone_name", "dinov2_vits14"))
    model.load_state_dict(bundle["state_dict"], strict=True)
    model.eval().cpu()

    # Match the existing PyTorch inference preprocessing exactly.
    prep = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
    ])

    session = ort.InferenceSession(
        str(ONNX_PATH), providers=["CPUExecutionProvider"]
    )

    folders = [p for p in DATA.iterdir() if p.is_dir()]
    if not folders:
        raise RuntimeError(
            f"Expected class subfolders under {DATA}; found none."
        )

    rows = []
    for folder in sorted(folders):
        if folder.name not in classes:
            print(f"Skipping unknown class folder: {folder.name}")
            continue

        true_index = classes.index(folder.name)
        images = sorted(
            p for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in EXTENSIONS
        )

        for path in images:
            with Image.open(path) as im:
                x = prep(im.convert("RGB")).unsqueeze(0).numpy()

            with torch.inference_mode():
                t0 = time.perf_counter()
                torch_logit = float(model(torch.from_numpy(x)).item())
                torch_seconds = time.perf_counter() - t0

            t0 = time.perf_counter()
            onnx_logit = float(
                session.run(["logits"], {"pixel_values": x})[0].reshape(-1)[0]
            )
            onnx_seconds = time.perf_counter() - t0

            torch_prob = 1.0 / (1.0 + np.exp(-torch_logit))
            onnx_prob = 1.0 / (1.0 + np.exp(-onnx_logit))

            # Existing pipeline uses sigmoid(logit) and threshold 0.5.
            torch_pred = int(torch_prob >= 0.5)
            onnx_pred = int(onnx_prob >= 0.5)

            rows.append({
                "image": str(path.relative_to(DATA)),
                "true_class": folder.name,
                "torch_prediction": classes[torch_pred],
                "onnx_prediction": classes[onnx_pred],
                "torch_correct": torch_pred == true_index,
                "onnx_correct": onnx_pred == true_index,
                "decision_agreement": torch_pred == onnx_pred,
                "torch_probability_positive": float(torch_prob),
                "onnx_probability_positive": float(onnx_prob),
                "absolute_probability_difference": float(abs(torch_prob - onnx_prob)),
                "torch_seconds": torch_seconds,
                "onnx_cpu_seconds": onnx_seconds,
            })

    if not rows:
        raise RuntimeError("No labeled images found in the test folders.")

    report = {
        "checkpoint": str(CHECKPOINT),
        "onnx_path": str(ONNX_PATH),
        "dataset": str(DATA),
        "images_tested": len(rows),
        "torch_accuracy": float(np.mean([r["torch_correct"] for r in rows])),
        "onnx_accuracy": float(np.mean([r["onnx_correct"] for r in rows])),
        "prediction_agreement": float(np.mean([r["decision_agreement"] for r in rows])),
        "mean_absolute_probability_difference": float(np.mean([
            r["absolute_probability_difference"] for r in rows
        ])),
        "max_absolute_probability_difference": float(max(
            r["absolute_probability_difference"] for r in rows
        )),
        "torch_median_seconds": float(np.median([
            r["torch_seconds"] for r in rows
        ])),
        "onnx_cpu_median_seconds": float(np.median([
            r["onnx_cpu_seconds"] for r in rows
        ])),
        "per_class": {},
        "per_image": rows,
    }

    for cls in classes:
        subset = [r for r in rows if r["true_class"] == cls]
        if subset:
            report["per_class"][cls] = {
                "samples": len(subset),
                "torch_accuracy": float(np.mean([r["torch_correct"] for r in subset])),
                "onnx_accuracy": float(np.mean([r["onnx_correct"] for r in subset])),
                "prediction_agreement": float(np.mean([
                    r["decision_agreement"] for r in subset
                ])),
            }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    summary = {k: v for k, v in report.items() if k != "per_image"}
    print(json.dumps(summary, indent=2))
    print("Saved full report:", OUT)

    if report["prediction_agreement"] != 1.0:
        print("WARNING: Some PyTorch and ONNX decisions differ.")
    else:
        print("REAL-IMAGE PREDICTION AGREEMENT: PASSED")


if __name__ == "__main__":
    main()


