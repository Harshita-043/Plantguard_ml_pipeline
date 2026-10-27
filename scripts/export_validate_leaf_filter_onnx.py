from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "artifacts/dino_leaf_filter/dino_leaf_filter.pt"
OUT_DIR = ROOT / "artifacts/optimized/dino_leaf_filter/onnx"
ONNX_PATH = OUT_DIR / "dino_leaf_filter_fp32_224.onnx"
REPORT_PATH = ROOT / "reports/device_validation/dino_leaf_filter/onnx_fp32_validation.json"


class DinoBinaryFilter(nn.Module):
    def __init__(self, backbone_name="dinov2_vits14"):
        super().__init__()
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", backbone_name
        )
        dim = self.backbone.embed_dim
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1),
        )

    def forward(self, x):
        return self.head(self.backbone(x)).squeeze(1)


def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Missing checkpoint: {CHECKPOINT}")

    bundle = torch.load(
        CHECKPOINT, map_location="cpu", weights_only=False
    )
    backbone_name = bundle.get("backbone_name", "dinov2_vits14")
    image_size = int(bundle.get("image_size", 224))

    if image_size != 224:
        raise ValueError(f"Expected 224 input, found {image_size}")

    model = DinoBinaryFilter(backbone_name)
    model.load_state_dict(bundle["state_dict"], strict=True)
    model.eval().cpu()

    # Deterministic test tensor; no training or weight updates.
    rng = np.random.default_rng(20261027)
    sample = rng.normal(
        size=(1, 3, image_size, image_size)
    ).astype(np.float32)
    x = torch.from_numpy(sample)

    with torch.inference_mode():
        torch_logits = model(x).numpy()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Export to a new path only.
    torch.onnx.export(
        model,
        (x,),
        str(ONNX_PATH),
        input_names=["pixel_values"],
        output_names=["logits"],
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
    )

    graph = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(graph)

    session = ort.InferenceSession(
        str(ONNX_PATH),
        providers=["CPUExecutionProvider"],
    )

    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]
    if input_info.name != "pixel_values":
        raise RuntimeError(f"Unexpected ONNX input: {input_info.name}")
    if output_info.name != "logits":
        raise RuntimeError(f"Unexpected ONNX output: {output_info.name}")

    # Warm up CPU execution before timing.
    session.run(["logits"], {"pixel_values": sample})

    start = time.perf_counter()
    onnx_logits = session.run(
        ["logits"], {"pixel_values": sample}
    )[0]
    elapsed = time.perf_counter() - start

    torch_prob = 1.0 / (1.0 + np.exp(-torch_logits))
    onnx_prob = 1.0 / (1.0 + np.exp(-onnx_logits))

    logit_diff = np.abs(torch_logits - onnx_logits)
    prob_diff = np.abs(torch_prob - onnx_prob)

    # Strict threshold agreement is important near probability 0.5.
    torch_decision = torch_prob >= 0.5
    onnx_decision = onnx_prob >= 0.5
    decision_agreement = bool(
        np.array_equal(torch_decision, onnx_decision)
    )

    report = {
        "checkpoint": str(CHECKPOINT),
        "onnx_path": str(ONNX_PATH),
        "backbone_name": backbone_name,
        "classes": bundle.get("classes", ["negative", "positive"]),
        "image_size": image_size,
        "providers": session.get_providers(),
        "onnx_size_bytes": ONNX_PATH.stat().st_size,
        "torch_logit": torch_logits.tolist(),
        "onnx_logit": onnx_logits.tolist(),
        "torch_positive_probability": torch_prob.tolist(),
        "onnx_positive_probability": onnx_prob.tolist(),
        "max_abs_logit_difference": float(logit_diff.max()),
        "mean_abs_logit_difference": float(logit_diff.mean()),
        "max_abs_probability_difference": float(prob_diff.max()),
        "decision_threshold": 0.5,
        "decision_agreement": decision_agreement,
        "onnx_cpu_inference_seconds": elapsed,
    }

    REPORT_PATH.write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    print("Saved report:", REPORT_PATH)

    # Numerical mismatch should fail visibly rather than silently pass.
    if not np.allclose(
        torch_logits, onnx_logits, rtol=1e-3, atol=1e-4
    ):
        raise RuntimeError(
            "FP32 ONNX output differs from PyTorch beyond tolerance."
        )
    if not decision_agreement:
        raise RuntimeError(
            "PyTorch and ONNX binary decisions disagree on the test tensor."
        )

    print("FP32 ONNX VALIDATION: PASSED")


if __name__ == "__main__":
    main()
