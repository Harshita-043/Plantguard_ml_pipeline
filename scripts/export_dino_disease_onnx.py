from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

# Make the src-layout package importable.
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

from plantguard_ml.models.dino_classifier import DinoClassifier


CHECKPOINT = ROOT / "artifacts/dino_disease/dino_disease_classifier.pt"
OUTPUT = (
    ROOT
    / "artifacts/optimized/dino_disease/onnx/"
    / "dino_disease_fp32_224.onnx"
)
REPORT = (
    ROOT
    / "reports/device_validation/dino_disease/"
    / "onnx_fp32_export_validation.json"
)

IMAGE_SIZE = 224
OPSET = 18
ATOL = 1e-4
RTOL = 1e-4


class DiseaseClassifierExport(nn.Module):
    """
    Export both class logits and the exact backbone embeddings
    consumed by the existing Mahalanobis OOD implementation.
    """

    def __init__(self, classifier: nn.Module):
        super().__init__()
        self.classifier = classifier

    def forward(self, images):
        embeddings = self.classifier.backbone(images)
        logits = self.classifier.head(embeddings)
        return logits, embeddings


def main():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"Original classifier checkpoint not found: {CHECKPOINT}"
        )

    if OUTPUT.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing export: {OUTPUT}\n"
            "Choose a new output filename if you want another export."
        )

    print("Loading original classifier checkpoint...")
    bundle = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    required = {"state_dict", "classes", "backbone_name"}
    missing = required - set(bundle.keys())
    if missing:
        raise ValueError(
            f"Checkpoint is missing required fields: {sorted(missing)}"
        )

    classes = bundle["classes"]
    backbone_name = bundle["backbone_name"]
    image_size = int(bundle.get("image_size", IMAGE_SIZE))

    if len(classes) != 38:
        raise ValueError(
            f"Expected 38 class labels, found {len(classes)}"
        )

    labels_path = ROOT / "artifacts/dino_disease/classes.json"
    if labels_path.is_file():
        saved_classes = json.loads(labels_path.read_text(encoding="utf-8"))
        if saved_classes != classes:
            raise ValueError(
                "Checkpoint class ordering differs from classes.json. "
                "Resolve this before exporting."
            )

    print("Backbone:", backbone_name)
    print("Image size:", image_size)
    print("Number of classes:", len(classes))
    print("Embedding dimension: expected 768")

    # Instantiate the existing architecture and load the trained weights.
    classifier = DinoClassifier(
        backbone_name=backbone_name,
        num_classes=len(classes),
        freeze_backbone=True,
    )
    classifier.load_state_dict(bundle["state_dict"], strict=True)
    classifier.eval()
    classifier.cpu()

    export_model = DiseaseClassifierExport(classifier).eval()

    # Deterministic smoke-test input. Real-image validation follows later.
    torch.manual_seed(42)
    sample = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)

    with torch.inference_mode():
        torch_logits, torch_embeddings = export_model(sample)

    if tuple(torch_logits.shape) != (1, len(classes)):
        raise RuntimeError(
            f"Unexpected logits shape: {tuple(torch_logits.shape)}"
        )

    if torch_embeddings.shape[1] != 768:
        raise RuntimeError(
            f"Expected 768-dimensional embeddings, "
            f"received {torch_embeddings.shape[1]}"
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    print("\nExporting FP32 ONNX...")
    print("Output:", OUTPUT)

    torch.onnx.export(
        export_model,
        (sample,),
        str(OUTPUT),
        export_params=True,
        opset_version=OPSET,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["logits", "embeddings"],
        dynamic_axes={
            "images": {0: "batch_size"},
            "logits": {0: "batch_size"},
            "embeddings": {0: "batch_size"},
        },
        dynamo=False,
    )

    print("Checking ONNX graph...")
    graph = onnx.load(str(OUTPUT))
    onnx.checker.check_model(graph)

    print("Creating CPU ONNX Runtime session...")
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    session = ort.InferenceSession(
        str(OUTPUT),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )

    print("Runtime providers:", session.get_providers())

    # Compare logits and embeddings against the original PyTorch model.
    with torch.inference_mode():
        expected_logits, expected_embeddings = export_model(sample)

    actual_logits, actual_embeddings = session.run(
        ["logits", "embeddings"],
        {"images": sample.numpy()},
    )

    expected_logits = expected_logits.cpu().numpy()
    expected_embeddings = expected_embeddings.cpu().numpy()

    logits_abs_error = float(
        np.max(np.abs(expected_logits - actual_logits))
    )
    embedding_abs_error = float(
        np.max(np.abs(expected_embeddings - actual_embeddings))
    )

    logits_close = bool(
        np.allclose(
            expected_logits, actual_logits, atol=ATOL, rtol=RTOL
        )
    )
    embeddings_close = bool(
        np.allclose(
            expected_embeddings,
            actual_embeddings,
            atol=ATOL,
            rtol=RTOL,
        )
    )

    # Verify dynamic batch support with two samples.
    batch_test = np.concatenate(
        [sample.numpy(), sample.numpy()], axis=0
    )
    batch_logits, batch_embeddings = session.run(
        ["logits", "embeddings"],
        {"images": batch_test},
    )

    dynamic_batch_ok = (
        batch_logits.shape == (2, len(classes))
        and batch_embeddings.shape == (2, 768)
    )

    report = {
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "onnx_model": str(OUTPUT.relative_to(ROOT)),
        "backbone": backbone_name,
        "image_size": image_size,
        "num_classes": len(classes),
        "embedding_dim": 768,
        "opset": OPSET,
        "precision": "FP32",
        "providers": session.get_providers(),
        "logits_max_abs_error": logits_abs_error,
        "embeddings_max_abs_error": embedding_abs_error,
        "logits_allclose": logits_close,
        "embeddings_allclose": embeddings_close,
        "dynamic_batch_ok": bool(dynamic_batch_ok),
        "onnx_bytes": OUTPUT.stat().st_size,
        "validation_passed": bool(
            logits_close and embeddings_close and dynamic_batch_ok
        ),
        "classes": classes,
    }

    REPORT.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n========== EXPORT VALIDATION ==========")
    print("ONNX file size (MB):", round(OUTPUT.stat().st_size / 1e6, 2))
    print("Logits max absolute error:", logits_abs_error)
    print("Embeddings max absolute error:", embedding_abs_error)
    print("Logits match:", logits_close)
    print("Embeddings match:", embeddings_close)
    print("Dynamic batch works:", dynamic_batch_ok)
    print("Validation report:", REPORT)

    if not report["validation_passed"]:
        raise RuntimeError(
            "FP32 ONNX numerical validation failed. "
            "Keep the original checkpoint and do not integrate this export."
        )

    print("\nFP32 ONNX export and numerical validation PASSED.")
    print("Original checkpoint was not modified.")


if __name__ == "__main__":
    main()
