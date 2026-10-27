from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    QuantType,
    quantize_dynamic,
)

ROOT = Path.cwd()

FP32_MODEL = (
    ROOT / "artifacts/optimized/dino_disease/onnx/"
    "dino_disease_fp32_224.onnx"
)

INT8_MODEL = (
    ROOT / "artifacts/optimized/dino_disease/onnx_int8/"
    "dino_disease_dynamic_int8.onnx"
)

REPORT = (
    ROOT / "reports/device_validation/dino_disease/"
    "onnx_int8_smoke_validation.json"
)


def main():
    if not FP32_MODEL.is_file():
        raise FileNotFoundError(f"FP32 model not found: {FP32_MODEL}")

    if INT8_MODEL.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing INT8 model: {INT8_MODEL}"
        )

    INT8_MODEL.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    print("Source FP32 model:", FP32_MODEL)
    print("Source size (MiB):", round(FP32_MODEL.stat().st_size / 2**20, 2))
    print("\nQuantizing supported MatMul/Gemm weights to INT8...")

    quantize_dynamic(
        model_input=str(FP32_MODEL),
        model_output=str(INT8_MODEL),
        weight_type=QuantType.QInt8,
        per_channel=True,
        op_types_to_quantize=["MatMul", "Gemm"],
    )

    print("\nChecking quantized ONNX graph...")
    graph = onnx.load(str(INT8_MODEL))
    onnx.checker.check_model(graph)

    options = ort.SessionOptions()
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    fp32_session = ort.InferenceSession(
        str(FP32_MODEL),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )

    int8_session = ort.InferenceSession(
        str(INT8_MODEL),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )

    # Deterministic smoke test only; this is not an accuracy test.
    rng = np.random.default_rng(42)
    sample = rng.standard_normal(
        (1, 3, 224, 224)
    ).astype(np.float32)

    fp_logits, fp_embeddings = fp32_session.run(
        ["logits", "embeddings"], {"images": sample}
    )

    q_logits, q_embeddings = int8_session.run(
        ["logits", "embeddings"], {"images": sample}
    )

    logits_error = float(np.max(np.abs(fp_logits - q_logits)))
    embeddings_error = float(
        np.max(np.abs(fp_embeddings - q_embeddings))
    )

    fp_prediction = int(np.argmax(fp_logits[0]))
    q_prediction = int(np.argmax(q_logits[0]))

    fp_prob = np.exp(
        fp_logits[0] - np.max(fp_logits[0])
    )
    fp_prob /= fp_prob.sum()

    q_prob = np.exp(
        q_logits[0] - np.max(q_logits[0])
    )
    q_prob /= q_prob.sum()

    probability_error = float(np.max(np.abs(fp_prob - q_prob)))

    report = {
        "fp32_model": str(FP32_MODEL.relative_to(ROOT)),
        "int8_model": str(INT8_MODEL.relative_to(ROOT)),
        "quantization": "dynamic weight-only INT8",
        "weight_type": "QInt8",
        "quantized_operators": ["MatMul", "Gemm"],
        "fp32_size_bytes": FP32_MODEL.stat().st_size,
        "int8_size_bytes": INT8_MODEL.stat().st_size,
        "size_reduction_percent": round(
            100 * (
                1 - INT8_MODEL.stat().st_size / FP32_MODEL.stat().st_size
            ),
            3,
        ),
        "logits_max_abs_error_random_input": logits_error,
        "embedding_max_abs_error_random_input": embeddings_error,
        "probability_max_abs_error_random_input": probability_error,
        "fp32_random_input_top1": fp_prediction,
        "int8_random_input_top1": q_prediction,
        "random_input_top1_agrees": fp_prediction == q_prediction,
        "fp32_providers": fp32_session.get_providers(),
        "int8_providers": int8_session.get_providers(),
        "real_image_validation_completed": False,
        "production_approved": False,
    }

    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n========== INT8 SMOKE TEST ==========")
    print("INT8 model size (MiB):", round(INT8_MODEL.stat().st_size / 2**20, 2))
    print("Size reduction (%):", report["size_reduction_percent"])
    print("Maximum logit difference:", logits_error)
    print("Maximum embedding difference:", embeddings_error)
    print("Maximum probability difference:", probability_error)
    print("Random-input top-1 agrees:", report["random_input_top1_agrees"])
    print("Report:", REPORT)

    print(
        "\nSmoke test complete. This random-input test does not establish "
        "real-image accuracy or OOD compatibility."
    )


if __name__ == "__main__":
    main()
