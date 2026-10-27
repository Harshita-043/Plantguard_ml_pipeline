from pathlib import Path
import json
import time

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

FP32_PATH = ROOT / "artifacts/optimized/segformer/onnx/segformer_fp32_512.onnx"
INT8_DIR = ROOT / "artifacts/optimized/segformer/onnx_int8"
REPORT_DIR = ROOT / "reports/device_validation/segformer"
MODEL_DIR = ROOT / "outputs/segformer_plantseg/best_model"
IMAGE_PATH = ROOT / "data/processed/plantseg_binary/images/test/0000000_apple_black_rot_143.jpg"

INT8_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

INT8_PATH = INT8_DIR / "segformer_dynamic_int8.onnx"

for p in (FP32_PATH, IMAGE_PATH, MODEL_DIR / "preprocessing.json"):
    if not p.is_file():
        raise FileNotFoundError(f"Required file missing: {p}")

print("Creating dynamic INT8 ONNX candidate...")
quantize_dynamic(
    model_input=str(FP32_PATH),
    model_output=str(INT8_PATH),
    weight_type=QuantType.QInt8,
    op_types_to_quantize=["MatMul", "Gemm"],
    per_channel=True,
)

print("INT8 model size MB:", round(INT8_PATH.stat().st_size / 1024**2, 2))
print("Loading preprocessing settings...")

with open(MODEL_DIR / "preprocessing.json", encoding="utf-8") as f:
    prep = json.load(f)

size = int(prep["size"])
image = Image.open(IMAGE_PATH).convert("RGB")
image = image.resize((size, size), Image.Resampling.BILINEAR)

arr = np.asarray(image, dtype=np.float32) / 255.0
x = arr.transpose(2, 0, 1)[None].copy()
mean = np.asarray(prep["mean"], dtype=np.float32).reshape(1, 3, 1, 1)
std = np.asarray(prep["std"], dtype=np.float32).reshape(1, 3, 1, 1)
x = np.ascontiguousarray((x - mean) / std, dtype=np.float32)

print("Creating ONNX Runtime sessions...")
fp32_session = ort.InferenceSession(
    str(FP32_PATH), providers=["CPUExecutionProvider"]
)
int8_session = ort.InferenceSession(
    str(INT8_PATH), providers=["CPUExecutionProvider"]
)

# Warm up both sessions.
fp32_session.run(["logits"], {"pixel_values": x})
int8_session.run(["logits"], {"pixel_values": x})

def benchmark(session, repeats=10):
    times = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = session.run(["logits"], {"pixel_values": x})[0]
        times.append(time.perf_counter() - start)
    return result, {
        "median_seconds": float(np.median(times)),
        "mean_seconds": float(np.mean(times)),
        "min_seconds": float(np.min(times)),
    }

fp32_logits, fp32_time = benchmark(fp32_session)
int8_logits, int8_time = benchmark(int8_session)

fp32_mask = fp32_logits.argmax(axis=1)[0]
int8_mask = int8_logits.argmax(axis=1)[0]

report = {
    "fp32_model": str(FP32_PATH),
    "int8_model": str(INT8_PATH),
    "providers": ["CPUExecutionProvider"],
    "fp32_size_bytes": FP32_PATH.stat().st_size,
    "int8_size_bytes": INT8_PATH.stat().st_size,
    "fp32_cpu_benchmark": fp32_time,
    "int8_cpu_benchmark": int8_time,
    "logits_mean_absolute_difference": float(np.abs(fp32_logits-int8_logits).mean()),
    "logits_max_absolute_difference": float(np.abs(fp32_logits-int8_logits).max()),
    "pixel_mask_agreement": float(np.mean(fp32_mask == int8_mask)),
    "fp32_diseased_pixel_pct": float(np.mean(fp32_mask == 1) * 100),
    "int8_diseased_pixel_pct": float(np.mean(int8_mask == 1) * 100),
}

with open(REPORT_DIR / "int8_single_image_comparison.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

Image.fromarray((int8_mask * 255).astype(np.uint8)).save(
    REPORT_DIR / "segformer_int8_test_mask.png"
)

print("\n=== FP32 VS INT8 ===")
for key, value in report.items():
    print(f"{key}: {value}")

print("\nComparison saved to:", REPORT_DIR / "int8_single_image_comparison.json")
print("INT8 candidate created; release decision requires validation on multiple images.")
