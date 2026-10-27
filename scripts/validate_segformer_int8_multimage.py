from pathlib import Path
import json
import time

import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/plantseg_binary/images/test"
PREP = ROOT / "outputs/segformer_plantseg/best_model/preprocessing.json"
FP32 = ROOT / "artifacts/optimized/segformer/onnx/segformer_fp32_512.onnx"
INT8 = ROOT / "artifacts/optimized/segformer/onnx_int8/segformer_dynamic_int8.onnx"
OUT = ROOT / "reports/device_validation/segformer/multi_image_int8.json"

for p in (DATA, PREP, FP32, INT8):
    if not p.exists():
        raise FileNotFoundError(f"Missing: {p}")

prep = json.loads(PREP.read_text(encoding="utf-8"))
size = int(prep["size"])
mean = np.array(prep["mean"], dtype=np.float32).reshape(1, 3, 1, 1)
std = np.array(prep["std"], dtype=np.float32).reshape(1, 3, 1, 1)

files = sorted(
    p for p in DATA.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
)[:20]

if not files:
    raise RuntimeError(f"No test images found in {DATA}")

fp = ort.InferenceSession(str(FP32), providers=["CPUExecutionProvider"])
q = ort.InferenceSession(str(INT8), providers=["CPUExecutionProvider"])

rows = []
for path in files:
    im = Image.open(path).convert("RGB").resize(
        (size, size), Image.Resampling.BILINEAR
    )
    a = np.asarray(im, dtype=np.float32) / 255.0
    x = (a.transpose(2, 0, 1)[None] - mean) / std
    x = np.ascontiguousarray(x, dtype=np.float32)

    # Warm-up is excluded from timing.
    fp.run(["logits"], {"pixel_values": x})
    q.run(["logits"], {"pixel_values": x})

    t0 = time.perf_counter()
    lf = fp.run(["logits"], {"pixel_values": x})[0]
    fp_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    lq = q.run(["logits"], {"pixel_values": x})[0]
    q_time = time.perf_counter() - t0

    mf = lf.argmax(axis=1)[0]
    mq = lq.argmax(axis=1)[0]

    rows.append({
        "image": path.name,
        "pixel_agreement": float(np.mean(mf == mq)),
        "mean_abs_logit_diff": float(np.abs(lf - lq).mean()),
        "fp32_seconds": fp_time,
        "int8_seconds": q_time,
        "fp32_disease_pct": float(np.mean(mf == 1) * 100),
        "int8_disease_pct": float(np.mean(mq == 1) * 100),
    })

report = {
    "images_tested": len(rows),
    "mean_pixel_agreement": float(np.mean([r["pixel_agreement"] for r in rows])),
    "minimum_pixel_agreement": float(min(r["pixel_agreement"] for r in rows)),
    "mean_logit_difference": float(np.mean([r["mean_abs_logit_diff"] for r in rows])),
    "fp32_median_seconds": float(np.median([r["fp32_seconds"] for r in rows])),
    "int8_median_seconds": float(np.median([r["int8_seconds"] for r in rows])),
    "per_image": rows,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
print("Saved:", OUT)
