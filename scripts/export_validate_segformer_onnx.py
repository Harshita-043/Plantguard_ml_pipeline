from pathlib import Path
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
import onnx
import onnxruntime as ort
from PIL import Image
from transformers import SegformerForSemanticSegmentation

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "outputs/segformer_plantseg/best_model"
IMAGE_PATH = ROOT / "data/processed/plantseg_binary/images/test/0000000_apple_black_rot_143.jpg"
OUT_DIR = ROOT / "artifacts/optimized/segformer/onnx"
REPORT_DIR = ROOT / "reports/device_validation/segformer"

OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

ONNX_PATH = OUT_DIR / "segformer_fp32_512.onnx"
REPORT_PATH = REPORT_DIR / "onnx_cpu_validation.json"

for p in (
    MODEL_DIR / "config.json",
    MODEL_DIR / "model.safetensors",
    MODEL_DIR / "preprocessing.json",
    IMAGE_PATH,
):
    if not p.is_file():
        raise FileNotFoundError(f"Required file missing: {p}")

with open(MODEL_DIR / "preprocessing.json", encoding="utf-8") as f:
    prep = json.load(f)

size = int(prep["size"])
mean_np = np.asarray(prep["mean"], dtype=np.float32).reshape(1, 3, 1, 1)
std_np = np.asarray(prep["std"], dtype=np.float32).reshape(1, 3, 1, 1)

image = Image.open(IMAGE_PATH).convert("RGB")
image = image.resize((size, size), Image.Resampling.BILINEAR)
arr = np.array(image, dtype=np.float32, copy=True) / 255.0
x_np = (arr.transpose(2, 0, 1)[None] - mean_np) / std_np
x_torch = torch.from_numpy(x_np.copy())

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Reference device:", device)
print("Export input:", x_np.shape)

model = SegformerForSemanticSegmentation.from_pretrained(
    str(MODEL_DIR), local_files_only=True
).to(device).eval()

class ExportWrapper(torch.nn.Module):
    def __init__(self, inner_model):
        super().__init__()
        self.inner_model = inner_model

    def forward(self, pixel_values):
        logits = self.inner_model(pixel_values=pixel_values).logits
        return F.interpolate(
            logits,
            size=(size, size),
            mode="bilinear",
            align_corners=False,
        )

wrapper = ExportWrapper(model).eval()
example = x_torch.to(device)

print("Exporting ONNX FP32...")
torch.onnx.export(
    wrapper,
    (example,),
    str(ONNX_PATH),
    input_names=["pixel_values"],
    output_names=["logits"],
    opset_version=18,
    do_constant_folding=True,
    dynamo=False,
)

onnx_model = onnx.load(str(ONNX_PATH))
onnx.checker.check_model(onnx_model)
print("ONNX checker: PASSED")
print("ONNX file size MB:", round(ONNX_PATH.stat().st_size / 1024**2, 2))

print("Running original PyTorch reference...")
with torch.inference_mode():
    torch_logits = wrapper(example).detach().float().cpu().numpy()

print("Running ONNX Runtime CPU...")
session = ort.InferenceSession(
    str(ONNX_PATH),
    providers=["CPUExecutionProvider"],
)
print("ONNX providers:", session.get_providers())

start = time.perf_counter()
onnx_logits = session.run(["logits"], {"pixel_values": x_np})[0]
cpu_seconds = time.perf_counter() - start

torch_mask = torch_logits.argmax(axis=1)[0]
onnx_mask = onnx_logits.argmax(axis=1)[0]

abs_diff = np.abs(torch_logits - onnx_logits)
mask_agreement = float(np.mean(torch_mask == onnx_mask))
max_abs = float(abs_diff.max())
mean_abs = float(abs_diff.mean())

report = {
    "model": "SegformerForSemanticSegmentation",
    "checkpoint": str(MODEL_DIR),
    "onnx_path": str(ONNX_PATH),
    "opset": 18,
    "input_shape": list(x_np.shape),
    "output_shape": list(onnx_logits.shape),
    "providers": session.get_providers(),
    "cpu_inference_seconds_single_run": cpu_seconds,
    "logits_max_absolute_difference": max_abs,
    "logits_mean_absolute_difference": mean_abs,
    "pixel_mask_agreement": mask_agreement,
    "pytorch_diseased_pixel_pct": float(np.mean(torch_mask == 1) * 100),
    "onnx_diseased_pixel_pct": float(np.mean(onnx_mask == 1) * 100),
    "onnx_size_bytes": ONNX_PATH.stat().st_size,
}

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

Image.fromarray((onnx_mask * 255).astype(np.uint8)).save(
    REPORT_DIR / "segformer_onnx_cpu_mask.png"
)

print("\n=== SEGFORMER ONNX CPU VALIDATION ===")
for key, value in report.items():
    print(f"{key}: {value}")

if not np.isfinite(onnx_logits).all():
    raise RuntimeError("ONNX produced non-finite logits")

if mask_agreement < 0.99:
    print("WARNING: Mask agreement is below 99%; inspect numerical differences.")
else:
    print("Mask agreement >= 99%.")

print("Report saved:", REPORT_PATH)
print("ONNX CPU validation completed.")
