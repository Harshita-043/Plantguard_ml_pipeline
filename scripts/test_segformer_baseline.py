from pathlib import Path
import json
import time
import numpy as np
import torch
from PIL import Image
from transformers import SegformerForSemanticSegmentation

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "outputs/segformer_plantseg/best_model"
IMAGE_PATH = ROOT / "data/processed/plantseg_binary/images/test/0000000_apple_black_rot_143.jpg"
OUT_DIR = ROOT / "reports/device_validation/segformer"
OUT_DIR.mkdir(parents=True, exist_ok=True)

for p in (MODEL_DIR / "config.json", MODEL_DIR / "model.safetensors", IMAGE_PATH):
    if not p.is_file():
        raise FileNotFoundError(f"Required file not found: {p}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

with open(MODEL_DIR / "preprocessing.json", encoding="utf-8") as f:
    prep = json.load(f)

size = int(prep["size"])
mean = torch.tensor(prep["mean"], dtype=torch.float32).view(1, 3, 1, 1)
std = torch.tensor(prep["std"], dtype=torch.float32).view(1, 3, 1, 1)

image = Image.open(IMAGE_PATH).convert("RGB")
original_size = image.size
image = image.resize((size, size), Image.Resampling.BILINEAR)
arr = np.array(image, dtype=np.float32, copy=True) / 255.0
x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
x = (x - mean) / std

print("Loading SegFormer checkpoint...")
start = time.perf_counter()
model = SegformerForSemanticSegmentation.from_pretrained(
    str(MODEL_DIR), local_files_only=True
).to(device).eval()
print(f"Load time: {time.perf_counter() - start:.2f}s")
print("Model classes:", model.config.num_labels)
print("Input tensor:", tuple(x.shape))

start = time.perf_counter()
with torch.inference_mode():
    logits = model(pixel_values=x.to(device)).logits
    logits = torch.nn.functional.interpolate(
        logits, size=(size, size), mode="bilinear", align_corners=False
    )
    mask = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

print(f"Inference time: {time.perf_counter() - start:.3f}s")
print("Logits shape:", tuple(logits.shape))
print("Predicted classes:", np.unique(mask).tolist())
print("Predicted diseased pixel percentage:", round(float((mask == 1).mean() * 100), 2))

Image.fromarray(mask * 255).resize(
    original_size, Image.Resampling.NEAREST
).save(OUT_DIR / "segformer_baseline_mask.png")

np.save(OUT_DIR / "segformer_baseline_logits.npy", logits.cpu().numpy())
print("Saved baseline mask and logits to:", OUT_DIR)
print("SEGFORMER BASELINE TEST: PASSED")
