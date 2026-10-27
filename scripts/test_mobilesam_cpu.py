from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "artifacts/mobilesam/mobile_sam.pt"
IMAGE_PATH = ROOT / "data/test_samples/plantdoc_apple_leaf.jpg"
OUT_DIR = ROOT / "reports/mobilesam_cpu"
OUT_DIR.mkdir(parents=True, exist_ok=True)

if not CHECKPOINT.is_file():
    raise FileNotFoundError(CHECKPOINT)
if not IMAGE_PATH.is_file():
    raise FileNotFoundError(IMAGE_PATH)

torch.set_num_threads(8)

image = np.asarray(Image.open(IMAGE_PATH).convert("RGB"))
print("Image shape:", image.shape)
print("Device: CPU")
print("PyTorch threads:", torch.get_num_threads())

model = sam_model_registry["vit_t"](checkpoint=str(CHECKPOINT))
model.to(device="cpu")
model.eval()

generator = SamAutomaticMaskGenerator(
    model,
    points_per_side=16,
    pred_iou_thresh=0.86,
    stability_score_thresh=0.90,
    crop_n_layers=0,
    min_mask_region_area=100,
)

start = time.perf_counter()
with torch.inference_mode():
    masks = generator.generate(image)
elapsed = time.perf_counter() - start

print("Masks generated:", len(masks))
print("Inference seconds:", round(elapsed, 3))

overlay = Image.fromarray(image).convert("RGBA")
mask_layer = Image.new("RGBA", overlay.size, (0, 0, 0, 0))
draw = ImageDraw.Draw(mask_layer)

rng = np.random.default_rng(42)
for item in sorted(masks, key=lambda m: m["area"], reverse=True):
    color = tuple(int(v) for v in rng.integers(40, 256, size=3))
    binary = item["segmentation"]
    layer = Image.new("RGBA", overlay.size, color + (0,))
    alpha = Image.fromarray((binary.astype(np.uint8) * 90), mode="L")
    layer.putalpha(alpha)
    mask_layer = Image.alpha_composite(mask_layer, layer)

overlay = Image.alpha_composite(overlay, mask_layer).convert("RGB")
output = OUT_DIR / "plantdoc_mobilesam_overlay.jpg"
overlay.save(output, quality=92)

print("Overlay saved:", output)
print("Top mask areas:", [m["area"] for m in sorted(
    masks, key=lambda m: m["area"], reverse=True
)[:10]])
