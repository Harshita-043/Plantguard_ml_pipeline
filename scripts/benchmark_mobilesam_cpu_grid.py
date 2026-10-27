from pathlib import Path
import json
import time

import numpy as np
from PIL import Image
import torch
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "artifacts/mobilesam/mobile_sam.pt"
IMAGE_PATH = ROOT / "data/test_samples/plantdoc_apple_leaf.jpg"
OUT_DIR = ROOT / "reports/mobilesam_cpu"
REPORT = OUT_DIR / "reduced_grid_benchmark.json"

if not CHECKPOINT.is_file() or not IMAGE_PATH.is_file():
    raise FileNotFoundError("Checkpoint or PlantDoc test image is missing.")
if REPORT.exists():
    raise FileExistsError(f"Refusing to overwrite {REPORT}")

torch.set_num_threads(8)
image = np.asarray(Image.open(IMAGE_PATH).convert("RGB"))

model = sam_model_registry["vit_t"](checkpoint=str(CHECKPOINT))
model.to(device="cpu").eval()

results = []

for grid in [8, 4]:
    generator = SamAutomaticMaskGenerator(
        model,
        points_per_side=grid,
        points_per_batch=16,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.90,
        crop_n_layers=0,
        min_mask_region_area=100,
    )

    start = time.perf_counter()
    with torch.inference_mode():
        masks = generator.generate(image)
    elapsed = time.perf_counter() - start

    results.append({
        "points_per_side": grid,
        "prompt_points": grid * grid,
        "mask_count": len(masks),
        "latency_seconds": round(elapsed, 3),
        "top_10_mask_areas": [
            int(m["area"])
            for m in sorted(masks, key=lambda x: x["area"], reverse=True)[:10]
        ],
    })
    print(results[-1])

REPORT.write_text(json.dumps(results, indent=2), encoding="utf-8")
print("Report saved:", REPORT)
