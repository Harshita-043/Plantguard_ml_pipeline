from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "artifacts/mobilesam/mobile_sam.pt"
IMAGE_PATH = ROOT / "data/test_samples/plantdoc_apple_leaf.jpg"
OUT_DIR = ROOT / "reports/mobilesam_cpu/grid_visuals"

if not CHECKPOINT.is_file() or not IMAGE_PATH.is_file():
    raise FileNotFoundError("MobileSAM checkpoint or test image is missing.")

OUT_DIR.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(8)
image = np.asarray(Image.open(IMAGE_PATH).convert("RGB"))

model = sam_model_registry["vit_t"](checkpoint=str(CHECKPOINT))
model.to("cpu").eval()

for grid in (8, 4):
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

    # Color each mask independently; draw larger masks first.
    overlay = Image.fromarray(image).convert("RGBA")
    for i, item in enumerate(sorted(masks, key=lambda m: m["area"], reverse=True)):
        rng = np.random.default_rng(1000 + i)
        color = tuple(int(x) for x in rng.integers(40, 256, size=3))
        rgba = np.zeros((*item["segmentation"].shape, 4), dtype=np.uint8)
        rgba[item["segmentation"]] = (*color, 95)
        overlay = Image.alpha_composite(
            overlay, Image.fromarray(rgba, mode="RGBA")
        )

    output = OUT_DIR / f"mobilesam_grid_{grid}x{grid}.jpg"
    overlay.convert("RGB").save(output, quality=92)
    print(
        f"Grid {grid}x{grid}: masks={len(masks)}, "
        f"seconds={elapsed:.3f}, output={output}"
    )
