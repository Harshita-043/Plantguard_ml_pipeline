from pathlib import Path
import sys
import time
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "artifacts/sam2/sam2.1_hiera_base_plus.pt"
CONFIG = ROOT / "external/sam2/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml"

if len(sys.argv) != 2:
    raise SystemExit('Usage: python scripts/test_sam2_cuda.py "IMAGE_PATH"')

image_path = Path(sys.argv[1])
for path in (CHECKPOINT, CONFIG, image_path):
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable")

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

print("GPU:", torch.cuda.get_device_name(0))
print("Loading SAM 2.1 Hiera Base+...")
torch.cuda.reset_peak_memory_stats()
start = time.perf_counter()

model = build_sam2(str(CONFIG), str(CHECKPOINT), device="cuda")
generator = SAM2AutomaticMaskGenerator(
    model,
    points_per_side=16,
    pred_iou_thresh=0.75,
    stability_score_thresh=0.90,
    min_mask_region_area=200,
)

torch.cuda.synchronize()
print(f"Load time: {time.perf_counter() - start:.2f} seconds")

image = np.asarray(Image.open(image_path).convert("RGB"))
print("Image dimensions:", image.shape)

start = time.perf_counter()
with torch.inference_mode():
    masks = generator.generate(image)
torch.cuda.synchronize()

print("Masks generated:", len(masks))
print(f"Inference time: {time.perf_counter() - start:.2f} seconds")
print(f"Peak allocated VRAM: {torch.cuda.max_memory_allocated()/1024**3:.2f} GiB")
print(f"Peak reserved VRAM: {torch.cuda.max_memory_reserved()/1024**3:.2f} GiB")

if masks:
    print("Largest mask area:", max(int(m["area"]) for m in masks))
    print("SAM2 GPU TEST: PASSED")
else:
    print("WARNING: No masks generated")
