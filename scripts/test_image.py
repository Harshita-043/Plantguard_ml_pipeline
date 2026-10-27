
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plantguard_ml.models.sam2_adapter import SAM2LeafSegmenter

CONFIG_PATH = ROOT / "configs" / "plantguard.yaml"
IMAGE_PATH = (
    ROOT / "data" / "raw" / "plantdoc_hf" / "images" / "plantdoc_00001.jpg"
)


def main():
    print("Loading configuration...")
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    checkpoint = ROOT / cfg["paths"]["sam2_checkpoint"]
    sam_cfg = ROOT / cfg["sam2"]["config"]

    print("Image:", IMAGE_PATH)
    print("Checkpoint:", checkpoint)
    print("SAM config:", sam_cfg)
    print("CUDA available:", torch.cuda.is_available())

    for path in (IMAGE_PATH, checkpoint, sam_cfg):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    image = Image.open(IMAGE_PATH).convert("RGB")
    image_rgb = np.asarray(image)

    print("Image dimensions (width, height):", image.size)
    print("Loading SAM 2.1...")
    model = SAM2LeafSegmenter(
        checkpoint=str(checkpoint),
        config=str(sam_cfg),
        device=cfg["project"]["device"],
    )

    print("Generating candidate masks...")
    candidates = model.candidate_masks(
        image_rgb,
        min_area_ratio=cfg["sam2"]["min_area_ratio"],
        max_area_ratio=cfg["sam2"]["max_area_ratio"],
    )

    print("Candidate masks:", len(candidates))
    for i, candidate in enumerate(candidates[:5]):
        mask = candidate["mask"]
        print(
            f"Mask {i}: shape={mask.shape}, "
            f"area={int(mask.sum())}, "
            f"score={candidate['score']:.3f}, "
            f"stability={candidate['stability']:.3f}, "
            f"bbox={candidate['bbox']}"
        )
        assert mask.shape == image_rgb.shape[:2], "Mask dimensions mismatch"

    if candidates:
        print("SAM 2.1 single-image test PASSED.")
    else:
        print(
            "SAM loaded, but no candidates passed the configured area filters. "
            "This is not necessarily a model-loading failure."
        )


if __name__ == "__main__":
    main()
