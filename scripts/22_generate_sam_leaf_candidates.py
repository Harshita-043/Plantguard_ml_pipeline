from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SAM2_REPO = PROJECT_ROOT / "external" / "sam2"

CHECKPOINT = (
    PROJECT_ROOT
    / "artifacts"
    / "sam2"
    / "sam2.1_hiera_base_plus.pt"
)

MODEL_CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "leaf_filter"
    / "positive_pool"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "leaf_filter"
    / "sam_candidates"
)

CROPS_DIR = OUTPUT_DIR / "crops"

MANIFEST_FILE = OUTPUT_DIR / "candidates.jsonl"

SUMMARY_FILE = OUTPUT_DIR / "candidate_summary.json"


# ============================================================
# SETTINGS
# ============================================================

MAX_IMAGES_PER_SOURCE = 1000

# IMPORTANT:
# SAM will never receive an image larger than this dimension.
MAX_IMAGE_SIZE = 1024

POINTS_PER_SIDE = 16

PRED_IOU_THRESHOLD = 0.70

STABILITY_SCORE_THRESHOLD = 0.80

MIN_MASK_AREA = 300

MAX_MASK_AREA_RATIO = 0.85

MIN_CROP_SIZE = 32

MAX_CANDIDATES_PER_IMAGE = 100


VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# PATH CHECK
# ============================================================

def check_paths():

    if not SAM2_REPO.exists():

        raise FileNotFoundError(
            f"SAM2 repository not found:\n"
            f"{SAM2_REPO}"
        )

    if not CHECKPOINT.exists():

        raise FileNotFoundError(
            f"SAM2 checkpoint not found:\n"
            f"{CHECKPOINT}"
        )

    if not INPUT_DIR.exists():

        raise FileNotFoundError(
            f"Positive pool not found:\n"
            f"{INPUT_DIR}"
        )


# ============================================================
# LOAD SAM
# ============================================================

def load_sam():

    print("\nLoading SAM 2.1...")

    sys.path.insert(
        0,
        str(SAM2_REPO),
    )

    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import (
        SAM2AutomaticMaskGenerator,
    )

    model = build_sam2(
        MODEL_CFG,
        str(CHECKPOINT),
        device="cuda",
    )

    mask_generator = (
        SAM2AutomaticMaskGenerator(
            model,

            points_per_side=POINTS_PER_SIDE,

            pred_iou_thresh=PRED_IOU_THRESHOLD,

            stability_score_thresh=(
                STABILITY_SCORE_THRESHOLD
            ),

            min_mask_region_area=MIN_MASK_AREA,
        )
    )

    print("SAM 2.1 loaded.")

    return mask_generator


# ============================================================
# GET IMAGES
# ============================================================

def get_images():

    images = []

    for source_name in [
        "plantdoc",
        "plantvillage",
    ]:

        source_dir = (
            INPUT_DIR / source_name
        )

        if not source_dir.exists():

            continue

        files = sorted(
            p
            for p in source_dir.rglob("*")
            if (
                p.is_file()
                and p.suffix.lower()
                in VALID_EXTENSIONS
            )
        )

        files = files[
            :MAX_IMAGES_PER_SOURCE
        ]

        for file in files:

            images.append(
                (
                    source_name,
                    file,
                )
            )

    return images


# ============================================================
# RESIZE IMAGE
# ============================================================

def prepare_image(image):

    original_width, original_height = (
        image.size
    )

    longest_side = max(
        original_width,
        original_height,
    )

    if longest_side <= MAX_IMAGE_SIZE:

        return image, 1.0

    scale = (
        MAX_IMAGE_SIZE
        / float(longest_side)
    )

    new_width = max(
        1,
        int(original_width * scale),
    )

    new_height = max(
        1,
        int(original_height * scale),
    )

    resized = image.resize(
        (
            new_width,
            new_height,
        ),
        Image.Resampling.LANCZOS,
    )

    return resized, scale


# ============================================================
# MASK → CROP
# ============================================================

def crop_from_mask(
    image,
    mask,
):

    ys, xs = np.where(mask)

    if len(xs) == 0:

        return None, None

    x1 = int(xs.min())
    y1 = int(ys.min())

    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1

    if (
        x2 - x1
        < MIN_CROP_SIZE
    ):

        return None, None

    if (
        y2 - y1
        < MIN_CROP_SIZE
    ):

        return None, None

    crop = image.crop(
        (
            x1,
            y1,
            x2,
            y2,
        )
    )

    return crop, (
        x1,
        y1,
        x2,
        y2,
    )


# ============================================================
# CLEAR CUDA
# ============================================================

def cleanup_cuda():

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

        try:

            torch.cuda.ipc_collect()

        except Exception:

            pass


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)

    print(
        "PlantGuard AI - Step 22"
    )

    print(
        "SAM 2.1 Safe Candidate Generation"
    )

    print("=" * 72)

    check_paths()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CROPS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    images = get_images()

    print(
        f"\nImages selected: "
        f"{len(images)}"
    )

    print(
        f"Maximum image dimension: "
        f"{MAX_IMAGE_SIZE}"
    )

    print(
        f"Points per side: "
        f"{POINTS_PER_SIDE}"
    )

    mask_generator = load_sam()

    # --------------------------------------------------------
    # IMPORTANT
    #
    # We append to the existing JSONL instead of deleting it.
    # --------------------------------------------------------

    manifest_mode = "a"

    processed = 0

    failed = 0

    total_candidates = 0

    with MANIFEST_FILE.open(
        manifest_mode,
        encoding="utf-8",
    ) as manifest:

        for source_name, image_path in tqdm(
            images,
            desc="Generating SAM candidates",
        ):

            try:

                # ------------------------------------------------
                # Load image
                # ------------------------------------------------

                original_image = (
                    Image.open(
                        image_path
                    ).convert("RGB")
                )

                original_width, original_height = (
                    original_image.size
                )

                # ------------------------------------------------
                # Resize before SAM
                # ------------------------------------------------

                image, scale = prepare_image(
                    original_image
                )

                image_np = np.asarray(
                    image
                )

                image_area = (
                    image_np.shape[0]
                    * image_np.shape[1]
                )

                # ------------------------------------------------
                # SAM
                # ------------------------------------------------

                masks = (
                    mask_generator.generate(
                        image_np
                    )
                )

                # ------------------------------------------------
                # Highest quality first
                # ------------------------------------------------

                masks = sorted(
                    masks,
                    key=lambda x: (
                        float(
                            x.get(
                                "predicted_iou",
                                0.0,
                            )
                        ),
                        float(
                            x.get(
                                "stability_score",
                                0.0,
                            )
                        ),
                    ),
                    reverse=True,
                )

                masks = masks[
                    :MAX_CANDIDATES_PER_IMAGE
                ]

                image_candidates = 0

                # ------------------------------------------------
                # Save candidates
                # ------------------------------------------------

                for mask_index, candidate in enumerate(
                    masks
                ):

                    segmentation = (
                        candidate[
                            "segmentation"
                        ]
                    )

                    area = int(
                        candidate["area"]
                    )

                    area_ratio = (
                        area
                        / float(image_area)
                    )

                    if (
                        area_ratio
                        > MAX_MASK_AREA_RATIO
                    ):

                        continue

                    crop, bbox = (
                        crop_from_mask(
                            image,
                            segmentation,
                        )
                    )

                    if crop is None:

                        continue

                    crop_name = (
                        f"{source_name}_"
                        f"{image_path.stem}_"
                        f"mask_{mask_index:04d}.jpg"
                    )

                    crop_path = (
                        CROPS_DIR
                        / crop_name
                    )

                    crop.save(
                        crop_path,
                        quality=95,
                    )

                    record = {

                        "source":
                            source_name,

                        "source_image":
                            str(
                                image_path.relative_to(
                                    PROJECT_ROOT
                                )
                            ),

                        "candidate_crop":
                            str(
                                crop_path.relative_to(
                                    PROJECT_ROOT
                                )
                            ),

                        "mask_index":
                            mask_index,

                        "area":
                            area,

                        "area_ratio":
                            area_ratio,

                        "bbox_resized":
                            list(bbox),

                        "resize_scale":
                            scale,

                        "original_width":
                            original_width,

                        "original_height":
                            original_height,

                        "predicted_iou":
                            float(
                                candidate.get(
                                    "predicted_iou",
                                    0.0,
                                )
                            ),

                        "stability_score":
                            float(
                                candidate.get(
                                    "stability_score",
                                    0.0,
                                )
                            ),

                        "label":
                            "candidate",

                        "review_status":
                            "unreviewed",
                    }

                    manifest.write(
                        json.dumps(
                            record
                        )
                        + "\n"
                    )

                    manifest.flush()

                    image_candidates += 1

                total_candidates += (
                    image_candidates
                )

                processed += 1

                # ------------------------------------------------
                # CLEAN VRAM
                # ------------------------------------------------

                del masks
                del image_np
                del image
                del original_image

                cleanup_cuda()

            except torch.cuda.OutOfMemoryError:

                failed += 1

                print(
                    "\nCUDA OOM:"
                )

                print(
                    image_path
                )

                print(
                    "Skipping this image "
                    "and continuing."
                )

                cleanup_cuda()

                continue

            except Exception as exc:

                failed += 1

                print(
                    "\nWARNING:"
                )

                print(
                    image_path
                )

                print(
                    "Reason:",
                    exc,
                )

                cleanup_cuda()

                continue

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {

        "stage":
            "SAM 2.1 safe candidate generation",

        "model":
            "SAM 2.1 Hiera Base+",

        "images_processed":
            processed,

        "images_failed":
            failed,

        "candidate_crops":
            total_candidates,

        "parameters": {

            "max_images_per_source":
                MAX_IMAGES_PER_SOURCE,

            "max_image_size":
                MAX_IMAGE_SIZE,

            "points_per_side":
                POINTS_PER_SIDE,

            "max_candidates_per_image":
                MAX_CANDIDATES_PER_IMAGE,

            "pred_iou_threshold":
                PRED_IOU_THRESHOLD,

            "stability_threshold":
                STABILITY_SCORE_THRESHOLD,

            "minimum_mask_area":
                MIN_MASK_AREA,
        },

        "important_note":
            "SAM produces segmentation masks. "
            "Bounding boxes are derived from those masks. "
            "Candidates are not automatically classified "
            "as leaf or non-leaf.",
    }

    with SUMMARY_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    print("\n" + "=" * 72)

    print(
        "STEP 22 COMPLETE"
    )

    print("=" * 72)

    print(
        f"\nSuccessfully processed: "
        f"{processed}"
    )

    print(
        f"Skipped/failed: "
        f"{failed}"
    )

    print(
        f"Candidate crops created "
        f"this run: "
        f"{total_candidates}"
    )

    print(
        "\nExisting candidates were preserved."
    )


if __name__ == "__main__":

    main()