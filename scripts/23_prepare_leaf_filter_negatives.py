from __future__ import annotations

import json
import random
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "coco_leaf_filter"
)

IMAGE_DIR = (
    RAW_DIR / "val2017"
)

ANNOTATION_DIR = (
    RAW_DIR / "annotations"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "leaf_filter"
    / "normal_negatives"
)

IMAGE_ZIP = (
    RAW_DIR / "val2017.zip"
)

ANNOTATION_ZIP = (
    RAW_DIR / "annotations_trainval2017.zip"
)

ANNOTATION_JSON = (
    ANNOTATION_DIR
    / "instances_val2017.json"
)


IMAGE_URL = (
    "http://images.cocodataset.org/"
    "zips/val2017.zip"
)

ANNOTATION_URL = (
    "http://images.cocodataset.org/"
    "annotations/"
    "annotations_trainval2017.zip"
)


# ============================================================
# NEGATIVE CATEGORIES
# ============================================================

NEGATIVE_CATEGORIES = {

    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",

    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",

    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",

    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",

    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",

    "chair",
    "couch",
    "bed",
    "dining table",
    "toilet",

    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",

    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",

    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
}


# ============================================================
# IMPORTANT
# ============================================================

# Do NOT use plant/fruit/vegetable classes
# as ordinary negatives.

EXCLUDED_PLANT_CATEGORIES = {
    "potted plant",
    "banana",
    "apple",
    "orange",
    "broccoli",
    "carrot",
}


MAX_NEGATIVES = 5000

SEED = 42


# ============================================================
# DOWNLOAD
# ============================================================

def download_file(
    url: str,
    destination: Path,
):

    if destination.exists():

        print(
            f"Already exists: "
            f"{destination}"
        )

        return

    print(
        f"\nDownloading:\n{url}"
    )

    urllib.request.urlretrieve(
        url,
        destination,
    )

    print(
        f"Saved:\n{destination}"
    )


# ============================================================
# EXTRACT
# ============================================================

def extract_zip(
    zip_path: Path,
    destination: Path,
):

    print(
        f"\nExtracting "
        f"{zip_path.name}..."
    )

    with zipfile.ZipFile(
        zip_path,
        "r",
    ) as archive:

        archive.extractall(
            destination
        )

    print(
        "Extraction complete."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)

    print(
        "PlantGuard AI - Step 23"
    )

    print(
        "Normal Negative Dataset Preparation"
    )

    print("=" * 72)

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Download images
    # --------------------------------------------------------

    download_file(
        IMAGE_URL,
        IMAGE_ZIP,
    )

    # --------------------------------------------------------
    # Download annotations
    # --------------------------------------------------------

    download_file(
        ANNOTATION_URL,
        ANNOTATION_ZIP,
    )

    # --------------------------------------------------------
    # Extract images
    # --------------------------------------------------------

    if not IMAGE_DIR.exists():

        extract_zip(
            IMAGE_ZIP,
            RAW_DIR,
        )

    # --------------------------------------------------------
    # Extract annotations
    # --------------------------------------------------------

    if not ANNOTATION_JSON.exists():

        extract_zip(
            ANNOTATION_ZIP,
            RAW_DIR,
        )

    if not ANNOTATION_JSON.exists():

        raise FileNotFoundError(
            f"Could not find:\n"
            f"{ANNOTATION_JSON}"
        )

    # --------------------------------------------------------
    # Load COCO
    # --------------------------------------------------------

    print(
        "\nLoading annotations..."
    )

    with ANNOTATION_JSON.open(
        "r",
        encoding="utf-8",
    ) as f:

        coco = json.load(f)

    categories = {

        category["id"]:
            category["name"]

        for category
        in coco["categories"]
    }

    images = {

        image["id"]:
            image

        for image
        in coco["images"]
    }

    # --------------------------------------------------------
    # Find suitable object annotations
    # --------------------------------------------------------

    candidates = []

    for annotation in (
        coco["annotations"]
    ):

        category = categories[
            annotation["category_id"]
        ]

        if category not in NEGATIVE_CATEGORIES:

            continue

        if category in EXCLUDED_PLANT_CATEGORIES:

            continue

        x, y, w, h = (
            annotation["bbox"]
        )

        if w < 40 or h < 40:

            continue

        image_info = images[
            annotation["image_id"]
        ]

        candidates.append(
            {
                "image_id":
                    annotation["image_id"],

                "file_name":
                    image_info["file_name"],

                "category":
                    category,

                "bbox":
                    [x, y, w, h],
            }
        )

    print(
        f"\nAvailable negative "
        f"object annotations: "
        f"{len(candidates)}"
    )

    random.seed(SEED)

    random.shuffle(
        candidates
    )

    candidates = candidates[
        :MAX_NEGATIVES
    ]

    # --------------------------------------------------------
    # Prepare output
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = []

    # --------------------------------------------------------
    # Create object crops
    # --------------------------------------------------------

    for index, item in enumerate(
        tqdm(
            candidates,
            desc="Creating negative crops",
        )
    ):

        image_path = (
            IMAGE_DIR
            / item["file_name"]
        )

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

            x, y, w, h = (
                item["bbox"]
            )

            x1 = max(
                0,
                int(x),
            )

            y1 = max(
                0,
                int(y),
            )

            x2 = min(
                image.width,
                int(x + w),
            )

            y2 = min(
                image.height,
                int(y + h),
            )

            if x2 <= x1:
                continue

            if y2 <= y1:
                continue

            crop = image.crop(
                (
                    x1,
                    y1,
                    x2,
                    y2,
                )
            )

            if crop.width < 32:
                continue

            if crop.height < 32:
                continue

            safe_category = (
                item["category"]
                .replace(
                    " ",
                    "_",
                )
            )

            filename = (
                f"coco_"
                f"{safe_category}_"
                f"{index:06d}.jpg"
            )

            output_path = (
                OUTPUT_DIR
                / filename
            )

            crop.save(
                output_path,
                quality=95,
            )

            manifest.append(
                {
                    "path":
                        str(
                            output_path.relative_to(
                                PROJECT_ROOT
                            )
                        ),

                    "label":
                        "negative",

                    "source":
                        "COCO",

                    "category":
                        item["category"],

                    "original_image":
                        item["file_name"],
                }
            )

        except Exception as exc:

            print(
                f"\nWARNING: "
                f"{image_path}"
            )

            print(
                "Reason:",
                exc,
            )

    # --------------------------------------------------------
    # Save manifest
    # --------------------------------------------------------

    manifest_path = (
        OUTPUT_DIR
        / "manifest.json"
    )

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            manifest,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Category statistics
    # --------------------------------------------------------

    category_counts = {}

    for item in manifest:

        category = item[
            "category"
        ]

        category_counts[
            category
        ] = (
            category_counts.get(
                category,
                0,
            )
            + 1
        )

    summary = {

        "stage":
            "Normal negative preparation",

        "dataset":
            "COCO 2017 validation set",

        "negative_count":
            len(manifest),

        "categories":
            category_counts,

        "label_policy":
            "COCO object crops belonging to "
            "non-plant categories are normal "
            "non-leaf negatives.",

        "sam_candidates_used":
            False,

        "reason":
            "SAM candidates are not "
            "automatically assumed to be "
            "non-leaf.",
    }

    with (
        OUTPUT_DIR
        / "summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    print("\n" + "=" * 72)

    print(
        "STEP 23 COMPLETE"
    )

    print("=" * 72)

    print(
        f"\nNegative crops created: "
        f"{len(manifest)}"
    )

    print(
        "\nOutput:"
    )

    print(
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()