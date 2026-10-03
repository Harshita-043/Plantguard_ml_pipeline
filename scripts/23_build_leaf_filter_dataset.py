from __future__ import annotations

import csv
import json
import random
import shutil
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# INPUT DATA
# ============================================================

POSITIVE_POOL = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "leaf_filter"
    / "positive_pool"
)

NEGATIVE_POOL = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "leaf_filter"
    / "normal_negatives"
)


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "leaf_filter"
    / "training_dataset"
)

TRAIN_DIR = OUTPUT_DIR / "train"
VAL_DIR = OUTPUT_DIR / "val"
TEST_DIR = OUTPUT_DIR / "test"

MANIFEST_FILE = (
    OUTPUT_DIR
    / "manifest.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "dataset_summary.json"
)


# ============================================================
# DATASET LIMITS
# ============================================================

MAX_PLANTDOC = 2500

MAX_PLANTVILLAGE = 5000

MAX_NEGATIVES = 7500


# ============================================================
# SPLIT
# ============================================================

TRAIN_RATIO = 0.80

VAL_RATIO = 0.10

TEST_RATIO = 0.10

SEED = 42


VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# COLLECT IMAGES
# ============================================================

def collect_images(
    directory: Path,
):

    if not directory.exists():

        return []

    return [
        p
        for p in directory.rglob("*")
        if (
            p.is_file()
            and p.suffix.lower()
            in VALID_EXTENSIONS
        )
    ]


# ============================================================
# VERIFY IMAGE
# ============================================================

def verify_image(
    path: Path,
):

    try:

        with Image.open(path) as image:

            image.verify()

        return True

    except Exception:

        return False


# ============================================================
# SPLIT
# ============================================================

def split_dataset(
    files,
):

    files = list(files)

    random.shuffle(
        files
    )

    n = len(files)

    train_end = int(
        n * TRAIN_RATIO
    )

    val_end = (
        train_end
        + int(
            n * VAL_RATIO
        )
    )

    train = files[
        :train_end
    ]

    val = files[
        train_end:val_end
    ]

    test = files[
        val_end:
    ]

    return (
        train,
        val,
        test,
    )


# ============================================================
# CLEAN OUTPUT
# ============================================================

def prepare_output():

    if OUTPUT_DIR.exists():

        print(
            f"\nRemoving previous "
            f"training dataset:\n"
            f"{OUTPUT_DIR}"
        )

        shutil.rmtree(
            OUTPUT_DIR
        )

    for split_dir in [
        TRAIN_DIR,
        VAL_DIR,
        TEST_DIR,
    ]:

        for label in [
            "positive",
            "negative",
        ]:

            (
                split_dir / label
            ).mkdir(
                parents=True,
                exist_ok=True,
            )


# ============================================================
# COPY FILES
# ============================================================

def copy_files(
    files,
    split,
    label,
    source,
):

    output_dir = (
        OUTPUT_DIR
        / split
        / label
    )

    records = []

    for index, source_path in enumerate(
        files
    ):

        filename = (
            f"{source}_"
            f"{index:06d}"
            f"{source_path.suffix.lower()}"
        )

        destination = (
            output_dir
            / filename
        )

        shutil.copy2(
            source_path,
            destination,
        )

        records.append(
            {
                "path":
                    str(
                        destination.relative_to(
                            PROJECT_ROOT
                        )
                    ),

                "label":
                    label,

                "source":
                    source,

                "original_path":
                    str(
                        source_path.relative_to(
                            PROJECT_ROOT
                        )
                    ),

                "split":
                    split,
            }
        )

    return records


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)

    print(
        "PlantGuard AI - Step 24"
    )

    print(
        "Build DINOv2 Leaf Filter Dataset"
    )

    print("=" * 72)

    random.seed(
        SEED
    )

    prepare_output()

    # ========================================================
    # PLANTDOC
    # ========================================================

    plantdoc = [
        p
        for p in collect_images(
            POSITIVE_POOL
            / "plantdoc"
        )
        if verify_image(p)
    ]

    random.shuffle(
        plantdoc
    )

    plantdoc = plantdoc[
        :MAX_PLANTDOC
    ]

    print(
        f"\nPlantDoc positives: "
        f"{len(plantdoc)}"
    )

    # ========================================================
    # PLANTVILLAGE
    # ========================================================

    plantvillage = [
        p
        for p in collect_images(
            POSITIVE_POOL
            / "plantvillage"
        )
        if verify_image(p)
    ]

    random.shuffle(
        plantvillage
    )

    plantvillage = plantvillage[
        :MAX_PLANTVILLAGE
    ]

    print(
        f"PlantVillage positives: "
        f"{len(plantvillage)}"
    )

    # ========================================================
    # NEGATIVES
    # ========================================================

    negatives = [
        p
        for p in collect_images(
            NEGATIVE_POOL
        )
        if verify_image(p)
    ]

    random.shuffle(
        negatives
    )

    negatives = negatives[
        :MAX_NEGATIVES
    ]

    print(
        f"Normal negatives: "
        f"{len(negatives)}"
    )

    if not negatives:

        raise RuntimeError(
            "\nNo normal negatives found.\n\n"
            "Run:\n"
            "python scripts/"
            "23_prepare_leaf_filter_negatives.py"
        )

    # ========================================================
    # POSITIVE POOL
    # ========================================================

    positives = (
        plantdoc
        +
        plantvillage
    )

    random.shuffle(
        positives
    )

    # ========================================================
    # SPLIT
    # ========================================================

    (
        positive_train,
        positive_val,
        positive_test,
    ) = split_dataset(
        positives
    )

    (
        negative_train,
        negative_val,
        negative_test,
    ) = split_dataset(
        negatives
    )

    # ========================================================
    # COPY POSITIVES
    # ========================================================

    records = []

    records.extend(
        copy_files(
            positive_train,
            "train",
            "positive",
            "plant_positive",
        )
    )

    records.extend(
        copy_files(
            positive_val,
            "val",
            "positive",
            "plant_positive",
        )
    )

    records.extend(
        copy_files(
            positive_test,
            "test",
            "positive",
            "plant_positive",
        )
    )

    # ========================================================
    # COPY NEGATIVES
    # ========================================================

    records.extend(
        copy_files(
            negative_train,
            "train",
            "negative",
            "coco",
        )
    )

    records.extend(
        copy_files(
            negative_val,
            "val",
            "negative",
            "coco",
        )
    )

    records.extend(
        copy_files(
            negative_test,
            "test",
            "negative",
            "coco",
        )
    )

    # ========================================================
    # MANIFEST
    # ========================================================

    fields = [
        "path",
        "label",
        "source",
        "original_path",
        "split",
    ]

    with MANIFEST_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        writer.writerows(
            records
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {

        "stage":
            "DINOv2 Leaf Filter V1 dataset",

        "seed":
            SEED,

        "positive": {

            "plantdoc":
                len(plantdoc),

            "plantvillage":
                len(plantvillage),

            "total":
                len(positives),
        },

        "negative": {

            "normal_negative":
                len(negatives),

            "total":
                len(negatives),
        },

        "splits": {

            "train": {

                "positive":
                    len(positive_train),

                "negative":
                    len(negative_train),
            },

            "validation": {

                "positive":
                    len(positive_val),

                "negative":
                    len(negative_val),
            },

            "test": {

                "positive":
                    len(positive_test),

                "negative":
                    len(negative_test),
            },
        },

        "sam_candidates": {

            "used_for_v1_training":
                False,

            "reason":
                "SAM candidates are reserved "
                "for later hard-negative mining.",
        },

        "label_definition": {

            "positive":
                "PlantDoc and PlantVillage "
                "leaf-positive images.",

            "negative":
                "Explicit non-plant/object "
                "crops from COCO.",

            "candidate":
                "SAM segmentation region that "
                "has not yet been semantically "
                "classified.",
        },

        "next_stage":
            "Train DINOv2 Leaf Filter V1, "
            "then perform SAM candidate "
            "hard-negative mining.",
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

    # ========================================================
    # REPORT
    # ========================================================

    print("\n" + "=" * 72)

    print(
        "STEP 24 COMPLETE"
    )

    print("=" * 72)

    print(
        "\nTRAIN"
    )

    print(
        f"  Positive: "
        f"{len(positive_train)}"
    )

    print(
        f"  Negative: "
        f"{len(negative_train)}"
    )

    print(
        "\nVALIDATION"
    )

    print(
        f"  Positive: "
        f"{len(positive_val)}"
    )

    print(
        f"  Negative: "
        f"{len(negative_val)}"
    )

    print(
        "\nTEST"
    )

    print(
        f"  Positive: "
        f"{len(positive_test)}"
    )

    print(
        f"  Negative: "
        f"{len(negative_test)}"
    )

    print(
        "\nDataset:"
    )

    print(
        OUTPUT_DIR
    )

    print(
        "\nManifest:"
    )

    print(
        MANIFEST_FILE
    )

    print(
        "\nSAM candidates were NOT "
        "automatically labeled as negatives."
    )


if __name__ == "__main__":
    main()