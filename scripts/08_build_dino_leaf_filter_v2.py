from __future__ import annotations

import shutil
from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# PATHS
# ============================================================

V1_DATASET = (
    ROOT
    / "data"
    / "processed"
    / "leaf_filter"
    / "training_dataset"
)

HARD_NEGATIVE_DIR = (
    ROOT
    / "data"
    / "processed"
    / "leaf_filter"
    / "hard_negative_review"
    / "high_confidence_non_leaf"
)

V2_DATASET = (
    ROOT
    / "data"
    / "processed"
    / "leaf_filter"
    / "training_dataset_v2"
)


# ============================================================
# EXPECTED HARD NEGATIVES
# ============================================================

EXPECTED_HARD_NEGATIVES = 1000


# ============================================================
# IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}


# ============================================================
# COUNT IMAGES
# ============================================================

def get_images(directory):

    if not directory.exists():
        return []

    return [
        p
        for p in directory.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


# ============================================================
# COPY V1 DATASET
# ============================================================

def copy_v1_dataset():

    print()
    print("Copying original V1 dataset...")
    print("-" * 50)

    for split in [
        "train",
        "val",
        "test"
    ]:

        for class_name in [
            "positive",
            "negative"
        ]:

            source = (
                V1_DATASET
                / split
                / class_name
            )

            destination = (
                V2_DATASET
                / split
                / class_name
            )

            destination.mkdir(
                parents=True,
                exist_ok=True
            )

            files = get_images(source)

            for file in files:

                target = (
                    destination
                    / file.name
                )

                if not target.exists():

                    shutil.copy2(
                        file,
                        target
                    )

            print(
                f"{split}/{class_name}: "
                f"{len(files)} images"
            )


# ============================================================
# ADD VERIFIED HARD NEGATIVES
# ============================================================

def add_hard_negatives():

    files = get_images(
        HARD_NEGATIVE_DIR
    )

    print()
    print(
        "Verified hard-negative images found:",
        len(files)
    )

    if len(files) != EXPECTED_HARD_NEGATIVES:

        raise RuntimeError(
            "\nExpected exactly "
            f"{EXPECTED_HARD_NEGATIVES} "
            "verified hard negatives, but found "
            f"{len(files)}.\n\n"
            "Please check the folder before continuing."
        )

    destination = (
        V2_DATASET
        / "train"
        / "negative"
    )

    destination.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print("Adding verified hard negatives...")

    for index, file in enumerate(files):

        target = (
            destination
            / f"sam_hard_negative_{index:05d}"
            f"{file.suffix.lower()}"
        )

        shutil.copy2(
            file,
            target
        )

    print(
        "Added:",
        len(files)
    )


# ============================================================
# VERIFY DATASET
# ============================================================

def verify_dataset():

    print()
    print("=" * 72)
    print("FINAL V2 DATASET COUNTS")
    print("=" * 72)

    expected = {
        "train": {
            "positive": 6000,
            "negative": 5000
        },
        "val": {
            "positive": 750,
            "negative": 500
        },
        "test": {
            "positive": 750,
            "negative": 500
        }
    }

    success = True

    for split in [
        "train",
        "val",
        "test"
    ]:

        print()
        print(split.upper())

        for class_name in [
            "positive",
            "negative"
        ]:

            directory = (
                V2_DATASET
                / split
                / class_name
            )

            count = len(
                get_images(directory)
            )

            expected_count = (
                expected[split][class_name]
            )

            print(
                f"  {class_name:8s}: "
                f"{count}"
                f"  (expected {expected_count})"
            )

            if count != expected_count:

                success = False

    print()

    if success:

        print("✓ Dataset verification PASSED")

    else:

        print("✗ Dataset verification FAILED")

        raise RuntimeError(
            "V2 dataset counts do not match expected values."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("PlantGuard AI - Step 30A")
    print("Build DINOv2 Leaf Filter V2 Dataset")
    print("=" * 72)

    # --------------------------------------------------------
    # Validate V1 dataset
    # --------------------------------------------------------

    if not V1_DATASET.exists():

        raise FileNotFoundError(
            f"\nV1 dataset not found:\n{V1_DATASET}"
        )

    # --------------------------------------------------------
    # Validate hard-negative folder
    # --------------------------------------------------------

    if not HARD_NEGATIVE_DIR.exists():

        raise FileNotFoundError(
            f"\nVerified hard-negative folder not found:\n"
            f"{HARD_NEGATIVE_DIR}"
        )

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    hard_negative_files = get_images(
        HARD_NEGATIVE_DIR
    )

    print()
    print(
        "Verified hard negatives available:",
        len(hard_negative_files)
    )

    if len(hard_negative_files) != EXPECTED_HARD_NEGATIVES:

        raise RuntimeError(
            "\nSafety check failed.\n"
            f"Expected {EXPECTED_HARD_NEGATIVES} "
            f"images but found {len(hard_negative_files)}.\n"
            "\n"
            "The script will NOT build the dataset."
        )

    # --------------------------------------------------------
    # Build dataset
    # --------------------------------------------------------

    copy_v1_dataset()

    add_hard_negatives()

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    verify_dataset()

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("STEP 30A COMPLETE")
    print("=" * 72)

    print()
    print("V2 dataset:")
    print(V2_DATASET)

    print()
    print("Training:")
    print("  Positive : 6000")
    print("  Negative : 5000")
    print("  Total    : 11000")

    print()
    print("Validation:")
    print("  Positive : 750")
    print("  Negative : 500")
    print("  Total    : 1250")

    print()
    print("Test:")
    print("  Positive : 750")
    print("  Negative : 500")
    print("  Total    : 1250")

    print()
    print("Next step:")
    print("Train DINOv2 Leaf Filter V2.")


if __name__ == "__main__":
    main()