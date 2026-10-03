"""
Prepare PlantVillage color images for DINOv2 disease classification.

Source:
    data/raw/plantvillage_hf/data.zip

The ZIP contains:
    raw/color/<class_name>/<image>

This script:
1. Reads the ZIP without extracting the images.
2. Collects the 54k+ color images.
3. Builds the 38-class mapping.
4. Creates train/validation/test splits.
5. Saves a compact manifest.

The images remain inside the original ZIP.
"""

from pathlib import Path
import argparse
import json
import random
import zipfile


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--zip",
        default="data/raw/plantvillage_hf/data.zip"
    )

    parser.add_argument(
        "--out",
        default="data/processed/plantvillage_disease"
    )

    parser.add_argument(
        "--val",
        type=float,
        default=0.15
    )

    parser.add_argument(
        "--test",
        type=float,
        default=0.15
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    zip_path = Path(args.zip)
    out_dir = Path(args.out)

    out_dir.mkdir(parents=True, exist_ok=True)

    if not zip_path.exists():
        raise FileNotFoundError(
            f"PlantVillage ZIP not found: {zip_path}"
        )

    print("=" * 70)
    print("PlantGuard - PlantVillage Disease Dataset Preparation")
    print("=" * 70)

    print(f"ZIP: {zip_path}")
    print(f"Output: {out_dir}")
    print(f"Validation fraction: {args.val}")
    print(f"Test fraction: {args.test}")
    print(f"Seed: {args.seed}")
    print()

    if args.val + args.test >= 1.0:
        raise ValueError(
            "Validation + test fractions must be < 1.0"
        )

    rng = random.Random(args.seed)

    records = []

    print("Reading ZIP contents...")

    with zipfile.ZipFile(zip_path, "r") as z:

        for name in z.namelist():

            # We only want color images.
            if not name.startswith("raw/color/"):
                continue

            # Skip directories.
            if name.endswith("/"):
                continue

            # Supported image formats.
            if not name.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):
                continue

            parts = name.split("/")

            # Expected:
            # raw / color / CLASS / IMAGE
            if len(parts) != 4:
                continue

            class_name = parts[2]
            image_name = parts[3]

            records.append(
                {
                    "path": name,
                    "class_name": class_name,
                    "image_name": image_name,
                }
            )

    print(f"Found images: {len(records)}")

    if not records:
        raise RuntimeError(
            "No PlantVillage color images were found."
        )

    # ---------------------------------------------------------
    # Build deterministic class mapping
    # ---------------------------------------------------------

    classes = sorted(
        {r["class_name"] for r in records}
    )

    class_to_id = {
        name: idx
        for idx, name in enumerate(classes)
    }

    for record in records:
        record["class_id"] = class_to_id[
            record["class_name"]
        ]

    print(f"Number of classes: {len(classes)}")
    print()

    # ---------------------------------------------------------
    # Show class distribution
    # ---------------------------------------------------------

    class_counts = {
        class_name: 0
        for class_name in classes
    }

    for record in records:
        class_counts[record["class_name"]] += 1

    print("Class distribution:")
    print("-" * 70)

    for class_name in classes:
        print(
            f"{class_to_id[class_name]:2d} | "
            f"{class_counts[class_name]:5d} | "
            f"{class_name}"
        )

    print("-" * 70)

    # ---------------------------------------------------------
    # Split independently within each class
    # ---------------------------------------------------------
    #
    # NOTE:
    # We do not have leaf_id metadata in the actual ZIP
    # representation available locally.
    #
    # Therefore this is a stratified-by-class random split,
    # not a leaf-group split.
    # ---------------------------------------------------------

    grouped = {
        class_name: []
        for class_name in classes
    }

    for record in records:
        grouped[record["class_name"]].append(record)

    train_records = []
    val_records = []
    test_records = []

    for class_name in classes:

        items = grouped[class_name]

        rng.shuffle(items)

        n = len(items)

        n_test = int(n * args.test)

        n_val = int(n * args.val)

        test_items = items[:n_test]

        val_items = items[
            n_test:n_test + n_val
        ]

        train_items = items[
            n_test + n_val:
        ]

        train_records.extend(train_items)
        val_records.extend(val_items)
        test_records.extend(test_items)

    # Shuffle final splits.
    rng.shuffle(train_records)
    rng.shuffle(val_records)
    rng.shuffle(test_records)

    # ---------------------------------------------------------
    # Manifest
    # ---------------------------------------------------------

    manifest = {
        "dataset": "PlantVillage",
        "source_zip": str(zip_path),

        "image_root": "raw/color",

        "num_classes": len(classes),

        "classes": classes,

        "class_to_id": class_to_id,

        "seed": args.seed,

        "split_strategy": (
            "stratified_random_by_class"
        ),

        "leaf_id_available": False,

        "warning": (
            "The local ZIP representation does not expose "
            "leaf_id metadata. Therefore train/validation/test "
            "splits are stratified random splits within each class."
        ),

        "total_images": len(records),

        "train": train_records,

        "val": val_records,

        "test": test_records,
    }

    manifest_path = (
        out_dir / "manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2
        ),
        encoding="utf-8"
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    summary = {
        "total": len(records),
        "train": len(train_records),
        "val": len(val_records),
        "test": len(test_records),
        "classes": len(classes),
        "seed": args.seed,
        "split_strategy": "stratified_random_by_class",
    }

    summary_path = (
        out_dir / "summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2
        ),
        encoding="utf-8"
    )

    print()
    print("=" * 70)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 70)

    print(f"Total : {len(records)}")
    print(f"Train : {len(train_records)}")
    print(f"Val   : {len(val_records)}")
    print(f"Test  : {len(test_records)}")
    print(f"Classes: {len(classes)}")

    print()
    print(f"Manifest: {manifest_path}")
    print(f"Summary : {summary_path}")

    print()
    print("Images were NOT extracted.")
    print("The original ZIP remains untouched.")


if __name__ == "__main__":
    main()