from __future__ import annotations

import csv
import json
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "leaf_filter"

PLANTDOC_DIR = RAW_DIR / "plantdoc_hf"
PLANTVILLAGE_DIR = RAW_DIR / "plantvillage_hf"

PLANTDOC_IMAGES = PLANTDOC_DIR / "images"
PLANTVILLAGE_ZIP = PLANTVILLAGE_DIR / "data.zip"

POSITIVE_DIR = PROCESSED_DIR / "positive_pool"

PLANTDOC_OUT = POSITIVE_DIR / "plantdoc"
PLANTVILLAGE_OUT = POSITIVE_DIR / "plantvillage"

MANIFEST_DIR = PROCESSED_DIR / "manifests"
MANIFEST_FILE = MANIFEST_DIR / "positive_manifest.csv"
SUMMARY_FILE = PROCESSED_DIR / "dataset_summary.json"

# We do NOT need all 129k+ PlantVillage images for the leaf filter.
# Start with a balanced 5,000-image COLOR subset.
MAX_PLANTVILLAGE_IMAGES = 5000


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


def collect_plantdoc():
    """Reuse the PlantDoc images already exported by Step 20."""

    print("\n[1/2] Preparing PlantDoc source pool...")

    if not PLANTDOC_IMAGES.exists():
        raise FileNotFoundError(
            f"PlantDoc images not found:\n"
            f"{PLANTDOC_IMAGES}\n\n"
            f"Run Step 20 first."
        )

    files = sorted(
        p for p in PLANTDOC_IMAGES.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    print(f"PlantDoc local images found: {len(files)}")

    PLANTDOC_OUT.mkdir(
        parents=True,
        exist_ok=True
    )

    records = []

    for index, src in enumerate(files):

        output_name = (
            f"plantdoc_{index:05d}"
            f"{src.suffix.lower()}"
        )

        dst = PLANTDOC_OUT / output_name

        if not dst.exists():
            shutil.copy2(src, dst)

        records.append({
            "source": "plantdoc",
            "source_file": str(
                src.relative_to(PROJECT_ROOT)
            ),
            "output_file": str(
                dst.relative_to(PROJECT_ROOT)
            ),
            "label": "positive_source",
            "domain": "field",
        })

    return records


def find_color_members(zf):
    """
    Find actual COLOR image files inside PlantVillage data.zip.

    Expected archive structure:

        raw/color/<class>/<image>

    We do not extract the entire archive.
    """

    groups = defaultdict(list)

    for info in zf.infolist():

        if info.is_dir():
            continue

        name = info.filename.replace("\\", "/")

        parts = name.split("/")

        if len(parts) < 4:
            continue

        if parts[0] != "raw":
            continue

        if parts[1] != "color":
            continue

        if Path(name).suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        class_name = parts[2]

        groups[class_name].append(info)

    return groups


def balanced_selection(groups, maximum):
    """
    Select a deterministic, approximately balanced subset
    across PlantVillage disease/healthy classes.
    """

    class_names = sorted(groups.keys())

    if not class_names:
        raise RuntimeError(
            "No COLOR images were found inside PlantVillage data.zip."
        )

    print(
        f"\nPlantVillage COLOR classes found: "
        f"{len(class_names)}"
    )

    for class_name in class_names:
        groups[class_name].sort(
            key=lambda x: x.filename
        )

    selected = []

    # Round-robin selection keeps the subset balanced.
    position = 0

    while len(selected) < maximum:

        added_this_round = False

        for class_name in class_names:

            items = groups[class_name]

            if position < len(items):

                selected.append(
                    (class_name, items[position])
                )

                added_this_round = True

                if len(selected) >= maximum:
                    break

        if not added_this_round:
            break

        position += 1

    return selected


def extract_plantvillage():

    print("\n[2/2] Preparing PlantVillage COLOR source pool...")

    if not PLANTVILLAGE_ZIP.exists():

        raise FileNotFoundError(
            f"PlantVillage archive not found:\n"
            f"{PLANTVILLAGE_ZIP}\n\n"
            f"Download data.zip first."
        )

    print(f"Archive: {PLANTVILLAGE_ZIP}")

    print(
        f"Archive size: "
        f"{PLANTVILLAGE_ZIP.stat().st_size / (1024**3):.2f} GB"
    )

    PLANTVILLAGE_OUT.mkdir(
        parents=True,
        exist_ok=True
    )

    records = []

    with zipfile.ZipFile(
        PLANTVILLAGE_ZIP,
        "r"
    ) as zf:

        print("\nScanning archive for COLOR images...")

        groups = find_color_members(zf)

        total_color = sum(
            len(items)
            for items in groups.values()
        )

        print(
            f"COLOR images available: "
            f"{total_color}"
        )

        print(
            f"COLOR classes: "
            f"{len(groups)}"
        )

        selected = balanced_selection(
            groups,
            MAX_PLANTVILLAGE_IMAGES
        )

        print(
            f"Selected for export: "
            f"{len(selected)}"
        )

        print(
            "\nExtracting selected COLOR images..."
        )

        for index, (class_name, info) in enumerate(
            selected,
            start=1
        ):

            extension = (
                Path(info.filename)
                .suffix
                .lower()
            )

            output_name = (
                f"plantvillage_"
                f"{index:05d}"
                f"{extension}"
            )

            output_path = (
                PLANTVILLAGE_OUT /
                output_name
            )

            if not output_path.exists():

                with zf.open(info) as src, \
                        output_path.open("wb") as dst:

                    shutil.copyfileobj(
                        src,
                        dst
                    )

            records.append({
                "source": "plantvillage",
                "source_file": info.filename,
                "source_class": class_name,
                "output_file": str(
                    output_path.relative_to(
                        PROJECT_ROOT
                    )
                ),
                "label": "positive_leaf",
                "domain": "controlled",
            })

            if index % 250 == 0 or index == len(selected):

                print(
                    f"  extracted "
                    f"{index}/{len(selected)}"
                )

    return records


def write_manifest(records):

    MANIFEST_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = sorted({
        key
        for record in records
        for key in record.keys()
    })

    with MANIFEST_FILE.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for record in records:

            writer.writerow({
                key: record.get(key, "")
                for key in fieldnames
            })


def main():

    print("=" * 72)
    print("PlantGuard AI - Step 21")
    print("HF Leaf Filter Source Preparation")
    print("=" * 72)

    print("\nThis step uses:")
    print("  PlantDoc: existing 2,578 exported images")
    print("  PlantVillage: existing data.zip")
    print(
        f"  PlantVillage COLOR subset: "
        f"{MAX_PLANTVILLAGE_IMAGES} images"
    )

    print("\nThis step does NOT:")
    print("  - download PlantDoc again")
    print("  - download PlantVillage again")
    print("  - extract the entire 2.18 GB archive")
    print("  - create fake non-leaf labels")
    print("  - train DINOv2")

    # -------------------------------------------------------------
    # Create directories
    # -------------------------------------------------------------

    PLANTDOC_OUT.mkdir(
        parents=True,
        exist_ok=True
    )

    PLANTVILLAGE_OUT.mkdir(
        parents=True,
        exist_ok=True
    )

    MANIFEST_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # -------------------------------------------------------------
    # PlantDoc
    # -------------------------------------------------------------

    plantdoc_records = collect_plantdoc()

    # -------------------------------------------------------------
    # PlantVillage
    # -------------------------------------------------------------

    plantvillage_records = extract_plantvillage()

    # -------------------------------------------------------------
    # Combine
    # -------------------------------------------------------------

    records = (
        plantdoc_records +
        plantvillage_records
    )

    write_manifest(records)

    # -------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------

    summary = {
        "project": "PlantGuard AI",

        "stage": (
            "DINOv2 leaf filter "
            "positive source preparation"
        ),

        "plantdoc": {
            "count": len(plantdoc_records),
            "role": (
                "field-domain source images"
            ),
        },

        "plantvillage": {
            "count": len(plantvillage_records),
            "role": (
                "controlled-domain positive "
                "leaf examples"
            ),
            "archive": str(
                PLANTVILLAGE_ZIP.relative_to(
                    PROJECT_ROOT
                )
            ),
        },

        "total_positive_source_images": len(records),

        "negative_examples": {
            "count": 0,
            "reason": (
                "Negative examples will be generated "
                "through SAM 2.1 hard-negative mining."
            ),
        },

        "next_stage": (
            "SAM 2.1 candidate generation and "
            "hard-negative mining"
        ),
    }

    with SUMMARY_FILE.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )

    # -------------------------------------------------------------
    # Final
    # -------------------------------------------------------------

    print("\n" + "=" * 72)
    print("STEP 21 COMPLETE")
    print("=" * 72)

    print(
        f"\nPlantDoc source images: "
        f"{len(plantdoc_records)}"
    )

    print(
        f"PlantVillage COLOR images: "
        f"{len(plantvillage_records)}"
    )

    print(
        f"Total source images: "
        f"{len(records)}"
    )

    print("\nCreated:")

    print(
        f"  {PLANTDOC_OUT}"
    )

    print(
        f"  {PLANTVILLAGE_OUT}"
    )

    print(
        f"  {MANIFEST_FILE}"
    )

    print(
        f"  {SUMMARY_FILE}"
    )

    print("\nIMPORTANT:")
    print("No DINOv2 training has been performed.")
    print("No fake non-leaf labels were created.")

    print(
        "\nNext stage: "
        "SAM 2.1 candidate generation "
        "and hard-negative mining."
    )


if __name__ == "__main__":
    main()