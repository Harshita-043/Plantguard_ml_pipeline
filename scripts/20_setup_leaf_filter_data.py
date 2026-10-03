from __future__ import annotations
import json, subprocess, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW = PROJECT_ROOT / "data" / "raw"
PLANTDOC_DIR = RAW / "plantdoc_hf"
PLANTVILLAGE_DIR = RAW / "plantvillage_hf"
FIELD_DIR = RAW / "plantguard_field"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "leaf_filter"


def run(cmd):
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def main():
    print("=" * 70)
    print("PlantGuard AI - Leaf Filter Data Setup")
    print("HUGGING-FACE ONLY / WINDOWS-SAFE")
    print("=" * 70)

    for p in [PLANTDOC_DIR, PLANTVILLAGE_DIR, FIELD_DIR, OUTPUT_DIR]:
        p.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
        from PIL import Image
    except ImportError:
        print("Installing required packages...")
        run([
            sys.executable, "-m", "pip", "install", "-U",
            "datasets>=4.0",
            "huggingface_hub>=0.30",
            "pillow>=10",
            "tqdm>=4.66"
        ])
        from datasets import load_dataset
        from PIL import Image

    # ---------------------------------------------------------------
    # PlantDoc FULL
    # ---------------------------------------------------------------
    print("\n[1/2] Downloading PlantDoc FULL from Hugging Face...")
    print("Dataset: geraldmc/plantdoc-full @ v0.1.0")

    ds = load_dataset(
        "geraldmc/plantdoc-full",
        revision="v0.1.0",
        split="train",
        cache_dir=str(PLANTDOC_DIR),
    )

    expected = {
        "image",
        "class_label",
        "class_idx",
        "host",
        "disease",
        "is_healthy",
        "split",
        "filename"
    }

    missing = expected - set(ds.features.keys())

    if missing:
        raise RuntimeError(
            f"Unexpected PlantDoc schema. Missing: {missing}"
        )

    print(f"PlantDoc images: {len(ds)}")

    counts = {}

    for row in ds:
        s = str(row["split"])
        counts[s] = counts.get(s, 0) + 1

    print("Original partition:", counts)

    images_dir = PLANTDOC_DIR / "images"
    images_dir.mkdir(exist_ok=True)

    manifest = PLANTDOC_DIR / "metadata.jsonl"

    with manifest.open("w", encoding="utf-8") as f:

        for i, row in enumerate(ds):

            image = row["image"].convert("RGB")

            local_name = f"plantdoc_{i:05d}.jpg"

            image.save(
                images_dir / local_name,
                quality=95
            )

            record = {
                "local_image": f"images/{local_name}",
                "class_label": str(row["class_label"]),
                "class_idx": int(row["class_idx"]),
                "host": str(row["host"]),
                "disease": str(row["disease"]),
                "is_healthy": bool(row["is_healthy"]),
                "split": str(row["split"]),
                "filename": str(row["filename"]),
            }

            f.write(json.dumps(record) + "\n")

            if (i + 1) % 250 == 0 or i + 1 == len(ds):
                print(f"  saved {i + 1}/{len(ds)}")

    with (PLANTDOC_DIR / "plantdoc_metadata.json").open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "dataset": "geraldmc/plantdoc-full",
                "revision": "v0.1.0",
                "hf_split": "train",
                "images": len(ds),
                "source_split_counts": counts,
            },
            f,
            indent=2
        )

    # ---------------------------------------------------------------
    # PlantVillage
    # ---------------------------------------------------------------
    print("\n[2/2] Loading PlantVillage from Hugging Face...")
    print("Dataset: mohanty/PlantVillage")
    print("Config: default")

    pv = load_dataset(
        "mohanty/PlantVillage",
        "default",
        cache_dir=str(PLANTVILLAGE_DIR),
    )

    pv_counts = {
        split: len(pv[split])
        for split in pv
    }

    print("PlantVillage splits:", pv_counts)

    with (
        PLANTVILLAGE_DIR / "plantvillage_metadata.json"
    ).open("w", encoding="utf-8") as f:

        json.dump(
            {
                "dataset": "mohanty/PlantVillage",
                "config": "default",
                "split_counts": pv_counts,
            },
            f,
            indent=2
        )

    # ---------------------------------------------------------------
    # Project manifest
    # ---------------------------------------------------------------
    with (
        OUTPUT_DIR / "dataset_sources.json"
    ).open("w", encoding="utf-8") as f:

        json.dump(
            {
                "plantdoc": {
                    "dataset": "geraldmc/plantdoc-full",
                    "revision": "v0.1.0",
                    "path": str(
                        PLANTDOC_DIR.relative_to(PROJECT_ROOT)
                    ),
                    "purpose": "field-domain positive examples",
                },

                "plantvillage": {
                    "dataset": "mohanty/PlantVillage",
                    "config": "default",
                    "path": str(
                        PLANTVILLAGE_DIR.relative_to(PROJECT_ROOT)
                    ),
                    "purpose": "supplemental positive leaf examples",
                },

                "field_images": {
                    "path": str(
                        FIELD_DIR.relative_to(PROJECT_ROOT)
                    ),
                    "purpose": (
                        "user field images; later "
                        "SAM2 hard-negative mining"
                    ),
                },
            },
            f,
            indent=2
        )

    print("\n" + "=" * 70)
    print("SETUP COMPLETE")
    print("=" * 70)

    print("PlantDoc:", PLANTDOC_DIR)
    print("PlantVillage:", PLANTVILLAGE_DIR)
    print("Field images:", FIELD_DIR)

    print("\nIMPORTANT:")
    print("Do NOT run 21_prepare_leaf_filter_dataset.py yet.")
    print("We will create the HF-compatible preparation step next.")
    print("Do NOT train DINOv2 yet.")


if __name__ == "__main__":
    main()