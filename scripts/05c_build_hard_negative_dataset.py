from __future__ import annotations

import sys
import json
import csv
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


# ============================================================
# CONFIG
# ============================================================

PREDICTIONS = (
    ROOT
    / "artifacts"
    / "dino_leaf_filter"
    / "sam_candidate_inference"
    / "sam_candidate_predictions.jsonl"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "leaf_filter"
    / "hard_negative_review"
)

NUM_HIGH_CONFIDENCE = 1000
NUM_UNCERTAIN = 500

RANDOM_SEED = 42


# ============================================================
# LOAD PREDICTIONS
# ============================================================

def load_predictions():

    records = []

    with open(
        PREDICTIONS,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            records.append(
                json.loads(line)
            )

    return records


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("PlantGuard AI - Step 29")
    print("Hard-Negative Review Dataset")
    print("=" * 72)

    if not PREDICTIONS.exists():

        raise FileNotFoundError(
            f"Prediction file not found:\n{PREDICTIONS}"
        )

    random.seed(RANDOM_SEED)

    records = load_predictions()

    print()
    print("Total prediction records:", len(records))

    # --------------------------------------------------------
    # Split by confidence
    # --------------------------------------------------------

    high_conf_negative = [
        r for r in records
        if float(r["leaf_probability"]) <= 0.10
    ]

    uncertain = [
        r for r in records
        if 0.10 < float(r["leaf_probability"]) < 0.50
    ]

    print()
    print(
        "High-confidence non-leaf:",
        len(high_conf_negative)
    )

    print(
        "Uncertain:",
        len(uncertain)
    )

    # --------------------------------------------------------
    # Random sample
    # --------------------------------------------------------

    high_sample = random.sample(
        high_conf_negative,
        min(
            NUM_HIGH_CONFIDENCE,
            len(high_conf_negative)
        )
    )

    uncertain_sample = random.sample(
        uncertain,
        min(
            NUM_UNCERTAIN,
            len(uncertain)
        )
    )

    # --------------------------------------------------------
    # Create folders
    # --------------------------------------------------------

    high_dir = (
        OUTPUT_DIR
        / "high_confidence_non_leaf"
    )

    uncertain_dir = (
        OUTPUT_DIR
        / "uncertain"
    )

    high_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    uncertain_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Copy candidates
    # --------------------------------------------------------

    def copy_records(
        selected_records,
        destination,
        category
    ):

        rows = []

        for i, record in enumerate(
            selected_records
        ):

            source_path = (
                ROOT
                / record["candidate_crop"]
            )

            if not source_path.exists():

                print(
                    "WARNING: missing crop:",
                    source_path
                )

                continue

            destination_name = (
                f"{category}_{i:05d}"
                + source_path.suffix
            )

            destination_path = (
                destination
                / destination_name
            )

            shutil.copy2(
                source_path,
                destination_path
            )

            rows.append({

                "review_id": destination_name,

                "candidate_crop":
                    record["candidate_crop"],

                "source_image":
                    record.get("source_image", ""),

                "leaf_probability":
                    record["leaf_probability"],

                "model_prediction":
                    record["prediction"],

                "human_label": "",

            })

        return rows

    print()
    print("Copying high-confidence candidates...")

    high_rows = copy_records(
        high_sample,
        high_dir,
        "high_non_leaf"
    )

    print(
        "Copied:",
        len(high_rows)
    )

    print()
    print("Copying uncertain candidates...")

    uncertain_rows = copy_records(
        uncertain_sample,
        uncertain_dir,
        "uncertain"
    )

    print(
        "Copied:",
        len(uncertain_rows)
    )

    # --------------------------------------------------------
    # Create review CSV
    # --------------------------------------------------------

    review_rows = (
        high_rows
        + uncertain_rows
    )

    review_csv = (
        OUTPUT_DIR
        / "review_manifest.csv"
    )

    with open(
        review_csv,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "review_id",
                "candidate_crop",
                "source_image",
                "leaf_probability",
                "model_prediction",
                "human_label",
            ]
        )

        writer.writeheader()

        writer.writerows(
            review_rows
        )

    # --------------------------------------------------------
    # Instructions
    # --------------------------------------------------------

    instructions = (
        OUTPUT_DIR
        / "README.txt"
    )

    with open(
        instructions,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            """
PLANTGUARD DINOv2 V2 HARD-NEGATIVE REVIEW
==========================================

You must manually inspect the images.

For every image, fill the "human_label" column
in review_manifest.csv.

Allowed labels:

leaf
non_leaf
uncertain

IMPORTANT:

Do NOT automatically label all model predictions
as ground truth.

For DINOv2 predictions:

high_confidence_non_leaf:
    model probability <= 0.10

uncertain:
    model probability between 0.10 and 0.50

The final hard-negative dataset should contain ONLY
images that you manually verified as:

    non_leaf

Do not include uncertain or actual leaf images
as hard negatives.

After completing the CSV, use the verified rows
to create the V2 training dataset.
"""
        )

    print()
    print("=" * 72)
    print("STEP 29A COMPLETE")
    print("=" * 72)

    print()
    print("Review folder:")
    print(OUTPUT_DIR)

    print()
    print("Review CSV:")
    print(review_csv)

    print()
    print("IMPORTANT:")
    print(
        "Manually label human_label as "
        "leaf / non_leaf / uncertain."
    )


if __name__ == "__main__":
    main()