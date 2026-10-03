from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path

# ============================================================
# PROJECT PATH SETUP
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

sys.path.insert(0, str(SRC))

# ============================================================
# IMPORTS
# ============================================================

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from plantguard_ml.models.dino_leaf_filter import DinoBinaryFilter


# ============================================================
# DATASET
# ============================================================

class SAMCandidateDataset(Dataset):

    def __init__(self, records, transform):

        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):

        record = self.records[idx]

        crop_path = ROOT / record["candidate_crop"]

        try:
            image = Image.open(crop_path).convert("RGB")
        except Exception as e:
            print(
                f"\nWARNING: Could not read image: {crop_path}"
            )
            print("Error:", e)

            # Return a blank image so inference can continue.
            image = Image.new(
                "RGB",
                (224, 224),
                (0, 0, 0)
            )

        image = self.transform(image)

        return image, idx


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Run DINOv2 Leaf Filter V1 on SAM candidate crops"
    )

    parser.add_argument(
        "--manifest",
        default=(
            "data/processed/leaf_filter/"
            "sam_candidates/candidates.jsonl"
        )
    )

    parser.add_argument(
        "--model",
        default=(
            "artifacts/dino_leaf_filter/"
            "dino_leaf_filter.pt"
        )
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=2
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5
    )

    return parser.parse_args()


# ============================================================
# LOAD MANIFEST
# ============================================================

def load_manifest(path):

    records = []

    with open(
        path,
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

    args = parse_args()

    print("=" * 72)
    print("PlantGuard AI - Step 27")
    print("DINOv2 V1 on SAM Candidate Crops")
    print("=" * 72)

    manifest_path = ROOT / args.manifest
    model_path = ROOT / args.model

    # --------------------------------------------------------
    # Validate paths
    # --------------------------------------------------------

    if not manifest_path.exists():

        raise FileNotFoundError(
            f"\nSAM candidate manifest not found:\n"
            f"{manifest_path}"
        )

    if not model_path.exists():

        raise FileNotFoundError(
            f"\nDINOv2 model not found:\n"
            f"{model_path}"
        )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("Device:", device)

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # --------------------------------------------------------
    # Load SAM candidate manifest
    # --------------------------------------------------------

    print()
    print("Loading SAM candidate manifest...")

    records = load_manifest(
        manifest_path
    )

    print(
        f"Candidate records: {len(records)}"
    )

    # --------------------------------------------------------
    # Transform
    #
    # SAME preprocessing used during DINOv2 evaluation.
    # --------------------------------------------------------

    transform = transforms.Compose([

        transforms.Resize(256),

        transforms.CenterCrop(224),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    ])

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = SAMCandidateDataset(
        records,
        transform
    )

    loader = DataLoader(

        dataset,

        batch_size=args.batch_size,

        shuffle=False,

        num_workers=args.num_workers,

        pin_memory=(device.type == "cuda")
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print()
    print("Loading DINOv2 Leaf Filter V1...")

    checkpoint = torch.load(
        model_path,
        map_location=device
    )

    backbone_name = checkpoint.get(
        "backbone_name",
        "dinov2_vits14"
    )

    model = DinoBinaryFilter(
        backbone_name=backbone_name
    )

    model.load_state_dict(
        checkpoint["state_dict"]
    )

    model = model.to(device)

    model.eval()

    print("Model loaded successfully.")

    print()
    print("Checkpoint:")
    print(
        "  Best epoch:",
        checkpoint.get("best_epoch")
    )
    print(
        "  Best validation accuracy:",
        checkpoint.get("best_val_accuracy")
    )
    print(
        "  Best validation loss:",
        checkpoint.get("best_val_loss")
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    output_dir = (
        ROOT
        / "artifacts"
        / "dino_leaf_filter"
        / "sam_candidate_inference"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_manifest = (
        output_dir
        / "sam_candidate_predictions.jsonl"
    )

    summary_file = (
        output_dir
        / "sam_candidate_summary.json"
    )

    # --------------------------------------------------------
    # Resume support
    # --------------------------------------------------------

    #
    # If the output file already exists, we read the candidate
    # paths that have already been processed.
    #
    processed_paths = set()

    if output_manifest.exists():

        print()
        print(
            "Existing prediction manifest found."
        )

        with open(
            output_manifest,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                try:

                    old_record = json.loads(line)

                    processed_paths.add(
                        old_record["candidate_crop"]
                    )

                except Exception:
                    continue

        print(
            "Already processed:",
            len(processed_paths)
        )

    # --------------------------------------------------------
    # Filter records that still need inference
    # --------------------------------------------------------

    pending_records = []

    for record in records:

        crop_path = record["candidate_crop"]

        if crop_path not in processed_paths:

            pending_records.append(record)

    print()
    print(
        "Total candidates:",
        len(records)
    )

    print(
        "Already processed:",
        len(processed_paths)
    )

    print(
        "Remaining:",
        len(pending_records)
    )

    if len(pending_records) == 0:

        print()
        print(
            "All candidates are already processed."
        )

    else:

        pending_dataset = SAMCandidateDataset(
            pending_records,
            transform
        )

        pending_loader = DataLoader(

            pending_dataset,

            batch_size=args.batch_size,

            shuffle=False,

            num_workers=args.num_workers,

            pin_memory=(device.type == "cuda")
        )

        # ----------------------------------------------------
        # Open output in APPEND mode
        # ----------------------------------------------------

        with open(
            output_manifest,
            "a",
            encoding="utf-8"
        ) as out:

            # ------------------------------------------------
            # Inference
            # ------------------------------------------------

            with torch.no_grad():

                progress = tqdm(
                    pending_loader,
                    total=len(pending_loader),
                    desc="DINOv2 inference"
                )

                for images, indices in progress:

                    images = images.to(
                        device,
                        non_blocking=True
                    )

                    logits = model(images)

                    probabilities = torch.sigmoid(
                        logits
                    )

                    probabilities = (
                        probabilities
                        .detach()
                        .cpu()
                        .tolist()
                    )

                    indices = (
                        indices
                        .cpu()
                        .tolist()
                    )

                    # ----------------------------------------
                    # Save every prediction immediately
                    # ----------------------------------------

                    for local_index, probability in zip(
                        indices,
                        probabilities
                    ):

                        record = pending_records[
                            local_index
                        ]

                        prediction = (
                            "positive"
                            if probability >= args.threshold
                            else "negative"
                        )

                        result = dict(record)

                        result["leaf_probability"] = float(
                            probability
                        )

                        result["prediction"] = prediction

                        result["threshold"] = (
                            args.threshold
                        )

                        out.write(
                            json.dumps(
                                result
                            )
                            + "\n"
                        )

                    # Make sure results are written
                    # even if the program is interrupted.
                    out.flush()

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("Calculating summary...")

    total = 0
    positive = 0
    negative = 0

    probabilities = []

    with open(
        output_manifest,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            total += 1

            probability = float(
                record["leaf_probability"]
            )

            probabilities.append(
                probability
            )

            if record["prediction"] == "positive":

                positive += 1

            else:

                negative += 1

    # --------------------------------------------------------
    # Confidence buckets
    # --------------------------------------------------------

    high_confidence_leaf = 0
    medium_confidence_leaf = 0
    high_confidence_non_leaf = 0
    uncertain = 0

    for p in probabilities:

        if p >= 0.90:

            high_confidence_leaf += 1

        elif p >= 0.50:

            medium_confidence_leaf += 1

        elif p <= 0.10:

            high_confidence_non_leaf += 1

        else:

            uncertain += 1

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {

        "total_candidates": total,

        "positive_leaf_predictions": positive,

        "negative_non_leaf_predictions": negative,

        "positive_percentage": (
            positive / total * 100
            if total > 0
            else 0
        ),

        "negative_percentage": (
            negative / total * 100
            if total > 0
            else 0
        ),

        "confidence_buckets": {

            "high_confidence_leaf_p_ge_0.90":
                high_confidence_leaf,

            "medium_confidence_leaf_0.50_to_0.90":
                medium_confidence_leaf,

            "high_confidence_non_leaf_p_le_0.10":
                high_confidence_non_leaf,

            "uncertain_0.10_to_0.50":
                uncertain
        },

        "threshold": args.threshold,

        "model": str(model_path),

        "source_manifest": str(manifest_path)
    }

    with open(
        summary_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()
    print("=" * 72)
    print("STEP 27 COMPLETE")
    print("=" * 72)

    print()
    print(
        f"Total candidates : {total}"
    )

    print(
        f"Leaf predictions : {positive}"
    )

    print(
        f"Non-leaf predictions : {negative}"
    )

    print()
    print("Confidence buckets:")

    print(
        "  Leaf >= 0.90 :",
        high_confidence_leaf
    )

    print(
        "  Leaf 0.50-0.90 :",
        medium_confidence_leaf
    )

    print(
        "  Non-leaf <= 0.10 :",
        high_confidence_non_leaf
    )

    print(
        "  Uncertain 0.10-0.50 :",
        uncertain
    )

    print()
    print("Predictions:")
    print(output_manifest)

    print()
    print("Summary:")
    print(summary_file)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()