from __future__ import annotations

import sys
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

import argparse
import json

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from plantguard_ml.models.dino_leaf_filter import DinoBinaryFilter


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Evaluate PlantGuard DINOv2 Leaf Filter V1"
    )

    parser.add_argument(
        "--data",
        default="data/processed/leaf_filter/training_dataset",
        help="Leaf filter dataset directory"
    )

    parser.add_argument(
        "--model",
        default="artifacts/dino_leaf_filter/dino_leaf_filter.pt",
        help="Trained DINOv2 leaf filter checkpoint"
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
        default=0.5,
        help="Probability threshold for positive/leaf class"
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    print("=" * 72)
    print("PlantGuard AI - Step 26")
    print("DINOv2 Leaf Filter V1 Evaluation")
    print("=" * 72)

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    data_root = ROOT / args.data
    model_path = ROOT / args.model

    test_dir = data_root / "test"

    print()
    print("Dataset :", test_dir)
    print("Model   :", model_path)

    if not test_dir.exists():
        raise FileNotFoundError(
            f"Test dataset not found:\n{test_dir}"
        )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found:\n{model_path}"
        )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print()
    print("Device :", device)

    if device.type == "cuda":
        print(
            "GPU    :",
            torch.cuda.get_device_name(0)
        )

    # --------------------------------------------------------
    # Transform
    # --------------------------------------------------------

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    ])

    # --------------------------------------------------------
    # Test Dataset
    # --------------------------------------------------------

    test_dataset = ImageFolder(
        test_dir,
        transform=test_transform
    )

    print()
    print("Classes :", test_dataset.classes)
    print("Samples :", len(test_dataset))

    print()
    print("Class mapping:")

    for index, class_name in enumerate(test_dataset.classes):
        print(f"  {index} -> {class_name}")

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )

    # --------------------------------------------------------
    # Load Model
    # --------------------------------------------------------

    print()
    print("Loading DINOv2 Leaf Filter V1...")

    checkpoint = torch.load(
        model_path,
        map_location=device
    )

    model = DinoBinaryFilter(
        backbone_name=checkpoint.get(
            "backbone_name",
            "dinov2_vits14"
        )
    )

    model.load_state_dict(
        checkpoint["state_dict"]
    )

    model = model.to(device)
    model.eval()

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # Checkpoint information
    # --------------------------------------------------------

    print()
    print("Checkpoint information:")

    for key in [
        "best_epoch",
        "best_val_accuracy",
        "best_val_loss",
        "test_accuracy",
        "test_loss",
        "image_size",
    ]:

        if key in checkpoint:
            print(
                f"  {key}: {checkpoint[key]}"
            )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    all_labels = []
    all_predictions = []
    all_probabilities = []

    print()
    print("Running evaluation...")

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(
                device,
                non_blocking=True
            )

            logits = model(images)

            probabilities = torch.sigmoid(logits)

            predictions = (
                probabilities >= args.threshold
            ).long()

            all_labels.extend(
                labels.cpu().numpy().tolist()
            )

            all_predictions.extend(
                predictions.cpu().numpy().tolist()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy().tolist()
            )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    cm = confusion_matrix(
        all_labels,
        all_predictions
    )

    # --------------------------------------------------------
    # Print Metrics
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("TEST RESULTS")
    print("=" * 72)

    print(f"Threshold : {args.threshold:.2f}")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print()
    print("Confusion Matrix")
    print("----------------")

    print(cm)

    print()
    print("Classification Report")
    print("---------------------")

    print(
        classification_report(
            all_labels,
            all_predictions,
            target_names=test_dataset.classes,
            digits=4,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Probability statistics
    # --------------------------------------------------------

    probabilities_tensor = torch.tensor(
        all_probabilities
    )

    print()
    print("Probability Statistics")
    print("----------------------")

    print(
        f"Minimum : {probabilities_tensor.min().item():.6f}"
    )

    print(
        f"Maximum : {probabilities_tensor.max().item():.6f}"
    )

    print(
        f"Mean    : {probabilities_tensor.mean().item():.6f}"
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    output_dir = (
        ROOT /
        "artifacts" /
        "dino_leaf_filter" /
        "evaluation"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    results = {

        "model": str(model_path),

        "dataset": str(test_dir),

        "device": str(device),

        "threshold": args.threshold,

        "num_test_samples": len(all_labels),

        "accuracy": float(accuracy),

        "precision": float(precision),

        "recall": float(recall),

        "f1_score": float(f1),

        "confusion_matrix": cm.tolist(),

        "classes": test_dataset.classes,

        "checkpoint": {
            key: checkpoint[key]
            for key in [
                "best_epoch",
                "best_val_accuracy",
                "best_val_loss",
                "test_accuracy",
                "test_loss",
                "image_size",
            ]
            if key in checkpoint
        }
    }

    output_file = (
        output_dir /
        "evaluation_results.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=2
        )

    print()
    print("=" * 72)
    print("STEP 26 COMPLETE")
    print("=" * 72)

    print()
    print("Evaluation saved to:")

    print(output_file)


if __name__ == "__main__":
    main()