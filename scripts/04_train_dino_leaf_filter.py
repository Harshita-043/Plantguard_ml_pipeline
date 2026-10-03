from __future__ import annotations

# ============================================================
# PROJECT PATH
# ============================================================

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

sys.path.insert(0, str(SRC))


# ============================================================
# IMPORTS
# ============================================================

import argparse
import copy

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from plantguard_ml.models.dino_leaf_filter import DinoBinaryFilter


# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, loader, device, loss_fn):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.float().to(
                device,
                non_blocking=True
            )

            logits = model(images)

            loss = loss_fn(
                logits,
                labels
            )

            total_loss += loss.item() * len(labels)

            predictions = (
                torch.sigmoid(logits) >= 0.5
            )

            correct += (
                predictions == labels.bool()
            ).sum().item()

            total += len(labels)

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Train DINOv2 Leaf Filter"
    )

    parser.add_argument(
        "--data",
        default="data/processed/leaf_filter/training_dataset"
    )

    parser.add_argument(
        "--out",
        default="artifacts/dino_leaf_filter/dino_leaf_filter.pt"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=8
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
        "--lr",
        type=float,
        default=3e-4
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=3
    )

    args = parser.parse_args()

    # ========================================================
    # DEVICE
    # ========================================================

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 70)
    print("PlantGuard AI - DINOv2 Leaf Filter Training")
    print("=" * 70)

    print("\nDevice:", device)

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # ========================================================
    # DATASET PATHS
    # ========================================================

    data_root = ROOT / args.data

    train_dir = data_root / "train"
    val_dir = data_root / "val"
    test_dir = data_root / "test"

    # ========================================================
    # TRANSFORMS
    # ========================================================

    train_transform = transforms.Compose([

        transforms.RandomResizedCrop(
            224,
            scale=(0.6, 1.0)
        ),

        transforms.RandomHorizontalFlip(),

        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.05
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    ])

    eval_transform = transforms.Compose([

        transforms.Resize(256),

        transforms.CenterCrop(224),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )
    ])

    # ========================================================
    # DATASETS
    # ========================================================

    train_ds = ImageFolder(
        train_dir,
        transform=train_transform
    )

    val_ds = ImageFolder(
        val_dir,
        transform=eval_transform
    )

    test_ds = ImageFolder(
        test_dir,
        transform=eval_transform
    )

    print("\nClasses:", train_ds.classes)

    print("\nDataset sizes:")
    print("Train:", len(train_ds))
    print("Val  :", len(val_ds))
    print("Test :", len(test_ds))

    # ========================================================
    # DATALOADERS
    # ========================================================

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )

    # ========================================================
    # MODEL
    # ========================================================

    print("\nLoading DINOv2...")

    model = DinoBinaryFilter().to(device)

    # DINOv2 backbone is already frozen
    # inside DinoBinaryFilter.

    print("DINOv2 backbone : FROZEN")
    print("Classification head : TRAINABLE")

    # ========================================================
    # LOSS
    # ========================================================

    loss_fn = nn.BCEWithLogitsLoss()

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.head.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    # ========================================================
    # BEST MODEL TRACKING
    # ========================================================

    best_val_accuracy = -1.0
    best_val_loss = float("inf")
    best_epoch = 0

    best_state = None

    epochs_without_improvement = 0

    # ========================================================
    # TRAINING
    # ========================================================

    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    for epoch in range(args.epochs):

        model.train()

        # Keep frozen DINOv2 backbone in eval mode
        model.backbone.eval()

        running_loss = 0.0

        correct = 0
        total = 0

        # ----------------------------------------------------
        # TRAINING LOOP
        # ----------------------------------------------------

        for images, labels in train_loader:

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.float().to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad()

            logits = model(images)

            loss = loss_fn(
                logits,
                labels
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item() * len(labels)
            )

            predictions = (
                torch.sigmoid(logits) >= 0.5
            )

            correct += (
                predictions == labels.bool()
            ).sum().item()

            total += len(labels)

        train_loss = running_loss / total
        train_accuracy = correct / total

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            device,
            loss_fn
        )

        print(
            f"Epoch {epoch + 1:02d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.4f}"
        )

        # ====================================================
        # BEST MODEL CRITERION
        # ====================================================
        #
        # Priority:
        # 1. Higher validation accuracy
        # 2. If accuracy is tied, lower validation loss
        #
        # ====================================================

        is_better = (

            val_accuracy > best_val_accuracy

            or (

                val_accuracy == best_val_accuracy

                and val_loss < best_val_loss
            )
        )

        # ----------------------------------------------------
        # IMPROVEMENT
        # ----------------------------------------------------

        if is_better:

            best_val_accuracy = val_accuracy
            best_val_loss = val_loss
            best_epoch = epoch + 1

            best_state = copy.deepcopy(
                model.state_dict()
            )

            epochs_without_improvement = 0

            print(
                f"  ✓ Best model updated | "
                f"Epoch: {best_epoch} | "
                f"Val Acc: {best_val_accuracy:.4f} | "
                f"Val Loss: {best_val_loss:.4f}"
            )

        # ----------------------------------------------------
        # NO IMPROVEMENT
        # ----------------------------------------------------

        else:

            epochs_without_improvement += 1

            print(
                f"  No improvement "
                f"({epochs_without_improvement}/"
                f"{args.patience})"
            )

        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if epochs_without_improvement >= args.patience:

            print()
            print(
                f"Early stopping triggered after "
                f"{args.patience} epochs without improvement."
            )

            break

    # ========================================================
    # RESTORE BEST MODEL
    # ========================================================

    if best_state is not None:

        model.load_state_dict(
            best_state
        )

    # ========================================================
    # TEST
    # ========================================================

    test_loss, test_accuracy = evaluate(
        model,
        test_loader,
        device,
        loss_fn
    )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(
        f"Best epoch              : {best_epoch}"
    )

    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy:.4f}"
    )

    print(
        f"Best validation loss    : "
        f"{best_val_loss:.4f}"
    )

    print(
        f"Test accuracy           : "
        f"{test_accuracy:.4f}"
    )

    print(
        f"Test loss               : "
        f"{test_loss:.4f}"
    )

    # ========================================================
    # SAVE BEST CHECKPOINT
    # ========================================================

    output_path = ROOT / args.out

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    checkpoint = {

        # Model
        "state_dict":
            model.state_dict(),

        # Architecture
        "backbone_name":
            "dinov2_vits14",

        # Dataset classes
        "classes":
            train_ds.classes,

        # Training information
        "best_epoch":
            best_epoch,

        "best_val_accuracy":
            best_val_accuracy,

        "best_val_loss":
            best_val_loss,

        # Final test information
        "test_accuracy":
            test_accuracy,

        "test_loss":
            test_loss,

        # Input information
        "image_size":
            224
    }

    torch.save(
        checkpoint,
        output_path
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\nModel saved to:")
    print(output_path)

    print("\n" + "=" * 70)
    print("STEP 25 COMPLETE")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()