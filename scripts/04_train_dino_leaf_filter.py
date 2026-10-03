from pathlib import Path
import sys
import argparse
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

from plantguard_ml.models.dino_leaf_filter import DinoBinaryFilter


def evaluate(model, loader, device, criterion):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.float().to(device, non_blocking=True)

            z = model(x)

            loss = criterion(z, y)

            total_loss += loss.item() * len(y)

            pred = (torch.sigmoid(z) >= 0.5)

            correct += (pred == y.bool()).sum().item()
            total += len(y)

    return total_loss / total, correct / total


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default="data/processed/leaf_filter/training_dataset",
    )

    parser.add_argument(
        "--out",
        default="artifacts/dino_leaf_filter/dino_leaf_filter.pt",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
    )

    args = parser.parse_args()

    # =========================================================
    # DEVICE
    # =========================================================

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("PlantGuard AI - DINOv2 Leaf Filter V1")
    print("=" * 70)

    print("Device:", device)

    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # =========================================================
    # DATA PATHS
    # =========================================================

    root = Path(args.data)

    train_dir = root / "train"
    val_dir = root / "val"
    test_dir = root / "test"

    if not train_dir.exists():
        raise FileNotFoundError(
            f"Training directory not found:\n{train_dir}"
        )

    if not val_dir.exists():
        raise FileNotFoundError(
            f"Validation directory not found:\n{val_dir}"
        )

    if not test_dir.exists():
        raise FileNotFoundError(
            f"Test directory not found:\n{test_dir}"
        )

    # =========================================================
    # TRANSFORMS
    # =========================================================

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(
            224,
            scale=(0.6, 1.0),
        ),

        transforms.RandomHorizontalFlip(),

        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.05,
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
        ),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize(256),

        transforms.CenterCrop(224),

        transforms.ToTensor(),

        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
        ),
    ])

    # =========================================================
    # DATASETS
    # =========================================================

    train_ds = ImageFolder(
        train_dir,
        transform=train_tf,
    )

    val_ds = ImageFolder(
        val_dir,
        transform=eval_tf,
    )

    test_ds = ImageFolder(
        test_dir,
        transform=eval_tf,
    )

    print()
    print("Classes:", train_ds.classes)

    if train_ds.classes != ["negative", "positive"]:
        print(
            "WARNING: Expected classes "
            "['negative', 'positive']"
        )

    print("Train:", len(train_ds))
    print("Validation:", len(val_ds))
    print("Test:", len(test_ds))

    # =========================================================
    # DATALOADERS
    # =========================================================

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    # =========================================================
    # MODEL
    # =========================================================

    model = DinoBinaryFilter().to(device)

    # =========================================================
    # OPTIMIZER
    # =========================================================

    # Your current design trains only the binary head.
    # DINOv2 backbone remains frozen.
    optimizer = torch.optim.AdamW(
        model.head.parameters(),
        lr=3e-4,
        weight_decay=1e-4,
    )

    criterion = nn.BCEWithLogitsLoss()

    # =========================================================
    # TRAINING
    # =========================================================

    best_val_acc = 0.0
    best_state = None

    for epoch in range(args.epochs):

        model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:

            x = x.to(
                device,
                non_blocking=True,
            )

            y = y.float().to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            z = model(x)

            loss = criterion(z, y)

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item() * len(y)
            )

            pred = (
                torch.sigmoid(z) >= 0.5
            )

            correct += (
                (pred == y.bool())
                .sum()
                .item()
            )

            total += len(y)

        train_loss = total_loss / total
        train_acc = correct / total

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            device,
            criterion,
        )

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        # Save best validation model
        if val_acc > best_val_acc:

            best_val_acc = val_acc

            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

    # =========================================================
    # RESTORE BEST MODEL
    # =========================================================

    if best_state is not None:
        model.load_state_dict(best_state)

    # =========================================================
    # FINAL TEST
    # =========================================================

    test_loss, test_acc = evaluate(
        model,
        test_loader,
        device,
        criterion,
    )

    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(
        f"Best Validation Accuracy: "
        f"{best_val_acc:.4f}"
    )

    print(
        f"Test Loss: "
        f"{test_loss:.4f}"
    )

    print(
        f"Test Accuracy: "
        f"{test_acc:.4f}"
    )

    # =========================================================
    # SAVE
    # =========================================================

    output = Path(args.out)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "state_dict": model.state_dict(),
            "backbone_name": "dinov2_vits14",
            "classes": train_ds.classes,
            "best_val_accuracy": best_val_acc,
            "test_accuracy": test_acc,
            "image_size": 224,
        },
        output,
    )

    print()
    print("Model saved:")
    print(output)


if __name__ == "__main__":
    main()