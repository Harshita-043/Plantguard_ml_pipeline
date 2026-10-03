from __future__ import annotations

import sys
from pathlib import Path

# ------------------------------------------------------------------
# Project root / src-layout support
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ------------------------------------------------------------------
# Standard imports
# ------------------------------------------------------------------
import argparse
import json
import random
import zipfile
import yaml

import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from plantguard_ml.models.dino_classifier import DinoClassifier, save_bundle


class PlantVillageZipDataset(Dataset):
    """
    Reads PlantVillage images directly from data.zip.

    No image extraction is required.
    """

    def __init__(self, zip_path, records, train=False):
        self.zip_path = str(zip_path)
        self.records = records

        if train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(
                    224,
                    scale=(0.65, 1.0)
                ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(p=0.15),
                transforms.ColorJitter(
                    0.25, 0.25, 0.25, 0.05
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.485, 0.456, 0.406),
                    (0.229, 0.224, 0.225)
                ),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.485, 0.456, 0.406),
                    (0.229, 0.224, 0.225)
                ),
            ])

        self._zip = None

    def _get_zip(self):
        """
        Open one ZIP handle per DataLoader worker/process.
        """
        if self._zip is None:
            self._zip = zipfile.ZipFile(
                self.zip_path,
                "r"
            )
        return self._zip

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]

        z = self._get_zip()

        image_bytes = z.read(record["path"])

        from io import BytesIO

        image = Image.open(
            BytesIO(image_bytes)
        ).convert("RGB")

        image = self.transform(image)

        label = int(record["class_id"])

        return image, label


def evaluate(model, loader, device):
    model.eval()

    total = 0
    correct = 0

    with torch.no_grad():
        for images, labels in loader:

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            logits = model(images)

            predictions = logits.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    return correct / max(total, 1)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/plantguard.yaml"
    )

    parser.add_argument(
        "--manifest",
        default="data/processed/plantvillage_disease/manifest.json"
    )

    parser.add_argument(
        "--zip",
        default="data/raw/plantvillage_hf/data.zip"
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------

    with open(
        args.config,
        "r",
        encoding="utf-8"
    ) as f:
        config = yaml.safe_load(f)

    with open(
        args.manifest,
        "r",
        encoding="utf-8"
    ) as f:
        manifest = json.load(f)

    zip_path = Path(args.zip)

    if not zip_path.exists():
        raise FileNotFoundError(
            f"PlantVillage ZIP not found: {zip_path}"
        )

    classes = manifest["classes"]

    train_records = manifest["train"]
    val_records = manifest["val"]
    test_records = manifest["test"]

    print("=" * 70)
    print("PlantGuard - DINOv2 Disease Classifier")
    print("=" * 70)

    print(f"Classes : {len(classes)}")
    print(f"Train   : {len(train_records)}")
    print(f"Val     : {len(val_records)}")
    print(f"Test    : {len(test_records)}")
    print(f"ZIP     : {zip_path}")

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device  : {device}")

    if device == "cuda":
        print(
            "GPU     :",
            torch.cuda.get_device_name(0)
        )

    # ---------------------------------------------------------
    # Datasets
    # ---------------------------------------------------------

    train_dataset = PlantVillageZipDataset(
        zip_path,
        train_records,
        train=True
    )

    val_dataset = PlantVillageZipDataset(
        zip_path,
        val_records,
        train=False
    )

    test_dataset = PlantVillageZipDataset(
        zip_path,
        test_records,
        train=False
    )

    # ---------------------------------------------------------
    # DataLoaders
    # ---------------------------------------------------------

    dino_config = config["dino"]

    batch_size = dino_config["batch_size"]
    workers = dino_config["workers"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=(workers > 0)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=(workers > 0)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=(workers > 0)
    )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    model = DinoClassifier(
        dino_config["backbone"],
        len(classes),
        dino_config["freeze_backbone"]
    ).to(device)

    print()
    print("Backbone:", dino_config["backbone"])
    print(
        "Frozen backbone:",
        dino_config["freeze_backbone"]
    )

    # ---------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------

    trainable_params = [
        p
        for p in model.parameters()
        if p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=dino_config["lr"],
        weight_decay=dino_config["weight_decay"]
    )

    criterion = nn.CrossEntropyLoss(
        label_smoothing=dino_config["label_smoothing"]
    )

    # ---------------------------------------------------------
    # AMP
    # ---------------------------------------------------------

    amp_enabled = (
        device == "cuda"
        and dino_config.get("amp", True)
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled
    )

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    epochs = dino_config["epochs"]

    best_val_acc = 0.0

    artifact_path = Path(
        dino_config["disease_artifact"]
    )

    print()
    print("=" * 70)
    print("TRAINING")
    print("=" * 70)

    for epoch in range(epochs):

        model.train()

        total = 0
        correct = 0
        running_loss = 0.0

        for images, labels in train_loader:

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=amp_enabled
            ):

                logits = model(images)

                loss = criterion(
                    logits,
                    labels
                )

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

            batch_size_actual = labels.size(0)

            total += batch_size_actual

            correct += (
                logits.argmax(dim=1) == labels
            ).sum().item()

            running_loss += (
                loss.item() * batch_size_actual
            )

        train_loss = (
            running_loss /
            max(total, 1)
        )

        train_acc = (
            correct /
            max(total, 1)
        )

        val_acc = evaluate(
            model,
            val_loader,
            device
        )

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_acc:.4f} | "
            f"val_acc={val_acc:.4f}"
        )

        # -----------------------------------------------------
        # Save best checkpoint
        # -----------------------------------------------------

        if val_acc > best_val_acc:

            best_val_acc = val_acc

            save_bundle(
                model,
                classes,
                artifact_path,
                dino_config["backbone"]
            )

            print(
                f"  -> Saved best model "
                f"(val_acc={val_acc:.4f})"
            )

    # ---------------------------------------------------------
    # Load best model
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("BEST MODEL")
    print("=" * 70)

    checkpoint = torch.load(
        artifact_path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["state_dict"]
    )

    model.eval()

    # ---------------------------------------------------------
    # Final validation/test evaluation
    # ---------------------------------------------------------

    final_val_acc = evaluate(
        model,
        val_loader,
        device
    )

    test_acc = evaluate(
        model,
        test_loader,
        device
    )

    print(
        f"Best validation accuracy : "
        f"{best_val_acc:.4f}"
    )

    print(
        f"Final validation accuracy: "
        f"{final_val_acc:.4f}"
    )

    print(
        f"Test accuracy             : "
        f"{test_acc:.4f}"
    )

    print()
    print(
        "Disease classifier saved to:"
    )

    print(artifact_path)

    print("=" * 70)


if __name__ == "__main__":
    main()