import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerForSemanticSegmentation
from tqdm import tqdm


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class PlantSegDataset(Dataset):
    def __init__(self, root, split, size, augment=False):
        self.root = Path(root)
        self.size = size
        self.augment = augment
        self.images = sorted(
            (self.root / "images" / split).glob("*.jpg")
        )

        if not self.images:
            raise RuntimeError(f"No images found for split: {split}")

        self.pairs = []
        for image_path in self.images:
            mask_path = self.root / "masks" / split / (
                image_path.stem + ".png"
            )
            if not mask_path.exists():
                raise FileNotFoundError(mask_path)
            self.pairs.append((image_path, mask_path))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = image.resize(
            (self.size, self.size), Image.Resampling.BILINEAR
        )
        mask = mask.resize(
            (self.size, self.size), Image.Resampling.NEAREST
        )

        if self.augment and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        image = np.asarray(image, dtype=np.float32) / 255.0
        image = (image - MEAN) / STD

        # Masks must contain only 0 and 1.
        mask = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.int64)

        return (
            torch.from_numpy(image.transpose(2, 0, 1).copy()),
            torch.from_numpy(mask.copy()),
        )


def calculate_metrics(tp, fp, fn, tn):
    eps = 1e-8
    return {
        "disease_iou": float(tp / (tp + fp + fn + eps)),
        "disease_dice": float(2 * tp / (2 * tp + fp + fn + eps)),
        "precision": float(tp / (tp + fp + eps)),
        "recall": float(tp / (tp + fn + eps)),
        "pixel_accuracy": float((tp + tn) / (tp + tn + fp + fn + eps)),
    }


@torch.no_grad()
def evaluate(model, loader, device, amp):
    model.eval()
    tp = fp = fn = tn = 0
    losses = []

    for images, masks in tqdm(loader, desc="Validation", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=amp
        ):
            logits = model(pixel_values=images).logits
            logits = F.interpolate(
                logits,
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            loss = F.cross_entropy(logits, masks)

        predictions = logits.argmax(dim=1)

        tp += int(((predictions == 1) & (masks == 1)).sum().item())
        fp += int(((predictions == 1) & (masks == 0)).sum().item())
        fn += int(((predictions == 0) & (masks == 1)).sum().item())
        tn += int(((predictions == 0) & (masks == 0)).sum().item())
        losses.append(loss.item())

    return {
        "loss": float(np.mean(losses)),
        **calculate_metrics(tp, fp, fn, tn),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/plantseg_binary")
    parser.add_argument("--output", default="outputs/segformer_plantseg")
    parser.add_argument("--model", default="nvidia/mit-b0")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=6)
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"

    if device.type == "cpu":
        print("WARNING: CUDA is unavailable; training will be slow.")
    else:
        print("GPU:", torch.cuda.get_device_name(0))

    print("Device:", device, "| AMP:", amp)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    train_data = PlantSegDataset(
        args.data, "train", args.size, augment=True
    )
    val_data = PlantSegDataset(
        args.data, "val", args.size, augment=False
    )

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=amp,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=amp,
    )

    id2label = {0: "background", 1: "diseased_region"}
    label2id = {value: key for key, value in id2label.items()}

    model = SegformerForSemanticSegmentation.from_pretrained(
        args.model,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    best_dice = -1.0
    patience_count = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []

        progress = tqdm(
            train_loader, desc=f"Epoch {epoch}/{args.epochs}"
        )

        for images, masks in progress:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=amp,
            ):
                logits = model(pixel_values=images).logits
                logits = F.interpolate(
                    logits,
                    size=masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                loss = F.cross_entropy(logits, masks)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(loss.item())
            progress.set_postfix(loss=f"{loss.item():.4f}")

        validation = evaluate(model, val_loader, device, amp)

        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            **{f"val_{k}": v for k, v in validation.items()},
        }
        history.append(record)

        (output / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )

        print(json.dumps(record, indent=2))

        if validation["disease_dice"] > best_dice:
            best_dice = validation["disease_dice"]
            patience_count = 0

            best_path = output / "best_model"
            best_path.mkdir(parents=True, exist_ok=True)

            model.save_pretrained(best_path, safe_serialization=True)

            # Save exact preprocessing settings used by this script.
            (best_path / "preprocessing.json").write_text(
                json.dumps({
                    "size": args.size,
                    "mean": MEAN.tolist(),
                    "std": STD.tolist(),
                    "class_mapping": id2label,
                }, indent=2),
                encoding="utf-8",
            )

            print(f"Saved best checkpoint. Validation Dice={best_dice:.4f}")
        else:
            patience_count += 1

        if patience_count >= args.patience:
            print("Early stopping: validation Dice has stopped improving.")
            break

    summary = {
        "best_validation_disease_dice": best_dice,
        "epochs_completed": len(history),
        "arguments": vars(args),
    }
    (output / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("Training complete.")
    print("Best model:", (output / "best_model").resolve())
    print("Best validation Dice:", best_dice)


if __name__ == "__main__":
    main()