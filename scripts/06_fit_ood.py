
"""Fit DINOv2 Mahalanobis OOD statistics for PlantGuard AI.

Training split: fit class means and shared within-class covariance.
Validation split: calibrate the in-distribution threshold using predicted classes.
Test split: deliberately not used.

Supports an extracted PlantVillage directory or the original ZIP.
The classifier checkpoint and weights are never modified.
"""
from pathlib import Path
import argparse
import io
import json
import zipfile

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms

from plantguard_ml.models.dino_classifier import DinoClassifier


PROJECT_ROOT = Path.cwd()
DEFAULT_ZIP = PROJECT_ROOT / "data" / "raw" / "plantvillage_hf" / "data.zip"


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/plantguard.yaml")
    parser.add_argument(
        "--manifest",
        default="data/processed/plantvillage_disease/manifest.json",
    )
    parser.add_argument(
        "--image-root",
        default=None,
        help="Extracted image root or path to the PlantVillage ZIP.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--percentile", type=float, default=99.0)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda"]
    )
    return parser.parse_args()


def resolve_image_source(manifest, explicit_root=None):
    """Resolve a directory or ZIP containing the manifest image paths."""
    records = manifest.get("train", [])
    if not records:
        raise ValueError("Manifest training split is empty.")

    relative_path = records[0]["path"].replace("\\", "/")

    if explicit_root:
        candidate = Path(explicit_root).expanduser().resolve()

        if candidate.is_file() and candidate.suffix.lower() == ".zip":
            with zipfile.ZipFile(candidate, "r") as archive:
                if relative_path not in archive.namelist():
                    raise FileNotFoundError(
                        f"ZIP does not contain: {relative_path}"
                    )
            print(f"Image source ZIP: {candidate}")
            return candidate

        if candidate.is_dir() and (candidate / Path(relative_path)).is_file():
            print(f"Image root: {candidate}")
            return candidate

        raise FileNotFoundError(
            "--image-root must be a directory containing manifest paths "
            "or a ZIP containing those paths.\n"
            f"Checked: {candidate}\n"
            f"Example relative path: {relative_path}"
        )

    directory_candidates = [
        PROJECT_ROOT,
        PROJECT_ROOT / "data" / "raw" / "plantvillage_hf",
        PROJECT_ROOT / "data" / "processed" / "plantvillage_disease",
        PROJECT_ROOT / "data" / "processed" / "plantvillage_hf",
    ]

    for candidate in directory_candidates:
        if (candidate / Path(relative_path)).is_file():
            print(f"Image root: {candidate.resolve()}")
            return candidate.resolve()

    zip_candidates = [
        DEFAULT_ZIP,
        PROJECT_ROOT / "data" / "downloads" / "data.zip",
    ]

    for candidate in zip_candidates:
        if candidate.is_file():
            try:
                with zipfile.ZipFile(candidate, "r") as archive:
                    archive.getinfo(relative_path)
                print(f"Image source ZIP: {candidate.resolve()}")
                return candidate.resolve()
            except (KeyError, zipfile.BadZipFile):
                continue

    raise FileNotFoundError(
        "Could not locate the first manifest image in a directory or ZIP.\n"
        f"Example relative path: {relative_path}\n"
        "Pass --image-root with the correct directory or ZIP path."
    )


class ImageSource:
    """Read images from a directory or a ZIP kept open for reuse."""

    def __init__(self, source_path):
        self.source_path = Path(source_path)
        self.archive = None

    def __enter__(self):
        if (
            self.source_path.is_file()
            and self.source_path.suffix.lower() == ".zip"
        ):
            self.archive = zipfile.ZipFile(self.source_path, "r")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.archive is not None:
            self.archive.close()
            self.archive = None

    def open_rgb(self, relative_path):
        normalized = relative_path.replace("\\", "/")

        if self.archive is not None:
            try:
                image_bytes = self.archive.read(normalized)
            except KeyError as exc:
                raise FileNotFoundError(
                    f"Image not found inside ZIP: {normalized}"
                ) from exc

            with Image.open(io.BytesIO(image_bytes)) as image:
                return image.convert("RGB")

        image_path = self.source_path / Path(normalized)
        if not image_path.is_file():
            raise FileNotFoundError(str(image_path))

        with Image.open(image_path) as image:
            return image.convert("RGB")


def get_features(output):
    """Convert common DINOv2 output formats to [batch, feature_dim]."""
    if isinstance(output, dict):
        if output.get("pooler_output") is not None:
            output = output["pooler_output"]
        elif output.get("last_hidden_state") is not None:
            output = output["last_hidden_state"]
        else:
            raise ValueError(
                "Backbone output dictionary has no recognized feature key."
            )
    elif isinstance(output, (tuple, list)):
        output = output[0]

    if not torch.is_tensor(output):
        raise TypeError("Backbone did not return a tensor.")

    if output.ndim == 3:
        output = output[:, 0, :]

    if output.ndim != 2:
        raise ValueError(
            f"Expected [batch, features], got {tuple(output.shape)}."
        )

    return output


def extract_embeddings(
    model,
    records,
    image_source,
    transform,
    device,
    batch_size,
    split_name,
):
    embeddings = []
    labels = []
    total = len(records)

    for start in range(0, total, batch_size):
        batch_records = records[start:start + batch_size]
        images = []
        batch_labels = []

        for record in batch_records:
            image = image_source.open_rgb(record["path"])
            try:
                images.append(transform(image))
            finally:
                image.close()

            batch_labels.append(int(record["class_id"]))

        batch_tensor = torch.stack(images).to(device)

        with torch.inference_mode():
            features = get_features(model.backbone(batch_tensor))

        embeddings.append(features.float().cpu().numpy())
        labels.extend(batch_labels)

        completed = min(start + len(batch_records), total)
        if completed == total or completed % (batch_size * 100) == 0:
            print(f"{split_name}: processed {completed}/{total} images")

    return (
        np.concatenate(embeddings, axis=0).astype(np.float32),
        np.asarray(labels, dtype=np.int64),
    )


def main():
    args = get_args()

    if not 0 < args.percentile < 100:
        raise ValueError("--percentile must be between 0 and 100.")

    if args.batch_size < 1 or args.ridge <= 0:
        raise ValueError("Batch size and ridge must be positive.")

    with open(args.config, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    with open(args.manifest, encoding="utf-8") as file:
        manifest = json.load(file)

    for split in ("train", "val"):
        if split not in manifest or not manifest[split]:
            raise ValueError(
                f"Manifest is missing a non-empty '{split}' split."
            )

    train_records = manifest["train"]
    val_records = manifest["val"]
    class_to_id = manifest["class_to_id"]
    known_ids = sorted(int(value) for value in class_to_id.values())

    if len(known_ids) != int(manifest["num_classes"]):
        raise ValueError(
            "Manifest class mapping and num_classes disagree."
        )

    train_ids = {
        int(record["class_id"]) for record in train_records
    }
    if train_ids != set(known_ids):
        raise ValueError(
            "Training split does not cover exactly the manifest classes."
        )

    val_ids = {int(record["class_id"]) for record in val_records}
    if not val_ids.issubset(set(known_ids)):
        raise ValueError("Validation contains an unknown class ID.")

    image_source_path = resolve_image_source(manifest, args.image_root)

    artifact_path = Path(config["dino"]["disease_artifact"])
    if not artifact_path.is_file():
        raise FileNotFoundError(
            f"Classifier artifact not found: {artifact_path}"
        )

    # Read the existing checkpoint. Never save or modify classifier weights.
    bundle = torch.load(
        artifact_path, map_location="cpu", weights_only=False
    )

    required_keys = {"classes", "state_dict", "backbone_name"}
    if not required_keys.issubset(bundle):
        raise ValueError(
            "Classifier checkpoint must contain classes, state_dict, "
            "and backbone_name."
        )

    model_classes = list(bundle["classes"])
    manifest_classes = sorted(class_to_id, key=class_to_id.get)

    if len(model_classes) != len(known_ids):
        raise ValueError(
            "Classifier and manifest have different class counts."
        )

    if manifest_classes != model_classes:
        raise ValueError(
            "Classifier class ordering differs from manifest ordering."
        )

    model = DinoClassifier(
        bundle["backbone_name"], len(model_classes)
    )
    model.load_state_dict(bundle["state_dict"], strict=True)
    model.eval()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but CUDA is unavailable.")

    model.to(device)

    print(f"Device: {device}")
    print(f"Training images: {len(train_records)}")
    print(f"Validation images: {len(val_records)}")
    print(f"Classifier classes: {len(model_classes)}")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    with ImageSource(image_source_path) as image_source:
        print("\nExtracting training embeddings...")
        train_x, train_y = extract_embeddings(
            model,
            train_records,
            image_source,
            transform,
            device,
            args.batch_size,
            "Train",
        )

        print("\nExtracting validation embeddings...")
        val_x, val_y = extract_embeddings(
            model,
            val_records,
            image_source,
            transform,
            device,
            args.batch_size,
            "Validation",
        )

    # Fit one mean per class in explicit class-ID order.
    means = []
    for class_id in known_ids:
        class_features = train_x[train_y == class_id]

        if len(class_features) < 2:
            raise ValueError(
                f"Class ID {class_id} has fewer than two training samples."
            )

        means.append(class_features.mean(axis=0))

    means = np.stack(means).astype(np.float64)

    # Fit shared within-class covariance with ridge regularization.
    centered = np.concatenate([
        train_x[train_y == class_id] - means[position]
        for position, class_id in enumerate(known_ids)
    ], axis=0).astype(np.float64)

    covariance = np.atleast_2d(np.cov(centered, rowvar=False))
    scale = float(np.trace(covariance) / covariance.shape[0])

    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid covariance scale.")

    covariance += (
        np.eye(covariance.shape[0], dtype=np.float64)
        * args.ridge
        * scale
    )
    inv_cov = np.linalg.pinv(covariance)

    # Calibrate with predicted classes to match inference behavior.
    # val_x contains the backbone embeddings used by the classifier head.
    val_distances = np.empty(len(val_x), dtype=np.float64)

    with torch.inference_mode():
        for start in range(0, len(val_x), args.batch_size):
            end = min(start + args.batch_size, len(val_x))
            features = torch.from_numpy(val_x[start:end]).to(device)
            logits = model.head(features)
            predicted_positions = logits.argmax(dim=1).cpu().numpy()

            for local_index, predicted_position in enumerate(
                predicted_positions
            ):
                index = start + local_index
                embedding = val_x[index].astype(np.float64)
                delta = embedding - means[int(predicted_position)]
                distance = float(delta @ inv_cov @ delta.T)
                val_distances[index] = max(0.0, distance)

    threshold = float(
        np.percentile(val_distances, args.percentile)
    )

    output_path = (
        PROJECT_ROOT / "artifacts" / "dino_disease" / "ood.npz"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        class_labels=np.asarray(known_ids, dtype=np.int64),
        means=means.astype(np.float32),
        inv_cov=inv_cov.astype(np.float32),
        threshold=np.asarray(threshold, dtype=np.float32),
    )

    metadata = {
        "method": "class_conditional_mahalanobis",
        "classifier_artifact": str(artifact_path),
        "manifest": str(Path(args.manifest)),
        "image_source": str(image_source_path),
        "class_labels": known_ids,
        "feature_dimension": int(train_x.shape[1]),
        "train_samples": int(len(train_y)),
        "validation_samples": int(len(val_y)),
        "calibration_percentile": args.percentile,
        "threshold": threshold,
        "ridge": args.ridge,
        "device": device,
        "note": (
            "Threshold is calibrated on in-distribution validation "
            "embeddings using the classifier's predicted classes, "
            "matching inference. Evaluate on separate OOD examples "
            "before deployment; this threshold alone does not prove "
            "OOD detection quality."
        ),
    }

    metadata_path = output_path.with_suffix(".json")
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("\nOOD fitting complete.")
    print(f"Statistics: {output_path}")
    print(f"Metadata:   {metadata_path}")
    print(f"Threshold:  {threshold:.6f}")
    print("Classifier weights were not modified.")


if __name__ == "__main__":
    main()
