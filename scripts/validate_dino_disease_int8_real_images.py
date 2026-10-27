from __future__ import annotations

import io
import json
import time
import random
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps
from torchvision import transforms

ROOT = Path.cwd()
ARCHIVE = ROOT / "data/raw/plantvillage_hf/data.zip"
CLASSES_FILE = ROOT / "artifacts/dino_disease/classes.json"
OOD_FILE = ROOT / "artifacts/dino_disease/ood.npz"

FP32_MODEL = (
    ROOT / "artifacts/optimized/dino_disease/onnx/"
    "dino_disease_fp32_224.onnx"
)
INT8_MODEL = (
    ROOT / "artifacts/optimized/dino_disease/onnx_int8/"
    "dino_disease_dynamic_int8.onnx"
)
REPORT = (
    ROOT / "reports/device_validation/dino_disease/"
    "real_image_int8_comparison.json"
)

MAX_IMAGES_PER_CLASS = 30
SEED = 42
BATCH_SIZE = 4
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def softmax(logits):
    logits = logits.astype(np.float64)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def make_session(path):
    options = ort.SessionOptions()
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )
    options.intra_op_num_threads = max(1, min(8, __import__("os").cpu_count() or 1))
    return ort.InferenceSession(
        str(path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def mahalanobis_distance(embedding, mean, inv_cov):
    delta = embedding.astype(np.float64) - mean.astype(np.float64)
    value = float(delta @ inv_cov @ delta.T)
    return max(0.0, value)


def main():
    for path in (ARCHIVE, CLASSES_FILE, FP32_MODEL, INT8_MODEL, OOD_FILE):
        if not path.is_file():
            raise FileNotFoundError(f"Required file missing: {path}")

    classes = json.loads(CLASSES_FILE.read_text(encoding="utf-8"))
    class_to_index = {name: i for i, name in enumerate(classes)}

    with np.load(OOD_FILE, allow_pickle=False) as data:
        means = data["means"].astype(np.float64)
        inv_cov = data["inv_cov"].astype(np.float64)
        ood_threshold = float(np.asarray(data["threshold"]).item())

    if means.shape[0] != len(classes) or means.shape[1] != 768:
        raise ValueError(
            f"OOD means shape {means.shape} is incompatible with "
            f"{len(classes)} classes and 768 features."
        )
    if inv_cov.shape != (768, 768):
        raise ValueError(f"Unexpected OOD covariance shape: {inv_cov.shape}")

    # Same inference preprocessing as DinoClassifier.get_transform(train=False).
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
        ),
    ])

    print("Opening ZIP; no extraction will be performed.")
    by_class = defaultdict(list)

    with zipfile.ZipFile(ARCHIVE, "r") as archive:
        for name in archive.namelist():
            path = Path(name)
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            # Expected archive paths: raw/color/<class_name>/<image>
            class_name = next(
                (part for part in reversed(path.parts[:-1])
                 if part in class_to_index),
                None,
            )
            if class_name is not None:
                by_class[class_name].append(name)

        rng = random.Random(SEED)
        selected = []

        for class_name in classes:
            candidates = by_class.get(class_name, [])
            if not candidates:
                print(f"WARNING: no images found for {class_name}")
                continue

            rng.shuffle(candidates)
            for name in candidates[:MAX_IMAGES_PER_CLASS]:
                selected.append((name, class_name))

        rng.shuffle(selected)
        print("Classes with images:", sum(bool(by_class.get(c)) for c in classes))
        print("Images selected:", len(selected))
        print("Maximum per class:", MAX_IMAGES_PER_CLASS)

        fp32 = make_session(FP32_MODEL)
        int8 = make_session(INT8_MODEL)

        fp_logits_all = []
        q_logits_all = []
        fp_emb_all = []
        q_emb_all = []
        labels_all = []
        paths_all = []
        fp_latencies = []
        q_latencies = []
        skipped = 0

        # Decode each image once and send the identical tensor to both models.
        for start in range(0, len(selected), BATCH_SIZE):
            batch_items = selected[start:start + BATCH_SIZE]
            tensors = []
            batch_labels = []
            batch_paths = []

            for member, class_name in batch_items:
                try:
                    with archive.open(member) as stream:
                        with Image.open(io.BytesIO(stream.read())) as image:
                            image = ImageOps.exif_transpose(image).convert("RGB")
                            tensor = transform(image).numpy()

                    tensors.append(tensor)
                    batch_labels.append(class_to_index[class_name])
                    batch_paths.append(member)
                except Exception as exc:
                    skipped += 1
                    print(f"Skipping unreadable image: {member} ({exc})")

            if not tensors:
                continue

            batch = np.stack(tensors).astype(np.float32)

            t0 = time.perf_counter()
            fp_logits, fp_emb = fp32.run(
                ["logits", "embeddings"], {"images": batch}
            )
            fp_latencies.append((time.perf_counter() - t0) / len(batch))

            t0 = time.perf_counter()
            q_logits, q_emb = int8.run(
                ["logits", "embeddings"], {"images": batch}
            )
            q_latencies.append((time.perf_counter() - t0) / len(batch))

            fp_logits_all.append(fp_logits)
            q_logits_all.append(q_logits)
            fp_emb_all.append(fp_emb)
            q_emb_all.append(q_emb)
            labels_all.extend(batch_labels)
            paths_all.extend(batch_paths)

            done = min(start + BATCH_SIZE, len(selected))
            if done % 100 < BATCH_SIZE or done == len(selected):
                print(f"Processed archive entries: {done}/{len(selected)}")

    if not labels_all:
        raise RuntimeError("No images could be loaded for evaluation.")

    y = np.asarray(labels_all, dtype=np.int64)
    fp_logits = np.concatenate(fp_logits_all)
    q_logits = np.concatenate(q_logits_all)
    fp_emb = np.concatenate(fp_emb_all)
    q_emb = np.concatenate(q_emb_all)

    fp_probs = softmax(fp_logits)
    q_probs = softmax(q_logits)

    fp_pred = fp_logits.argmax(axis=1)
    q_pred = q_logits.argmax(axis=1)

    fp_dist = np.asarray([
        mahalanobis_distance(fp_emb[i], means[fp_pred[i]], inv_cov)
        for i in range(len(y))
    ])
    q_dist = np.asarray([
        mahalanobis_distance(q_emb[i], means[q_pred[i]], inv_cov)
        for i in range(len(y))
    ])

    fp_ood = fp_dist > ood_threshold
    q_ood = q_dist > ood_threshold

    # Per-class accuracy, plus an unweighted macro average.
    per_class = {}
    for index, name in enumerate(classes):
        mask = y == index
        if mask.any():
            per_class[name] = {
                "images": int(mask.sum()),
                "fp32_accuracy": float((fp_pred[mask] == y[mask]).mean()),
                "int8_accuracy": float((q_pred[mask] == y[mask]).mean()),
                "prediction_agreement": float((fp_pred[mask] == q_pred[mask]).mean()),
            }

    fp_size = FP32_MODEL.stat().st_size
    q_size = INT8_MODEL.stat().st_size

    report = {
        "dataset_archive": str(ARCHIVE.relative_to(ROOT)),
        "dataset_note": (
            "Deterministic capped sample from PlantVillage ZIP. "
            "Not an independent held-out test unless split provenance is verified."
        ),
        "seed": SEED,
        "max_images_per_class": MAX_IMAGES_PER_CLASS,
        "images_evaluated": int(len(y)),
        "images_skipped": int(skipped),
        "classes_with_images": int(len(per_class)),
        "class_count": len(classes),
        "batch_size": BATCH_SIZE,
        "fp32_model": str(FP32_MODEL.relative_to(ROOT)),
        "int8_model": str(INT8_MODEL.relative_to(ROOT)),
        "fp32_size_mib": fp_size / (1024 ** 2),
        "int8_size_mib": q_size / (1024 ** 2),
        "size_reduction_percent": 100 * (1 - q_size / fp_size),
        "fp32_top1_accuracy": float((fp_pred == y).mean()),
        "int8_top1_accuracy": float((q_pred == y).mean()),
        "fp32_int8_prediction_agreement": float((fp_pred == q_pred).mean()),
        "fp32_mean_max_probability": float(fp_probs.max(axis=1).mean()),
        "int8_mean_max_probability": float(q_probs.max(axis=1).mean()),
        "probability_max_abs_error": float(np.max(np.abs(fp_probs - q_probs))),
        "probability_mean_abs_error": float(np.mean(np.abs(fp_probs - q_probs))),
        "logits_max_abs_error": float(np.max(np.abs(fp_logits - q_logits))),
        "embeddings_max_abs_error": float(np.max(np.abs(fp_emb - q_emb))),
        "embeddings_mean_abs_error": float(np.mean(np.abs(fp_emb - q_emb))),
        "fp32_ood_rate": float(fp_ood.mean()),
        "int8_ood_rate": float(q_ood.mean()),
        "ood_decision_agreement": float((fp_ood == q_ood).mean()),
        "ood_threshold": ood_threshold,
        "ood_distance_mean_abs_difference": float(np.mean(np.abs(fp_dist - q_dist))),
        "fp32_median_latency_ms_per_image": float(np.median(fp_latencies) * 1000),
        "int8_median_latency_ms_per_image": float(np.median(q_latencies) * 1000),
        "per_class": per_class,
        "production_approved": False,
    }

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    if REPORT.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing report: {REPORT}"
        )
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n========== REAL IMAGE COMPARISON ==========")
    for key in [
        "images_evaluated",
        "images_skipped",
        "classes_with_images",
        "fp32_top1_accuracy",
        "int8_top1_accuracy",
        "fp32_int8_prediction_agreement",
        "probability_max_abs_error",
        "probability_mean_abs_error",
        "logits_max_abs_error",
        "embeddings_max_abs_error",
        "ood_decision_agreement",
        "ood_distance_mean_abs_difference",
        "fp32_median_latency_ms_per_image",
        "int8_median_latency_ms_per_image",
        "size_reduction_percent",
    ]:
        print(f"{key}: {report[key]}")

    print("Report:", REPORT)
    print("\nThis report is diagnostic; it does not automatically approve INT8.")


if __name__ == "__main__":
    main()
