import argparse
import json
import csv
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def get_split(annotation_file, image_file, root):
    aliases = {
        "train": "train", "training": "train",
        "val": "val", "valid": "val", "validation": "val",
        "test": "test", "testing": "test",
    }

    # Prefer split directory names.
    for path in (image_file, annotation_file):
        for part in reversed(path.relative_to(root).parts[:-1]):
            if part.lower() in aliases:
                return aliases[part.lower()]

    # Otherwise use names such as instances_train.json or val.json.
    name = annotation_file.stem.lower()
    for token, split in aliases.items():
        if name == token or name.endswith("_" + token) or name.startswith(token + "_"):
            return split

    return None


def locate_image(root, filename, json_path):
    filename = str(filename).replace("\\", "/")
    candidates = [
        json_path.parent / filename,
        root / filename,
        root / "images" / filename,
    ]

    for path in candidates:
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            return path.resolve()

    # Fallback for releases where file_name includes an unexpected prefix.
    matches = list(root.rglob(Path(filename).name))
    return matches[0].resolve() if len(matches) == 1 else None


def coco_mask(annotation, width, height):
    mask = Image.new("L", (width, height), 0)
    segmentation = annotation.get("segmentation", [])

    if isinstance(segmentation, dict):
        try:
            from pycocotools import mask as mask_utils
        except ImportError:
            raise RuntimeError(
                "COCO RLE found. Install pycocotools in this environment: "
                "python -m pip install pycocotools"
            )

        decoded = mask_utils.decode(segmentation)
        if decoded.ndim == 3:
            decoded = decoded.any(axis=2)
        return Image.fromarray(decoded.astype(np.uint8))

    draw = ImageDraw.Draw(mask)

    # COCO polygon format: [[x1, y1, x2, y2, ...], ...]
    for polygon in segmentation:
        if not isinstance(polygon, list) or len(polygon) < 6:
            continue

        points = [
            (float(polygon[i]), float(polygon[i + 1]))
            for i in range(0, len(polygon) - 1, 2)
        ]

        if len(points) >= 3:
            draw.polygon(points, fill=1)

    return mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/raw/plantseg")
    parser.add_argument("--out", default="data/processed/plantseg_binary")
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out = Path(args.out).resolve()

    if not root.is_dir():
        raise SystemExit(f"Dataset folder not found: {root}")

    if out.exists() and any(out.iterdir()):
        raise SystemExit(
            f"Output folder is not empty: {out}. "
            "Choose a new output folder to avoid overwriting data."
        )

    # Find COCO JSON files.
    annotation_files = []
    for path in root.rglob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            continue

        if (
            isinstance(data, dict)
            and isinstance(data.get("images"), list)
            and isinstance(data.get("annotations"), list)
        ):
            annotation_files.append((path, data))

    if not annotation_files:
        raise SystemExit(
            "No COCO JSON files found. Inspect plantseg_inspection.txt "
            "and adapt this script to the actual annotation schema."
        )

    # Check whether annotations can be assigned to the official splits.
    jobs = []
    missing_images = []
    missing_splits = []

    for json_path, data in annotation_files:
        annotations_by_image = {}
        for ann in data["annotations"]:
            annotations_by_image.setdefault(ann.get("image_id"), []).append(ann)

        for record in data["images"]:
            image_path = locate_image(root, record.get("file_name", ""), json_path)

            if image_path is None:
                missing_images.append(str(record.get("file_name")))
                continue

            split = get_split(json_path, image_path, root)
            if split not in {"train", "val", "test"}:
                missing_splits.append(str(json_path))
                continue

            jobs.append((
                image_path,
                split,
                annotations_by_image.get(record.get("id"), []),
                json_path,
            ))

    if missing_images:
        raise SystemExit(
            f"Could not locate {len(missing_images)} images. "
            f"Examples: {missing_images[:5]}. No output created."
        )

    if missing_splits:
        raise SystemExit(
            "Could not identify train/val/test from annotation filenames or "
            "directory paths. Inspect the report and update get_split(). "
            f"Examples: {missing_splits[:5]}. No output created."
        )

    if not jobs:
        raise SystemExit("No usable image/annotation pairs found.")

    counts = Counter(job[1] for job in jobs)
    if any(counts[s] == 0 for s in ("train", "val", "test")):
        raise SystemExit(f"Expected train, val, test splits; found {dict(counts)}")

    out.mkdir(parents=True, exist_ok=True)
    rows = []
    empty_masks = 0

    for index, (image_path, split, annotations, json_path) in enumerate(jobs):
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        mask = Image.new("L", (width, height), 0)

        # Merge all annotated disease regions into a single binary mask.
        for ann in annotations:
            one_mask = coco_mask(ann, width, height)
            mask = Image.fromarray(
                np.maximum(np.asarray(mask), np.asarray(one_mask)).astype(np.uint8)
            )

        image = image.resize((args.size, args.size), Image.Resampling.BILINEAR)
        mask = mask.resize((args.size, args.size), Image.Resampling.NEAREST)

        stem = f"{index:07d}_{image_path.stem}"
        image_out = out / "images" / split / f"{stem}.jpg"
        mask_out = out / "masks" / split / f"{stem}.png"

        image_out.parent.mkdir(parents=True, exist_ok=True)
        mask_out.parent.mkdir(parents=True, exist_ok=True)

        image.save(image_out, quality=95)
        mask.save(mask_out)

        positive_pixels = int((np.asarray(mask) > 0).sum())
        if positive_pixels == 0:
            empty_masks += 1

        rows.append({
            "split": split,
            "image": str(image_out),
            "mask": str(mask_out),
            "source_image": str(image_path),
            "source_json": str(json_path),
            "annotation_count": len(annotations),
            "positive_pixels": positive_pixels,
        })

        # Save a few overlays for manual quality control.
        if index < 30:
            overlay = image.copy()
            red = Image.new("RGB", image.size, (255, 0, 0))
            overlay.paste(red, mask=mask.point(lambda v: 100 if v > 0 else 0))
            overlay = Image.blend(image, overlay, 0.6)

            overlay_dir = out / "overlays"
            overlay_dir.mkdir(exist_ok=True)
            overlay.save(overlay_dir / f"{stem}_overlay.jpg")

        if (index + 1) % 500 == 0:
            print(f"Processed {index + 1}/{len(jobs)} images")

    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "source": str(root),
        "output": str(out),
        "records": len(rows),
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "empty_masks": empty_masks,
        "target_size": args.size,
    }

    (out / "preparation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print(json.dumps(report, indent=2))
    print(f"Inspect overlays in: {out / 'overlays'}")


if __name__ == "__main__":
    main()