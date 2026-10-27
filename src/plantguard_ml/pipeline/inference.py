
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from ..models.dino_classifier import DinoClassifier
from ..models.dino_leaf_filter import (
    DinoBinaryFilter,
    transform as leaf_transform,
)
from ..models.ood import MahalanobisOOD
from ..models.severity import make_features


class PlantGuardPipeline:
    def __init__(self, cfg):
        self.cfg = cfg

        requested_device = cfg["project"].get("device", "auto")
        if requested_device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        elif requested_device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA was requested, but CUDA is unavailable."
                )
            self.device = "cuda"
        elif requested_device == "cpu":
            self.device = "cpu"
        else:
            raise ValueError(
                f"Unsupported device setting: {requested_device}"
            )

        self.artifacts_dir = Path(cfg["paths"]["artifacts"])
        self.classifier = self._load_classifier()
        self.leaf_filter = self._load_leaf_filter()
        self.ood = self._load_ood()
        self.severity = self._load_severity()

        self.sam = None
        self.segmenter = None

        self.classify_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
            ),
        ])

    def _load_classifier(self):
        path = Path(self.cfg["dino"]["disease_artifact"])
        if not path.exists():
            return None

        bundle = torch.load(
            path, map_location="cpu", weights_only=False
        )
        model = DinoClassifier(
            bundle.get("backbone_name", "dinov2_vitb14"),
            len(bundle["classes"]),
            True,
        )
        model.load_state_dict(bundle["state_dict"])
        model.classes = bundle["classes"]

        return model.to(self.device).eval()

    def _load_leaf_filter(self):
        path = Path(self.cfg["dino"]["leaf_filter_artifact"])
        if not path.exists():
            return None

        bundle = torch.load(
            path, map_location="cpu", weights_only=False
        )
        model = DinoBinaryFilter(
            bundle.get("backbone_name", "dinov2_vits14")
        )
        model.load_state_dict(bundle["state_dict"])

        return model.to(self.device).eval()

    def _load_ood(self):
        path = self.artifacts_dir / "dino_disease" / "ood.npz"
        if not path.exists():
            return None

        if self.classifier is None:
            raise RuntimeError(
                "An OOD artifact exists, but the disease classifier "
                "could not be loaded."
            )

        ood = MahalanobisOOD.load(path)

        if len(ood.means) != len(self.classifier.classes):
            raise ValueError(
                "OOD class means do not match the classifier's class count. "
                "Refit OOD after verifying class ordering."
            )

        return ood

    def _load_severity(self):
        path = self.artifacts_dir / "severity" / "xgb_regressor.json"
        if not path.exists():
            return None

        try:
            import xgboost as xgb

            model = xgb.XGBRegressor()
            model.load_model(str(path))
            return model
        except Exception as exc:
            raise RuntimeError(
                f"Could not load the severity model: {exc}"
            ) from exc

    def attach_heavy_models(self, sam, segmenter=None):
        """Attach SAM2 and the existing segmentation adapter."""
        self.sam = sam
        self.segmenter = segmenter

    @torch.inference_mode()
    def _leaf_score(self, crop):
        if self.leaf_filter is None:
            return 1.0

        x = leaf_transform()(crop).unsqueeze(0).to(self.device)
        return float(torch.sigmoid(self.leaf_filter(x))[0].item())

    @torch.inference_mode()
    def _classify(self, image):
        if self.classifier is None:
            raise RuntimeError("Disease classifier is not loaded.")

        x = self.classify_transform(image).unsqueeze(0).to(self.device)
        logits = self.classifier(x)
        probabilities = logits.softmax(dim=1)[0]

        index = int(probabilities.argmax().item())
        confidence = float(probabilities[index].item())

        embedding = (
            self.classifier.backbone(x)[0]
            .detach()
            .float()
            .cpu()
            .numpy()
        )

        return index, confidence, embedding

    def _predict_disease_mask(self, crop):
        if self.segmenter is None:
            return None

        mask = self.segmenter.predict(np.asarray(crop.convert("RGB")))
        if mask is None:
            return None

        mask = np.asarray(mask)

        if mask.ndim != 2 or mask.size == 0:
            raise ValueError(
                "The segmentation adapter must return a non-empty "
                "2D disease mask."
            )

        if mask.shape != (crop.height, crop.width):
            raise ValueError(
                "Segmentation mask dimensions do not match the crop: "
                f"{mask.shape} versus {(crop.height, crop.width)}."
            )

        return mask

    def run(self, image):
        started = time.time()
        image = image.convert("RGB")
        rgb = np.asarray(image)

        if self.sam is None:
            raise RuntimeError("SAM2 adapter is not attached.")

        candidates = self.sam.candidate_masks(
            rgb,
            self.cfg["sam2"]["min_area_ratio"],
            self.cfg["sam2"]["max_area_ratio"],
        )

        results = []

        for candidate in candidates:
            mask = np.asarray(candidate["mask"], dtype=bool)
            ys, xs = np.where(mask)

            if len(xs) < 10:
                continue

            x1, x2 = int(xs.min()), int(xs.max()) + 1
            y1, y2 = int(ys.min()), int(ys.max()) + 1
            crop = image.crop((x1, y1, x2, y2))

            leaf_score = self._leaf_score(crop)
            if leaf_score < 0.5:
                continue

            if self.classifier is None:
                raise RuntimeError("Disease classifier is not loaded.")

            index, confidence, embedding = self._classify(crop)

            if self.ood is not None:
                ood_result = self.ood.score(
                    embedding, class_index=index
                )
            else:
                ood_result = {
                    "ood": None,
                    "distance": None,
                    "threshold": None,
                    "reason": "OOD model is not loaded",
                }

            disease_mask = self._predict_disease_mask(crop)
            severity = None
            disease_area = None

            # Only estimate severity when both a trained estimator and
            # an actual segmentation mask are available.
            if self.severity is not None and disease_mask is not None:
                features = make_features(
                    np.ones(disease_mask.shape, dtype=bool),
                    disease_mask,
                    np.array([confidence]),
                    1,
                    confidence,
                )

                prediction = self.severity.predict(
                    np.asarray(features).reshape(1, -1)
                )[0]

                severity = float(np.clip(prediction, 0, 100))
                disease_area = int((disease_mask > 0).sum())

            elif disease_mask is not None:
                disease_area = int((disease_mask > 0).sum())

            results.append({
                "leaf_score": leaf_score,
                "sam_score": float(candidate["score"]),
                "sam_stability": float(candidate["stability"]),
                "leaf_area_px": int(mask.sum()),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "disease_index": index,
                "disease": self.classifier.classes[index],
                "disease_confidence": confidence,
                "disease_area_px": disease_area,
                "severity": severity,
                "ood": ood_result,
            })

            if len(results) >= 5:
                break

        return {
            "status": "ok" if results else "no_valid_leaf",
            "leaves": results,
            "latency_ms": (time.time() - started) * 1000,
            "device": self.device,
        }
