from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps


class OnnxLeafFilter:
    """
    CPU inference adapter for the PlantGuard DINOv2 leaf-filter ONNX model.

    Expected model:
        artifacts/optimized/dino_leaf_filter/onnx_int8/
        dino_leaf_filter_dynamic_int8.onnx

    Model output:
        One logit per image.
        sigmoid(logit) represents the positive/leaf probability.

    Class mapping:
        0 -> negative
        1 -> positive

    This adapter does not train or modify model weights.
    """

    def __init__(
        self,
        model_path: str | Path = (
            "artifacts/optimized/dino_leaf_filter/"
            "onnx_int8/dino_leaf_filter_dynamic_int8.onnx"
        ),
        image_size: int = 224,
        resize_size: int = 256,
        threshold: float = 0.5,
        intra_op_threads: int | None = None,
    ) -> None:
        self.model_path = Path(model_path).resolve()
        self.image_size = int(image_size)
        self.resize_size = int(resize_size)
        self.threshold = float(threshold)

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"ONNX leaf-filter model not found: {self.model_path}"
            )

        if self.image_size <= 0:
            raise ValueError("image_size must be positive")

        if self.resize_size < self.image_size:
            raise ValueError(
                "resize_size must be greater than or equal to image_size"
            )

        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")

        options = ort.SessionOptions()
        options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        if intra_op_threads is not None:
            if intra_op_threads < 1:
                raise ValueError("intra_op_threads must be at least 1")
            options.intra_op_num_threads = int(intra_op_threads)

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self.input_shape = self.session.get_inputs()[0].shape

        # ImageNet normalization used during model validation.
        self.mean = np.asarray(
            [0.485, 0.456, 0.406], dtype=np.float32
        ).reshape(1, 1, 3)

        self.std = np.asarray(
            [0.229, 0.224, 0.225], dtype=np.float32
        ).reshape(1, 1, 3)

    @staticmethod
    def _sigmoid(logits: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        logits = np.asarray(logits, dtype=np.float32)
        result = np.empty_like(logits)

        positive = logits >= 0
        result[positive] = 1.0 / (
            1.0 + np.exp(-logits[positive])
        )

        exp_values = np.exp(logits[~positive])
        result[~positive] = exp_values / (1.0 + exp_values)

        return result

    def preprocess(self, image: Image.Image) -> np.ndarray:
        """
        Convert a PIL image to normalized NCHW float32 input.

        Intended preprocessing:
            Resize shortest edge to 256
            Center crop to 224 x 224
            RGB conversion
            ImageNet normalization
        """
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL.Image.Image")

        image = ImageOps.exif_transpose(image).convert("RGB")

        width, height = image.size

        if width <= 0 or height <= 0:
            raise ValueError("Image has invalid dimensions")

        # Match torchvision Resize(256): preserve aspect ratio and
        # resize the shorter edge to resize_size.
        if width <= height:
            new_width = self.resize_size
            new_height = int(
                self.resize_size * height / width
            )
        else:
            new_height = self.resize_size
            new_width = int(
                self.resize_size * width / height
            )

        image = image.resize(
            (new_width, new_height),
            resample=Image.Resampling.BILINEAR,
        )

        # Center crop, matching the common torchvision convention.
        left = int(round((new_width - self.image_size) / 2.0))
        top = int(round((new_height - self.image_size) / 2.0))

        right = left + self.image_size
        bottom = top + self.image_size

        image = image.crop((left, top, right, bottom))

        # Convert to float32 [0, 1].
        array = np.asarray(image, dtype=np.float32) / 255.0

        # ImageNet normalization.
        array = (array - self.mean) / self.std

        # HWC -> CHW; add batch dimension: NCHW.
        array = np.transpose(array, (2, 0, 1))
        array = np.expand_dims(array, axis=0)

        return np.ascontiguousarray(array, dtype=np.float32)

    def predict_probability(self, image: Image.Image) -> float:
        """Return the model's positive-class probability."""
        input_tensor = self.preprocess(image)

        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_tensor},
        )

        logits = np.asarray(outputs[0], dtype=np.float32).reshape(-1)

        if logits.size != 1:
            raise RuntimeError(
                f"Expected one output logit, received {logits.shape}"
            )

        probability = float(self._sigmoid(logits)[0])

        if not np.isfinite(probability):
            raise RuntimeError("Model returned a non-finite probability")

        return probability

    def predict(self, image: Image.Image) -> dict[str, Any]:
        """
        Predict whether an image contains a leaf.

        Returns a structured dictionary suitable for pipeline integration.
        """
        probability = self.predict_probability(image)

        return {
            "positive_probability": probability,
            "is_leaf": probability >= self.threshold,
            "threshold": self.threshold,
            "runtime": "onnxruntime-cpu",
            "model_path": str(self.model_path),
        }

    def predict_file(self, image_path: str | Path) -> dict[str, Any]:
        """Load an image file and return a prediction."""
        path = Path(image_path)

        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")

        with Image.open(path) as image:
            return self.predict(image)

    def predict_batch(
        self,
        images: list[Image.Image],
    ) -> list[dict[str, Any]]:
        """Predict a list of images individually."""
        return [self.predict(image) for image in images]
