
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


class TransformerCAM:
    """
    Gradient-based input saliency for a DINOv2 disease classifier.

    Despite the historical class name, this implementation does not
    perform transformer attention rollout or canonical Grad-CAM.

    Returns a normalized heatmap resized to the original image dimensions.
    Model weights are never modified.
    """

    def __init__(self, model, size: int = 224):
        self.model = model
        self.size = size

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
            ),
        ])

    def explain(
        self,
        image: Image.Image,
        class_index: int,
    ) -> np.ndarray:
        """
        Explain a selected class using absolute input gradients.

        Returns:
            float32 NumPy heatmap with shape (original_height, original_width).
            Values are normalized to [0, 1].
        """
        if image.width < 1 or image.height < 1:
            raise ValueError("Input image must have positive dimensions.")

        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

        original_mode = self.model.training
        self.model.eval()

        try:
            # Explicitly enable gradients even if the caller normally uses
            # inference_mode/no_grad for ordinary predictions.
            with torch.enable_grad():
                x = self.transform(
                    image.convert("RGB")
                ).unsqueeze(0).to(device)

                x.requires_grad_(True)

                logits = self.model(x)

                if logits.ndim != 2:
                    raise ValueError(
                        f"Expected classifier logits with shape (B, C), "
                        f"received {tuple(logits.shape)}."
                    )

                if not 0 <= class_index < logits.shape[1]:
                    raise ValueError(
                        f"class_index={class_index} is outside the valid "
                        f"range 0..{logits.shape[1] - 1}."
                    )

                score = logits[0, class_index]

                # Compute input gradients without accumulating parameter grads.
                gradient = torch.autograd.grad(
                    outputs=score,
                    inputs=x,
                    retain_graph=False,
                    create_graph=False,
                )[0]

                # Absolute input-gradient saliency.
                heat = gradient.detach().abs().mean(dim=1, keepdim=True)

                # Upsample to the original image dimensions.
                heat = F.interpolate(
                    heat,
                    size=(image.height, image.width),
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]

                heat = heat.float()
                heat_min = heat.min()
                heat_max = heat.max()

                heat = (heat - heat_min) / (
                    heat_max - heat_min + 1e-8
                )

                return heat.cpu().numpy().astype(np.float32)

        finally:
            self.model.train(original_mode)

    @staticmethod
    def overlay(
        image: Image.Image,
        heatmap: np.ndarray,
        alpha: float = 0.40,
    ) -> Image.Image:
        """
        Blend a saliency heatmap with the original RGB image.

        Returns an RGB PIL image. This method does not save files.
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1.")

        if heatmap.ndim != 2:
            raise ValueError("Expected a 2D heatmap.")

        if heatmap.shape != (image.height, image.width):
            raise ValueError(
                "Heatmap dimensions must match the original image."
            )

        if not np.isfinite(heatmap).all():
            raise ValueError("Heatmap contains non-finite values.")

        heat = np.clip(heatmap, 0.0, 1.0)
        heat_uint8 = (heat * 255).astype(np.uint8)

        # Use a simple blue-to-red color ramp without extra dependencies.
        red = heat_uint8
        blue = 255 - heat_uint8
        green = (255 - np.abs(2 * heat_uint8.astype(np.int16) - 255)).astype(
            np.uint8
        )

        color_map = np.stack([red, green, blue], axis=-1)
        color_image = Image.fromarray(color_map, mode="RGB")

        base = image.convert("RGB")
        return Image.blend(base, color_image, alpha=alpha)
