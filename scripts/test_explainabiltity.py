
from pathlib import Path
import sys

import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plantguard_ml.models.dino_classifier import DinoClassifier
from plantguard_ml.pipeline.cam import TransformerCAM


def main():
    config_path = ROOT / "configs" / "plantguard.yaml"

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    artifact = ROOT / cfg["dino"]["disease_artifact"]
    image_path = ROOT / "data" / "raw" / "plantdoc_hf" / "images" / "plantdoc_00001.jpg"

    if not artifact.is_file():
        raise FileNotFoundError(f"Classifier checkpoint missing: {artifact}")
    if not image_path.is_file():
        raise FileNotFoundError(f"Test image missing: {image_path}")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Loading disease classifier...")
    bundle = torch.load(
        artifact, map_location="cpu", weights_only=False
    )

    model = DinoClassifier(
        bundle.get("backbone_name", "dinov2_vitb14"),
        len(bundle["classes"]),
        freeze_backbone=True,
    )
    model.load_state_dict(bundle["state_dict"])
    model.classes = bundle["classes"]
    model = model.to(device).eval()

    image = Image.open(image_path).convert("RGB")

    # Obtain a predicted class using the classifier's normal forward pass.
    from torchvision import transforms

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
        ),
    ])

    with torch.inference_mode():
        logits = model(preprocess(image).unsqueeze(0).to(device))
        probabilities = logits.softmax(dim=1)[0]
        class_index = int(probabilities.argmax())
        confidence = float(probabilities[class_index])

    print("Device:", device)
    print("Predicted class:", model.classes[class_index])
    print("Confidence:", round(confidence, 4))
    print("Generating gradient-saliency heatmap...")

    explainer = TransformerCAM(model)
    heatmap = explainer.explain(image, class_index)
    overlay = explainer.overlay(image, heatmap)

    output_dir = ROOT / "outputs" / "explainability"
    output_dir.mkdir(parents=True, exist_ok=True)

    heat_path = output_dir / "plantdoc_heatmap.png"
    overlay_path = output_dir / "plantdoc_overlay.png"

    # Save the heatmap as a grayscale image.
    import numpy as np

    Image.fromarray(
        (np.clip(heatmap, 0, 1) * 255).astype(np.uint8),
        mode="L",
    ).save(heat_path)

    overlay.save(overlay_path)

    print("Heatmap:", heat_path)
    print("Overlay:", overlay_path)
    print("Heatmap shape:", heatmap.shape)
    print("Explainability test PASSED.")


if __name__ == "__main__":
    main()
