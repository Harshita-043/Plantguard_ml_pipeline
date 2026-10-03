# PlantGuard ML Architecture

```text
REAL FIELD IMAGE
      |
      v
SAM 2.1 Automatic Mask Generator
      |
      +--> candidate leaf masks
      |
      v
crop each candidate
      |
      v
DINOv2 Leaf Filter
      |
      +--> reject invalid/non-leaf crops
      |
      v
DINOv2 Disease Classifier
      |
      +--> class + probability + embedding
      |                     |
      |                     +--> Mahalanobis OOD
      |
      v
SegNeXt
      |
      +--> disease pixel mask
      |
      v
XGBoost Severity
      |
      +--> 0-100 estimate
      |
      +--> Transformer-CAM explanation
      |
      v
FastAPI /v1/inference
```

## Why DINOv2 appears twice

The same family of visual representations is used for two separate tasks:

- **Leaf filter:** binary valid-leaf decision.
- **Disease classifier:** multi-class disease decision.

They have separate heads and separate artifacts because the objectives and training data are different.

## Why segmentation comes after classification

Classification answers *what disease class is most likely*. SegNeXt answers *which pixels in the leaf are diseased*. The segmentation mask then supplies measurable spatial features to the severity model.

## Why XGBoost is after SegNeXt

The severity model is a tabular regression problem built from interpretable image features such as disease-area percentage, number of disease components, edge contact, classifier confidence, and leaf area.

## Model-agnostic boundary

The application should not know whether the classifier is DINOv2 or another model. It only consumes the FastAPI JSON contract. This makes model replacement possible without rewriting the React/orchestration layer.
