# PlantGuard ML Layer

End-to-end, model-agnostic ML layer for PlantGuard:

REAL FIELD IMAGE -> SAM 2.1 -> DINOv2 Leaf Filter -> DINOv2 Disease Classifier -> SegNeXt Disease Segmentation -> XGBoost Severity -> Transformer-CAM + OOD/Reliability -> FastAPI

## Important
- SAM 2.1 is used as a pretrained foundation model for leaf segmentation. Do not retrain it unless you have a curated leaf segmentation dataset and a high-end multi-GPU setup.
- DINOv2 disease classifier is trained on PlantVillage by this repository.
- DINOv2 leaf filter is a binary classifier. You must provide positive leaf images and negative/non-leaf images, or use the included bootstrap script to create a starting dataset.
- SegNeXt training uses the official SegNeXt/MMsegmentation implementation. The repository includes conversion/config/launcher scripts rather than copying the upstream framework into this project.
- Severity requires labeled severity targets. If PlantSeg has no numeric severity labels for your selected subset, you must create/obtain severity labels. The included script supports CSV labels and also supports a temporary segmentation-derived target for pipeline smoke testing, clearly marked as NOT a scientific severity ground truth.
- Model weights are intentionally not bundled in the ZIP.

## Repository layout

- `configs/`: central YAML configuration.
- `scripts/`: beginner-friendly commands for setup, data preparation, training, evaluation and optimization.
- `src/plantguard_ml/`: inference components and FastAPI boundary.
- `artifacts/`: final exported models are written here.
- `data/`: datasets are placed here manually; large datasets are not bundled.
- `docs/MANUAL.md`: complete start-to-end guide.

## Quick start

Windows PowerShell:

```powershell
conda create -n plantguard-ml python=3.11 -y
conda activate plantguard-ml
pip install -r requirements-cpu.txt
# For NVIDIA GPU, follow docs/MANUAL.md for the correct PyTorch CUDA install first.
python scripts/01_check_environment.py
python scripts/02_prepare_plantvillage.py --input <PATH_TO_PLANTVILLAGE>
python scripts/03_train_dino_disease.py --config configs/plantguard.yaml
```

For the full pipeline and exact dataset/checkpoint setup, read `docs/MANUAL.md` first.

## Deployment contract

The FastAPI endpoint accepts one image and returns a JSON object containing:
- leaf candidates/masks
- leaf validity scores
- disease predictions/probabilities
- disease segmentation mask statistics
- severity estimate + reliability
- OOD score
- explainability artifact path when enabled
- model/artifact versions

The application layer should only call the FastAPI endpoint; it should not import SAM/DINO/SegNeXt internals.

## Optimization

The optimization script exports supported PyTorch models to ONNX using the modern `torch.onnx.export(..., dynamo=True)` path and validates the result with ONNX Runtime. PyTorch documents this as the recommended exporter path. citeturn0search1turn0search3

DINOv2 upstream: https://github.com/facebookresearch/dinov2
SAM 2 upstream: https://github.com/facebookresearch/sam2
SegNeXt upstream: https://github.com/Visual-Attention-Network/SegNeXt
PlantSeg: https://github.com/tqwei05/PlantSeg
