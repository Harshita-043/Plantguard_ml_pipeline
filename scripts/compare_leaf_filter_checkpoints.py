from pathlib import Path
import json
import torch

root = Path("artifacts/dino_leaf_filter")

for name in ["dino_leaf_filter.pt", "dino_leaf_filter_v2.pt"]:
    path = root / name
    print(f"\n=== {name} ===")
    ckpt = torch.load(path, map_location="cpu", weights_only=True)

    for key in [
        "backbone_name", "classes", "image_size",
        "best_epoch", "best_val_accuracy", "best_val_loss",
        "test_accuracy", "test_loss"
    ]:
        print(f"{key}: {ckpt.get(key, 'NOT RECORDED')}")

report = root / "evaluation" / "evaluation_results.json"
print("\n=== EXISTING EVALUATION REPORT ===")
if report.exists():
    print(json.dumps(json.loads(report.read_text(encoding="utf-8")), indent=2))
else:
    print("No evaluation report found.")
