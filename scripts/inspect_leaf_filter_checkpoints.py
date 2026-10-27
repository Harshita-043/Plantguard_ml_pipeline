from pathlib import Path
import torch

files = [
    Path("artifacts/dino_leaf_filter/dino_leaf_filter.pt"),
    Path("artifacts/dino_leaf_filter/dino_leaf_filter_v2.pt"),
]

for path in files:
    print(f"\n=== {path} ===")
    if not path.is_file():
        print("MISSING")
        continue

    obj = torch.load(path, map_location="cpu", weights_only=True)
    print("Checkpoint type:", type(obj).__name__)

    if isinstance(obj, dict):
        print("Top-level keys:", list(obj.keys())[:30])
        state = obj.get("state_dict", obj.get("model_state_dict", obj))
        if isinstance(state, dict):
            print("Tensor count:", sum(torch.is_tensor(v) for v in state.values()))
            for key, value in list(state.items())[:12]:
                if torch.is_tensor(value):
                    print(f"  {key}: {tuple(value.shape)}")
        for key in ("backbone_name", "image_size", "classes", "class_names"):
            if key in obj:
                print(f"{key}: {obj[key]}")
    else:
        print("Unexpected checkpoint format")
