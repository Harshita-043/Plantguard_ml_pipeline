from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/validate_dino_disease_int8_real_images.py"
TARGET = ROOT / "scripts/validate_dino_disease_selective_int8_real_images.py"

if not SOURCE.is_file():
    raise FileNotFoundError(f"Existing validator not found: {SOURCE}")

if TARGET.exists():
    raise FileExistsError(
        f"Refusing to overwrite existing script: {TARGET}"
    )

text = SOURCE.read_text(encoding="utf-8")

old_model = '''INT8_MODEL = (
    ROOT / "artifacts/optimized/dino_disease/onnx_int8/"
    "dino_disease_dynamic_int8.onnx"
)'''

new_model = '''INT8_MODEL = (
    ROOT / "artifacts/optimized/dino_disease/onnx_selective_int8/"
    "dino_disease_mlp_int8.onnx"
)'''

old_report = '''REPORT = (
    ROOT / "reports/device_validation/dino_disease/"
    "real_image_int8_comparison.json"
)'''

new_report = '''REPORT = (
    ROOT / "reports/device_validation/dino_disease/"
    "real_image_selective_int8_comparison.json"
)'''

if text.count(old_model) != 1:
    raise RuntimeError(
        "Expected INT8_MODEL definition was not found exactly once. "
        "No file was written."
    )

if text.count(old_report) != 1:
    raise RuntimeError(
        "Expected REPORT definition was not found exactly once. "
        "No file was written."
    )

text = text.replace(old_model, new_model, 1)
text = text.replace(old_report, new_report, 1)

TARGET.write_text(text, encoding="utf-8")
print("Created:", TARGET.relative_to(ROOT))
print("Candidate:", new_model.splitlines()[1].strip())
print("New report:", new_report.splitlines()[1].strip())
print("Original validator and report were not modified.")
