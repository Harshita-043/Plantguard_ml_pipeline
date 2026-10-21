import json
from pathlib import Path
from collections import Counter

ROOT = Path("data/raw/plantseg").resolve()
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

if not ROOT.exists():
    raise SystemExit(f"Dataset folder not found: {ROOT}")

files = [p for p in ROOT.rglob("*") if p.is_file()]
images = [p for p in files if p.suffix.lower() in IMAGE_EXTENSIONS]
json_files = [p for p in files if p.suffix.lower() == ".json"]

report = [
    f"Dataset root: {ROOT}",
    f"Total files: {len(files)}",
    f"Images: {len(images)}",
    f"JSON files: {len(json_files)}",
    f"Image extensions: {Counter(p.suffix.lower() for p in images)}",
    "\nTop-level entries:",
]

report.extend(
    f"  {p.name}{'/' if p.is_dir() else ''}"
    for p in sorted(ROOT.iterdir())
)

for path in json_files[:100]:
    report.append(f"\n--- {path.relative_to(ROOT)} ---")

    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))

        if isinstance(data, dict):
            report.append(f"Keys: {list(data.keys())[:30]}")

            for key in ("images", "annotations", "categories", "shapes"):
                if isinstance(data.get(key), list):
                    report.append(f"{key}: {len(data[key])}")

            if data.get("images"):
                report.append(
                    "First image record: "
                    + json.dumps(data["images"][0])[:1000]
                )

            if data.get("annotations"):
                report.append(
                    "First annotation: "
                    + json.dumps(data["annotations"][0])[:1500]
                )

    except Exception as exc:
        report.append(f"JSON read error: {exc}")

output = Path("plantseg_inspection.txt")
output.write_text("\n".join(report), encoding="utf-8")

print("\n".join(report)[:12000])
print(f"\nFull report saved to: {output.resolve()}")