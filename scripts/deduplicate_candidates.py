import json
from pathlib import Path

BASE = Path("data/processed/leaf_filter/sam_candidates")
INPUT = BASE / "candidates.jsonl"
BACKUP = BASE / "candidates_backup.jsonl"
TEMP = BASE / "candidates_clean.jsonl"

# Backup original
if not BACKUP.exists():
    BACKUP.write_bytes(INPUT.read_bytes())
    print("Backup created:", BACKUP)
else:
    print("Backup already exists:", BACKUP)

seen = set()
total = 0
kept = 0
removed = 0

with INPUT.open("r", encoding="utf-8") as src, TEMP.open("w", encoding="utf-8") as dst:
    for line in src:
        if not line.strip():
            continue

        record = json.loads(line)

        # IMPORTANT:
        # Same source image can have MANY valid candidates.
        # Only an identical source_image + candidate_crop is considered duplicate.
        key = (
            record.get("source_image"),
            record.get("candidate_crop"),
        )

        total += 1

        if key in seen:
            removed += 1
            continue

        seen.add(key)
        dst.write(json.dumps(record) + "\n")
        kept += 1

# Replace working manifest with cleaned one
TEMP.replace(INPUT)

print()
print("===== DEDUPLICATION COMPLETE =====")
print("Original records :", total)
print("Records kept     :", kept)
print("Duplicates removed:", removed)
print("Clean manifest   :", INPUT)
print("Backup           :", BACKUP)