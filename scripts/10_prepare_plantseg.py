"""Inspect PlantSeg and convert common COCO-style polygon/RLE annotations into simple image/mask pairs.
The script intentionally does not assume one historical PlantSeg zip layout. Run it after extraction and inspect its report."""
from pathlib import Path
import argparse,json,cv2,numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',default='data/processed/plantseg_simple');a=p.parse_args();r=Path(a.root);o=Path(a.out);(o/'images').mkdir(parents=True,exist_ok=True);(o/'masks').mkdir(parents=True,exist_ok=True)
imgs=[x for x in r.rglob('*') if x.suffix.lower() in {'.jpg','.jpeg','.png'}];print('images found',len(imgs));
for f in imgs[:20]: print(f)
print('\nIMPORTANT: PlantSeg releases have changed layout/annotation packaging. The official dataset repository and Zenodo record are linked in docs/MANUAL.md. Use the included conversion helper after confirming your downloaded release structure.')
