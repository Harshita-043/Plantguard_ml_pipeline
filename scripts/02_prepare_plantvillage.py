"""Validate an extracted PlantVillage ImageFolder dataset and create a split manifest."""
from pathlib import Path
import argparse, json, random
from PIL import Image
p=argparse.ArgumentParser(); p.add_argument('--input',required=True); p.add_argument('--out',default='data/processed/plantvillage_split.json'); p.add_argument('--val',type=float,default=.15); p.add_argument('--test',type=float,default=.15); a=p.parse_args()
root=Path(a.input); classes=sorted([x.name for x in root.iterdir() if x.is_dir()]); items=[]
for ci,c in enumerate(classes):
 for f in (root/c).rglob('*'):
  if f.suffix.lower() in {'.jpg','.jpeg','.png','.webp'}:
   try: Image.open(f).verify(); items.append({'path':str(f.resolve()),'label':ci,'class':c})
   except: pass
random.seed(42); random.shuffle(items); n=len(items); nt=int(n*a.test); nv=int(n*a.val)
splits={'classes':classes,'test':items[:nt],'val':items[nt:nt+nv],'train':items[nt+nv:]}
out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(splits,indent=2)); print('Classes:',len(classes),'Images:',n,'Manifest:',out)
