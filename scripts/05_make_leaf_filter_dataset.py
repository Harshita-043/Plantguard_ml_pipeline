"""Create a bootstrap leaf-filter dataset. Positive images are copied from PlantVillage; negatives come from a user-provided folder.
Do not treat the bootstrap set as final research-grade data; add real-field non-leaf images manually."""
from pathlib import Path
import argparse,shutil,random
p=argparse.ArgumentParser();p.add_argument('--plantvillage',required=True);p.add_argument('--negative',required=True);p.add_argument('--out',default='data/processed/leaf_filter');p.add_argument('--limit',type=int,default=5000);a=p.parse_args();o=Path(a.out);(o/'leaf').mkdir(parents=True,exist_ok=True);(o/'non_leaf').mkdir(parents=True,exist_ok=True)
files=[x for x in Path(a.plantvillage).rglob('*') if x.suffix.lower() in {'.jpg','.jpeg','.png'}]; random.seed(42);random.shuffle(files)
for i,f in enumerate(files[:a.limit]): shutil.copy2(f,o/'leaf'/f'pv_{i}{f.suffix.lower()}')
neg=[x for x in Path(a.negative).rglob('*') if x.suffix.lower() in {'.jpg','.jpeg','.png'}];random.shuffle(neg)
for i,f in enumerate(neg[:a.limit]): shutil.copy2(f,o/'non_leaf'/f'neg_{i}{f.suffix.lower()}')
print('Created',o)
