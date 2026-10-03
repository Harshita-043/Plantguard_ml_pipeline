"""Download PlantVillage from Hugging Face and create a leakage-safe manifest using leaf_id.
This is the preferred preparation route because the upstream dataset exposes leaf-grouping metadata and predefined train/test splits."""
from pathlib import Path
import argparse,json,random
from datasets import load_dataset
p=argparse.ArgumentParser();p.add_argument('--out',default='data/processed/plantvillage_hf');p.add_argument('--val',type=float,default=.15);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
ds=load_dataset('mohanty/PlantVillage','color'); train=ds['train']; test=ds['test']
leaf_ids=list(dict.fromkeys(train['leaf_id']));random.seed(42);random.shuffle(leaf_ids);n=int(len(leaf_ids)*a.val);val_ids=set(leaf_ids[:n])
def row(ex,i):
 return {'hf_split':str(ex['_split']) if '_split' in ex else '', 'index':i,'label':int(ex['label']),'class':str(ex['label']),'leaf_id':str(ex['leaf_id'])}
# Save compact manifests. Training script needs HF dataset access, so use indices.
val_idx=[i for i,x in enumerate(train) if x['leaf_id'] in val_ids]; tr_idx=[i for i,x in enumerate(train) if x['leaf_id'] not in val_ids]
classes=train.features['label'].names
manifest={'dataset':'mohanty/PlantVillage','config':'color','classes':classes,'train_indices':tr_idx,'val_indices':val_idx,'test_size':len(test)}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2));print('saved',out/'manifest.json','classes',len(classes),'train',len(tr_idx),'val',len(val_idx),'official test',len(test))
