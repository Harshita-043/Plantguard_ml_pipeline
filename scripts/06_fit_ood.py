"""Fit class-conditional Mahalanobis statistics from DINO disease embeddings on the training split."""
from pathlib import Path
import argparse,json,yaml,numpy as np,torch
from PIL import Image
from torchvision import transforms
from datasets import load_dataset
from plantguard_ml.models.dino_classifier import DinoClassifier
p=argparse.ArgumentParser();p.add_argument('--config',default='configs/plantguard.yaml');p.add_argument('--manifest',default='data/processed/plantvillage_hf/manifest.json');a=p.parse_args();c=yaml.safe_load(open(a.config));m=json.load(open(a.manifest));ds=load_dataset(m['dataset'],m['config'])['train'];b=torch.load(c['dino']['disease_artifact'],map_location='cpu',weights_only=False);model=DinoClassifier(b['backbone_name'],len(b['classes']));model.load_state_dict(b['state_dict']);model.eval();tf=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize((.485,.456,.406),(.229,.224,.225))]);E=[];Y=[]
with torch.no_grad():
 for i in m['train_indices']:
  z=ds[i];x=tf(z['image'].convert('RGB')).unsqueeze(0);E.append(model.backbone(x)[0].numpy());Y.append(int(z['label']))
E=np.asarray(E);Y=np.asarray(Y);means=np.stack([E[Y==k].mean(0) for k in sorted(set(Y))]);cov=np.cov(E.T)+np.eye(E.shape[1])*1e-3;inv=np.linalg.pinv(cov);d=[]
for x,y in zip(E,Y):q=x-means[y];d.append(q@inv@q.T)
th=float(np.percentile(d,99));out=Path('artifacts/dino_disease/ood.npz');out.parent.mkdir(parents=True,exist_ok=True);np.savez(out,means=means,inv_cov=inv,threshold=th);print('saved',out,'threshold',th)
