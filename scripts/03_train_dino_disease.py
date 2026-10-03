from pathlib import Path
import argparse,yaml,json,torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from plantguard_ml.models.dino_classifier import DinoClassifier,save_bundle
from datasets import load_dataset

class HFDataset(Dataset):
 def __init__(self,ds,indices,train):
  self.ds=ds;self.indices=indices;self.tf=transforms.Compose([transforms.RandomResizedCrop(224,scale=(.65,1)),transforms.RandomHorizontalFlip(),transforms.ColorJitter(.25,.25,.25,.05),transforms.ToTensor(),transforms.Normalize((.485,.456,.406),(.229,.224,.225))]) if train else transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
 def __len__(self):return len(self.indices)
 def __getitem__(self,i):
  z=self.ds[self.indices[i]];return self.tf(z['image'].convert('RGB')),int(z['label'])

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',default='configs/plantguard.yaml');p.add_argument('--manifest',default='data/processed/plantvillage_hf/manifest.json');a=p.parse_args();c=yaml.safe_load(open(a.config));m=json.load(open(a.manifest));ds=load_dataset(m['dataset'],m['config']);train_ds=HFDataset(ds['train'],m['train_indices'],1);val_ds=HFDataset(ds['train'],m['val_indices'],0);dev='cuda' if torch.cuda.is_available() else 'cpu'
 model=DinoClassifier(c['dino']['backbone'],len(m['classes']),c['dino']['freeze_backbone']).to(dev);opt=torch.optim.AdamW([x for x in model.parameters() if x.requires_grad],lr=c['dino']['lr'],weight_decay=c['dino']['weight_decay']);lossfn=nn.CrossEntropyLoss(label_smoothing=c['dino']['label_smoothing']);scaler=torch.amp.GradScaler('cuda',enabled=dev=='cuda');tr=DataLoader(train_ds,c['dino']['batch_size'],shuffle=True,num_workers=c['dino']['workers'],pin_memory=True);va=DataLoader(val_ds,c['dino']['batch_size'],num_workers=c['dino']['workers']);best=0
 for ep in range(c['dino']['epochs']):
  model.train();seen=correct=0
  for x,y in tr:
   x,y=x.to(dev),y.to(dev);opt.zero_grad(set_to_none=True)
   with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=dev=='cuda'):z=model(x);l=lossfn(z,y)
   scaler.scale(l).backward();scaler.step(opt);scaler.update();seen+=len(y);correct+=(z.argmax(1)==y).sum().item()
  model.eval();vc=vn=0
  with torch.no_grad():
   for x,y in va:x,y=x.to(dev),y.to(dev);vc+=(model(x).argmax(1)==y).sum().item();vn+=len(y)
  acc=vc/max(vn,1);print(f'epoch {ep+1}/{c["dino"]["epochs"]} train_acc={correct/seen:.4f} val_acc={acc:.4f}')
  if acc>best:best=acc;save_bundle(model,m['classes'],c['dino']['disease_artifact'],c['dino']['backbone'])
 print('best',best)
if __name__=='__main__':main()
