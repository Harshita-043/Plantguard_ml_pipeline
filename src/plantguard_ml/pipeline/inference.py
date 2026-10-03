from __future__ import annotations
import time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from ..models.dino_classifier import DinoClassifier
from ..models.dino_leaf_filter import DinoBinaryFilter, transform as leaf_transform
from ..models.ood import MahalanobisOOD
from ..models.severity import make_features

class PlantGuardPipeline:
    def __init__(self,cfg):
        self.cfg=cfg; self.device='cuda' if cfg['project']['device']=='auto' and torch.cuda.is_available() else 'cpu'
        self.classifier=self._load_classifier(); self.leaf_filter=self._load_leaf_filter(); self.ood=self._load_ood(); self.severity=self._load_severity(); self.sam=None; self.segnext=None
    def _load_classifier(self):
        p=Path(self.cfg['dino']['disease_artifact'])
        if not p.exists(): return None
        b=torch.load(p,map_location='cpu',weights_only=False);m=DinoClassifier(b.get('backbone_name','dinov2_vitb14'),len(b['classes']),True);m.load_state_dict(b['state_dict']);m.classes=b['classes'];return m.to(self.device).eval()
    def _load_leaf_filter(self):
        p=Path(self.cfg['dino']['leaf_filter_artifact'])
        if not p.exists(): return None
        b=torch.load(p,map_location='cpu',weights_only=False);m=DinoBinaryFilter(b.get('backbone_name','dinov2_vits14'));m.load_state_dict(b['state_dict']);return m.to(self.device).eval()
    def _load_ood(self):
        p=Path('artifacts/dino_disease/ood.npz');return MahalanobisOOD.load(p) if p.exists() else None
    def _load_severity(self):
        try:
            import xgboost as xgb
            p=Path('artifacts/severity/xgb_regressor.json')
            if p.exists():m=xgb.XGBRegressor();m.load_model(str(p));return m
        except Exception:pass
        return None
    def attach_heavy_models(self,sam,segnext):self.sam=sam;self.segnext=segnext
    @torch.inference_mode()
    def _leaf_score(self,crop):
        if self.leaf_filter is None:return 1.0
        x=leaf_transform()(crop).unsqueeze(0).to(self.device);return float(torch.sigmoid(self.leaf_filter(x))[0])
    @torch.inference_mode()
    def _classify(self,pil):
        x=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize((.485,.456,.406),(.229,.224,.225))])(pil).unsqueeze(0).to(self.device);logits=self.classifier(x);probs=logits.softmax(1)[0];idx=int(probs.argmax());emb=self.classifier.backbone(x)[0].detach().cpu().numpy();return idx,float(probs[idx]),emb
    def run(self,pil):
        t=time.time();rgb=np.array(pil.convert('RGB'))
        if self.sam is None:raise RuntimeError('SAM2 adapter not attached')
        candidates=self.sam.candidate_masks(rgb,self.cfg['sam2']['min_area_ratio'],self.cfg['sam2']['max_area_ratio'])
        results=[]
        for cand in candidates:
            mask=cand['mask'];ys,xs=np.where(mask)
            if len(xs)<10:continue
            crop=pil.crop((xs.min(),ys.min(),xs.max()+1,ys.max()+1));leaf_score=self._leaf_score(crop)
            if leaf_score<.5:continue
            if self.classifier is None:continue
            idx,conf,embedding=self._classify(crop); disease_mask=self.segnext.predict(np.array(crop.convert('RGB'))) if self.segnext else np.zeros((crop.height,crop.width),np.uint8)
            ood=self.ood.score(embedding) if self.ood else {'ood':False,'distance':None,'threshold':None}
            sev=None
            if self.severity is not None:
                f=make_features(np.ones(disease_mask.shape,bool),disease_mask,np.array([conf]),1,conf);sev=float(np.clip(self.severity.predict(f.reshape(1,-1))[0],0,100))
            results.append({'leaf_score':leaf_score,'sam_score':cand['score'],'sam_stability':cand['stability'],'leaf_area_px':int(mask.sum()),'bbox':cand['bbox'],'disease_index':idx,'disease':self.classifier.classes[idx],'disease_confidence':conf,'disease_area_px':int((disease_mask>0).sum()),'severity':sev,'ood':ood})
            if len(results)>=5:break
        return {'status':'ok' if results else 'no_valid_leaf','leaves':results,'latency_ms':(time.time()-t)*1000,'device':self.device}
