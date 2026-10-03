"""Optional CPU dynamic quantization for the linear-heavy classifier heads.
This is an optimization experiment, not a replacement for validating accuracy and latency on your target machine.
"""
from pathlib import Path
import torch
from plantguard_ml.models.dino_classifier import DinoClassifier
from plantguard_ml.models.dino_leaf_filter import DinoBinaryFilter

def q(path,leaf):
 b=torch.load(path,map_location='cpu',weights_only=False)
 m=DinoBinaryFilter(b.get('backbone_name','dinov2_vits14')) if leaf else DinoClassifier(b['backbone_name'],len(b['classes']))
 m.load_state_dict(b['state_dict']);m.eval()
 qmodel=torch.quantization.quantize_dynamic(m,{torch.nn.Linear},dtype=torch.qint8)
 out=Path('artifacts/optimized')/('dino_leaf_filter_dynamic_int8.pt' if leaf else 'dino_disease_dynamic_int8.pt');out.parent.mkdir(parents=True,exist_ok=True)
 torch.save({'model':qmodel,'classes':b.get('classes'),'backbone_name':b.get('backbone_name')},out);print(out)
for p,l in [('artifacts/dino_disease/dino_disease_classifier.pt',False),('artifacts/dino_leaf_filter/dino_leaf_filter.pt',True)]:
 if Path(p).exists(): q(p,l)
