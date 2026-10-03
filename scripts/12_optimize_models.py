"""Export supported PyTorch classifier/filter models to ONNX and optionally dynamic-quantize them."""
from pathlib import Path
import argparse,torch
from plantguard_ml.models.dino_classifier import DinoClassifier
from plantguard_ml.models.dino_leaf_filter import DinoBinaryFilter
p=argparse.ArgumentParser();p.add_argument('--classifier',default='artifacts/dino_disease/dino_disease_classifier.pt');p.add_argument('--leaf',default='artifacts/dino_leaf_filter/dino_leaf_filter.pt');a=p.parse_args();out=Path('artifacts/optimized');out.mkdir(parents=True,exist_ok=True)

def export(bundle_path,is_leaf):
 b=torch.load(bundle_path,map_location='cpu',weights_only=False)
 if is_leaf: m=DinoBinaryFilter(b.get('backbone_name','dinov2_vits14'))
 else: m=DinoClassifier(b['backbone_name'],len(b['classes']))
 m.load_state_dict(b['state_dict']);m.eval();x=torch.randn(1,3,224,224)
 name='dino_leaf_filter' if is_leaf else 'dino_disease_classifier'; dest=out/(name+'.onnx')
 prog=torch.onnx.export(m,(x,),dynamo=True,input_names=['image'],output_names=['logits'],dynamic_shapes={'image':{0:'batch'}})
 prog.save(str(dest)); print('saved',dest)
 try:
  import onnx,onnxruntime as ort
  onnx.checker.check_model(onnx.load(dest)); print('ONNX validated; ORT providers:',ort.get_available_providers())
 except Exception as e: print('validation warning',e)
for f,l in [(a.classifier,False),(a.leaf,True)]:
 if Path(f).exists(): export(f,l)
 else: print('missing',f)
