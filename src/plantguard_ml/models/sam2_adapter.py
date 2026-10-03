from __future__ import annotations
from pathlib import Path
import numpy as np
import torch

class SAM2LeafSegmenter:
    """Automatic image segmentation using the official SAM2 automatic mask generator."""
    def __init__(self, checkpoint, config, device='auto', points_per_side=24):
        self.checkpoint=Path(checkpoint); self.config=config
        self.device='cuda' if device=='auto' and torch.cuda.is_available() else ('cpu' if device=='auto' else device)
        if not self.checkpoint.exists(): raise FileNotFoundError(f'SAM2 checkpoint missing: {self.checkpoint}')
        try:
            from sam2.build_sam import build_sam2
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        except ImportError as e: raise RuntimeError('Install SAM2: pip install -e external/sam2') from e
        model=build_sam2(config,self.checkpoint,device=self.device)
        self.generator=SAM2AutomaticMaskGenerator(model,points_per_side=points_per_side,pred_iou_thresh=0.75,stability_score_thresh=0.90,min_mask_region_area=200)
    def generate(self,image_rgb):
        return self.generator.generate(image_rgb)
    def candidate_masks(self,image_rgb,min_area_ratio=.01,max_area_ratio=.85,max_candidates=12):
        h,w=image_rgb.shape[:2]; total=h*w; anns=self.generate(image_rgb); out=[]
        for a in anns:
            mask=np.asarray(a['segmentation']).astype(bool); ratio=mask.mean()
            if min_area_ratio<=ratio<=max_area_ratio:
                out.append({'mask':mask,'score':float(a.get('predicted_iou',0.0)),'stability':float(a.get('stability_score',0.0)),'area_ratio':float(ratio),'bbox':a.get('bbox')})
        out.sort(key=lambda z:z['score']*0.7+z['stability']*0.3,reverse=True)
        return out[:max_candidates]
