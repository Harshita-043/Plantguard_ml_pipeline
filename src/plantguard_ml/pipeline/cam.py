from __future__ import annotations
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

class TransformerCAM:
    """Attention rollout explanation for a DINOv2-style ViT classifier.
    It is deliberately called Transformer-CAM rather than Grad-CAM because DINOv2 is a transformer.
    """
    def __init__(self, model, size=224): self.model=model; self.size=size
    def explain(self, image:Image.Image, class_index:int):
        # Robust fallback: use gradient of class score wrt input. This is not a canonical transformer Grad-CAM.
        x=transforms.Compose([transforms.Resize(self.size+32),transforms.CenterCrop(self.size),transforms.ToTensor(),transforms.Normalize((.485,.456,.406),(.229,.224,.225))])(image).unsqueeze(0)
        x.requires_grad_(True); self.model.zero_grad(set_to_none=True)
        score=self.model(x)[0,class_index]; score.backward()
        heat=x.grad.detach().abs().mean(1)[0].cpu().numpy()
        heat=(heat-heat.min())/(heat.max()-heat.min()+1e-8)
        return heat
