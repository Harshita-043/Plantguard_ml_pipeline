from __future__ import annotations
import torch
from torch import nn
from torchvision import transforms
from PIL import Image

class DinoBinaryFilter(nn.Module):
    def __init__(self, backbone_name='dinov2_vits14'):
        super().__init__(); self.backbone_name=backbone_name
        self.backbone=torch.hub.load('facebookresearch/dinov2',backbone_name)
        dim=self.backbone.embed_dim
        for p in self.backbone.parameters(): p.requires_grad=False
        self.head=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,1))
    def forward(self,x): return self.head(self.backbone(x)).squeeze(1)

def transform(size=224):
    return transforms.Compose([transforms.Resize(size+32),transforms.CenterCrop(size),transforms.ToTensor(),transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
