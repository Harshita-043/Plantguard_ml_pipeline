from __future__ import annotations
import json
from pathlib import Path
import torch
from torch import nn
from torchvision import transforms

class DinoClassifier(nn.Module):
    def __init__(self, backbone_name: str, num_classes: int, freeze_backbone: bool=True):
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone = torch.hub.load('facebookresearch/dinov2', backbone_name)
        dim = self.backbone.embed_dim
        if freeze_backbone:
            for p in self.backbone.parameters(): p.requires_grad = False
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))
    def forward_features(self, x):
        return self.backbone(x)
    def forward(self, x):
        return self.head(self.forward_features(x))


def get_transform(size=224, train=False):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.65, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.15),
            transforms.ColorJitter(0.25,0.25,0.25,0.05),
            transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
        ])
    return transforms.Compose([
        transforms.Resize(size+32), transforms.CenterCrop(size), transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])


def save_bundle(model, classes, path, backbone_name):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'classes': classes,
                'backbone_name': backbone_name, 'image_size': 224}, path)
    (path.parent/'classes.json').write_text(json.dumps(classes, indent=2))
