import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from src.models.encoders import _make_backbone, _get_embedding_dims
    
class ResNet(nn.Module):
    def __init__(self, 
                 num_classes=5,
                 num_birads=4,
                 multitask=False,
                 backbone='resnet50'):
        super(ResNet, self).__init__()

        self.backbone = _make_backbone(backbone)
        d = _get_embedding_dims(backbone)
        self.clf_layer = nn.Linear(d, num_classes)
        self.birads_layer = nn.Linear(d, num_birads)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.multitask = multitask
    
    def forward(self, x):
        h = self.backbone(x)
        b = self.birads_layer(h)
        y = self.clf_layer(h)
        y = self.softmax(y)
        return y, b