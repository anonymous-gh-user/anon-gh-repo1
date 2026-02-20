import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from src.models.encoders import _make_backbone, _get_embedding_dims

class BaseCMH(nn.Module):
    """ Concept Multihead Network """
    def __init__(self, 
                 num_concepts=5, 
                 num_classes=5,
                 backbone='resnet18'):
        super(BaseCMH, self).__init__()

        self.backbone = _make_backbone(backbone)
        d = _get_embedding_dims(backbone)
        self.concept_layer = nn.Linear(d, num_concepts)
        self.clf_layer = nn.Linear(d, num_classes)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.relu(x)
        c = self.concept_layer(x)
        y = self.clf_layer(x)
        #y = self.softmax(y)
        c = F.sigmoid(c)
        return c, y # [0-1]*7