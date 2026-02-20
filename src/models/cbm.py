import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from src.models.encoders import BiomedCLIPEncoder, _make_backbone, _get_embedding_dims

class BaseCBM(nn.Module):
    def __init__(self, 
                 num_concepts=5, 
                 num_classes=5,
                 backbone='resnet18'):
        super(BaseCBM, self).__init__()

        self.backbone = _make_backbone(backbone)
        d = _get_embedding_dims(backbone)
        self.concept_layer = nn.Linear(d, num_concepts)
        self.clf_layer = nn.Linear(num_concepts, num_classes)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def load_backbone_weights(self, path: str):
        # replace 'backbone' with '' in the state_dict keys
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items()}
        # remove clf_layer keys 
        state_dict = {k: v for k, v in state_dict.items() if 'clf_layer' not in k}
        self.backbone.load_state_dict(state_dict)

    def extract_concepts(self, x):
        x = self.backbone(x)
        x = self.relu(x)
        c = self.concept_layer(x)
        return c
    
    def forward(self, x):
        c = self.extract_concepts(x)
        c = F.sigmoid(c)
        y = self.clf_layer(c)
        # y = self.softmax(y)
        return c, y

class FusionCBM(nn.Module):
    def __init__(self, 
                 num_concepts=5, 
                 num_classes=5,
                 backbone='resnet18'):
        super(FusionCBM, self).__init__()

        self.backbone = _make_backbone(backbone)
        d = _get_embedding_dims(backbone)
        self.concept_layer = nn.Linear(d, num_concepts)
        self.clf_layer = nn.Linear(d+num_concepts, num_classes)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
    
    def forward(self, x):
        emb = self.backbone(x)
        c = self.concept_layer(emb)
        emb = torch.concat([emb, c], dim=1)
        y = self.clf_layer(emb)
        y = self.softmax(y)
        return c, y # [0-1]*7