import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from open_clip import create_model_from_pretrained, get_tokenizer

class BiomedCLIP(nn.Module):
    """
    Trainable BiomedCLIP model for medical image and text report processing.

    Methods: 
        forward(img_names, text_reports): Processes images and text reports to extract features.
        apply_clip_loss(im_feats, text_feats): Computes the CLIP loss between image and text features.
    """
    def __init__(self, conf):
        super(BiomedCLIP, self).__init__()
        self.model, self.preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        self.tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        
        self.device = conf.device
        self.model.to(self.device)
        self.context_length = 256

        self.architecture = conf.cbm.architecture
        if conf.cbm.architecture == 'cbl':
            self.classifier = nn.Sequential(
                nn.Linear(conf.cbm.num_concepts,
                          conf.cbm.num_classes, bias=True),
            ).to(self.device)

        elif conf.cbm.architecture == 'multihead':
            self.classifier = nn.Sequential(
                nn.Linear(conf.clip.d, 
                          conf.cbm.num_classes, bias=True)
            ).to(self.device)

        elif conf.cbm.architecture == 'fusion':
            # Fusion architecture combines image and concept features
            self.classifier = nn.Sequential(
                nn.Linear(conf.clip.d + conf.cbm.num_concepts,
                          conf.cbm.num_classes, bias=True),
            ).to(self.device)

        else:
            # cancer vs. benign classifier (binary classification)
            self.classifier = nn.Linear(conf.clip.d, conf.cbm.num_classes, bias=True).to(self.device)

        # birads classifier (multi-class classification)
        self.birads_classifier = nn.Sequential(
            nn.Linear(conf.cbm.num_concepts, conf.cbm.num_birads, bias=True),
        ).to(self.device)

        self.concept_classifiers = []  # Initialize with empty list
        # Create a separate classifier for each concept
        # Each classifier will have a linear layer followed by ReLU and another linear layer
        for i in range(conf.cbm.num_concepts):
            self.concept_classifiers.append(
                nn.Sequential(
                    nn.Linear(conf.clip.d, conf.clip.adapter_dim),
                    nn.ReLU(),
                    nn.Linear(conf.clip.adapter_dim, 1)
                ).to(self.device)
            )

    def eval(self):
        """
        Set the model to evaluation mode.
        """
        self.model.eval()
        for classifier in self.concept_classifiers:
            classifier.eval()
        self.classifier.eval()
        self.birads_classifier.eval()
        return self

    def train(self, mode=True):
        """
        Set the model to training mode.
        """
        self.model.train(mode)
        for classifier in self.concept_classifiers:
            classifier.train(mode)
        self.classifier.train(mode)
        self.birads_classifier.train(mode)
        return self

    def get_param_groups(self):
        """
        Get parameter groups for the optimizer.
        """
        return [
            {'params': self.model.parameters(), 'lr': 1e-4},
            {'params': self.tokenizer.parameters(), 'lr': 1e-4}
        ]

    def forward(self, img_names, text_reports):
        text_reports = [t[0] if isinstance(t, list) else t for t in text_reports]
        images = torch.stack([self.preprocess(Image.open(img)) for img in img_names]).to(self.device)
        text_tokens = self.tokenizer(text_reports, context_length=self.context_length).to(self.device)
        image_feats, text_feats, logit_scale = self.model(images, text_tokens)
        return image_feats, text_feats, logit_scale
    
    def get_cancer_pred(self, image_feats):
        """
        Forward pass through the classifier to get predictions for cancer vs. benign classification.
        """
        # Classify using image features
        if self.architecture == 'cbl':
            # If using CBL architecture, get concept predictions first
            predicted_concepts = self.get_concept_preds(image_feats)
            return self.classifier(predicted_concepts)
        elif self.architecture == 'multihead':
            # If using multihead architecture, directly classify image features
            return self.classifier(image_feats)
        elif self.architecture == 'fusion':
            # If using fusion architecture, concatenate image features with concept predictions
            concept_preds = self.get_concept_preds(image_feats)
            combined_feats = torch.cat((image_feats, concept_preds), dim=1)
            return self.classifier(combined_feats)

        return self.classifier(image_feats)
    
    def get_concept_preds(self, image_feats):
        """
        Forward pass through the concept classifiers to get predictions for each concept.
        """
        concept_preds = []
        for classifier in self.concept_classifiers:
            # Classify using image features
            concept_pred = classifier(image_feats)
            concept_preds.append(concept_pred)

        return torch.cat(concept_preds, dim=1)
    
    def get_birads_pred(self, concept_preds):
        """
        Forward pass through the birads classifier to get predictions for BIRADS classification.
        """
        # Classify using concept predictions
        birads_preds = self.birads_classifier(concept_preds)
        return birads_preds

    def apply_clip_loss(self, im_feats, text_feats):
        # Normalize features
        im_feats = im_feats / im_feats.norm(dim=-1, keepdim=True)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

        # Compute cosine similarity
        logits = im_feats @ text_feats.T

        # Apply softmax to get probabilities
        probs = logits.softmax(dim=-1)

        return probs


    def apply_clip_loss(self, image_features, text_features, temperature=0.07):
        """
        Compute the CLIP-style contrastive loss between image and text embeddings.
        """
        # Normalize features
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        # Cosine similarity matrix (batch_size x batch_size)
        logits_per_image = image_features @ text_features.T
        logits_per_text = text_features @ image_features.T

        # Scale by temperature
        logits_per_image = logits_per_image / temperature
        logits_per_text = logits_per_text / temperature

        # Labels: assume positives are on the diagonal
        batch_size = image_features.size(0)
        labels = torch.arange(batch_size, device=image_features.device)

        # Cross entropy loss
        loss_i2t = F.cross_entropy(logits_per_image, labels, reduction='mean')  # image -> text
        loss_t2i = F.cross_entropy(logits_per_text, labels, reduction='mean')  # text -> image

        return (loss_i2t + loss_t2i) / 2 
