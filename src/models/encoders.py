import torch
import torch.nn.functional as F
from torch import load, nn
from torchvision import models

from open_clip import create_model_from_pretrained, create_model_and_transforms, get_tokenizer
from src.utils.helpers import LayerNorm2d

from src.models.usfm import get_usfm_backbone
from src.models.vit import make_vit

from src.models.clip import CLIPRN50, CLIPViT, CLIPViTL, SigLIP, BiomedCLIP

from PIL import Image

SAM_CHN_DIM = 64
SAM_EMB_DIM = 16

def _make_backbone(style):
    if style == 'resnet18':
        return ResNet18Encoder()
    if style == 'resnet50':
        return ResNet50Encoder()
    if style == 'clip-rn50':
        return CLIPResNet50Encoder()
    if style == 'sam':
        return SAMEncoder()
    if style == 'medsam':
        return MedSAMEncoder()
    if style == 'usfm':
        return USFMEncoder()
    if style == 'vit-t':
        return ViTEncoder('t')
    if style == 'vit-s':
        return ViTEncoder('s')
    if style == 'vit-b':
        return ViTEncoder('b')
    if style == 'biomedclip':
        return BiomedCLIPViTEncoder()
    if style == 'siglip':
        return SigLIPEncoder()
    if style == 'clip-vit':
        return CLIPViTEncoder()
    if style == 'clip-vitl':
        return CLIPViTLEncoder()
    
    else:
        print(f"Unknown encoder {style}. Defaulting to resnet18.")
        return ResNet18Encoder()

def _get_embedding_dims(style):
    return {
        'resnet18': 512, 'resnet50': 2048,
        'clip-rn50': 512, 
        'clip-vit': 768,
        'sam': SAM_CHN_DIM * SAM_EMB_DIM * SAM_EMB_DIM,
        'medsam': 256,
        'usfm': SAM_CHN_DIM * SAM_EMB_DIM * SAM_EMB_DIM,
        'vit-t': 192,
        'vit-s': 384, 
        'vit-b': 768,
        'biobert': 512,
        'biomedclip': 512,
        'siglip': 1024,
    }[style]

class ResNet18Encoder(nn.Module):
    def __init__(self):
        super(ResNet18Encoder, self).__init__()
        self.model = models.resnet18(weights='DEFAULT')
        self.model.fc = nn.Identity()

    def forward(self, x):
        return self.model(x)

class ResNet50Encoder(nn.Module):
    def __init__(self):
        super(ResNet50Encoder, self).__init__()
        self.model = models.resnet50(weights='DEFAULT')
        self.model.fc = nn.Identity()

    def forward(self, x):
        return self.model(x)
    
class USFMEncoder(nn.Module):
    def __init__(self):
        super(USFMEncoder, self).__init__()
        self.model = get_usfm_backbone(pretrained_path='/h/ANON-USER/checkpoint/USFM_latest.pth')

    def forward(self, x):
        feat = self.model(x)
        return torch.stack(feat)
    
class BiomedCLIPEncoder(nn.Module):
    def __init__(self, device='cuda'):
        super(BiomedCLIPEncoder, self).__init__()
        self.model, self.preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        self.tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(device)
        #self.model.eval()
        self.device = device
        self.context_length = 256

        self.classifier = nn.Linear(12, 2)


    def forward(self, img_names):
        prompt = [
            "Shadowing",
            "Enhancement",
            "Halo",
            "Calcifications",
            "Skin thickening",
            "Circumscribed margins",
            "Spiculated margins",
            "Indistinct margins",
            "Angular margins",
            "Microlobulated margins",
            "Regular shape",
            "Hyperechoic",
            "Hypoechoic",
            "Heterogeneous",
            "Cystic"
        ]

        images = []
        for im in img_names:
            if isinstance(im, str):
                im = Image.open(im)
            elif isinstance(im, torch.Tensor):
                im = im.cpu().numpy()
                im = Image.fromarray(im)
            else:
                raise ValueError("Unsupported input type. Expected str or torch.Tensor.")
            images.append(im)

        # Preprocess images
        xs = []
        for i in range(len(images)):
            if images[i].mode != 'RGB':
                images[i] = images[i].convert('RGB')
                x = self.preprocess(images[i]).to(self.device)
                xs.append(x)

        text = self.tokenizer(prompt, context_length=self.context_length).to(self.device)
        cs = []
        with torch.no_grad():
            for x in xs:
                image_features, text_features, logit_scale = self.model(x.unsqueeze(0), text)
                concept_logits = (logit_scale * image_features @ text_features.t()).detach()#.softmax(dim=-1)
                cs.append(concept_logits)
        cs = torch.stack(cs)

        return cs

class BiomedCLIPViTEncoder(nn.Module):
    def __init__(self, device='cuda'):
        super(BiomedCLIPViTEncoder, self).__init__()
        self.model, _ = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(device)
        self.device = device

    def forward(self, x):
        image_features = self.model.visual(x)
        return image_features

class CLIPViTEncoder(nn.Module):
    def __init__(self, device='cuda'):
        super(CLIPViTEncoder, self).__init__()
        self.model, _, preprocess = create_model_and_transforms(
            model_name="ViT-B-32", 
            pretrained="laion2b_e16", 
            device=device
        )
        
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(device)
        self.device = device

    def forward(self, images):
        images = images.to(self.device)
        return self.model.encode_image(images)

class CLIPResNet50Encoder(nn.Module):
    def __init__(self, device='cuda'):
        super(CLIPResNet50Encoder, self).__init__()
        self.model, _, preprocess = create_model_and_transforms(
            model_name="RN50", 
            pretrained="openai", 
            device=device
        )
        
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(device)
        self.device = device

    def forward(self, images):
        images = images.to(self.device)
        return self.model.encode_image(images)

class SigLIPEncoder(nn.Module):
    def __init__(self, model_name='hf-hub:timm/ViT-L-16-SigLIP-384', device='cuda'):
        super(SigLIPEncoder, self).__init__()
        # load the SigLIP model (vision+text)
        self.model, preprocess = create_model_from_pretrained(
            model_name,
            device=device
        )
        self.device = device
        self.preprocess = preprocess

    def forward(self, images):
        images = images.to(self.device)
        return self.model.encode_image(images)

class ViTEncoder(nn.Module):
    def __init__(self, size):
        super(ViTEncoder, self).__init__()
        self.model = make_vit(size)
        self.model.mlp_head = nn.Identity()
        self.mlp_head = nn.Linear(192, 2)

    def forward(self, x):
        feat = self.model(x)
        return feat

class SAMEncoder(nn.Module):
    def __init__(self):
        super(SAMEncoder, self).__init__()
        self.model = build_sam()
        self.image_size_for_features = 1024
        self.compressor = nn.AdaptiveAvgPool2d(1)

        #self.compressor = nn.Sequential(
        #    nn.ConvTranspose2d(
        #        d, d // 4, kernel_size=2, stride=2
        #    ),
        #    LayerNorm2d(d // 4),
        #    nn.GELU(),
        #    nn.ConvTranspose2d(
        #        d // 4, d // 8, kernel_size=2, stride=2
        #    ),
        #    nn.GELU(),
        #)

    def forward(self, image):
        B, C, H, W = image.shape
        if H != self.image_size_for_features or W != self.image_size_for_features:
            image_resized_for_features = torch.nn.functional.interpolate(
                image, size=(self.image_size_for_features, self.image_size_for_features)
            )
        else:
            image_resized_for_features = image
        image_feats = self.model.image_encoder(image_resized_for_features.float())
        image_feats = self.compressor(image_feats) # Now Bx64x64x64
        #image_feats = F.adaptive_avg_pool2d(image_feats, (SAM_EMB_DIM, SAM_EMB_DIM))  # Now Bx64x16x16

        return image_feats.reshape(B, -1)

class MedSAMEncoder(SAMEncoder):
    def __init__(self):
        super(MedSAMEncoder, self).__init__()
        self.model = build_medsam()
        self.image_size_for_features = 1024