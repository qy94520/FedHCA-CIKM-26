import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ResNet import resnet18_cbam

class MobileNetV2FeatureExtractor(nn.Module):

    def __init__(self):
        super().__init__()
        try:
            from torchvision.models import mobilenet_v2
            try:
                model = mobilenet_v2(weights=None)
            except TypeError:
                model = mobilenet_v2(pretrained=False)
            self.features = model.features
            self.out_dim = int(model.last_channel)
        except Exception:
            self.features = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU6(inplace=True), nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, groups=32, bias=False), nn.BatchNorm2d(64), nn.ReLU6(inplace=True), nn.Conv2d(64, 128, kernel_size=1, bias=False), nn.BatchNorm2d(128), nn.ReLU6(inplace=True), nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, groups=128, bias=False), nn.BatchNorm2d(128), nn.ReLU6(inplace=True), nn.Conv2d(128, 1280, kernel_size=1, bias=False), nn.BatchNorm2d(1280), nn.ReLU6(inplace=True))
            self.out_dim = 1280
        self.train(True)

    def train(self, mode: bool=True):
        super().train(mode)
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        return self

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        return torch.flatten(x, 1)

class CIFARViTFeatureExtractor(nn.Module):

    def __init__(self, img_size: int=32, patch_size: int=4, embed_dim: int=192, depth: int=6, num_heads: int=3, mlp_ratio: float=4.0):
        super().__init__()
        img_size = int(img_size)
        patch_size = int(patch_size)
        if img_size % patch_size != 0:
            raise ValueError(f'img_size={img_size} must be divisible by patch_size={patch_size}')
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = int(embed_dim)
        self.patch_embed = nn.Conv2d(3, self.embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=int(num_heads), dim_feedforward=int(self.embed_dim * float(mlp_ratio)), dropout=0.0, activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(depth))
        self.norm = nn.LayerNorm(self.embed_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                fan_out //= module.groups
                nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.encoder(x)
        return self.norm(x[:, 0])

def normalize_backbone_name(name: str) -> str:
    value = str(name or 'resnet18').strip().lower().replace('-', '_')
    aliases = {'default': 'resnet18', 'resnet': 'resnet18', 'resnet18_cbam': 'resnet18', 'current': 'resnet18', 'mobile_net_v2': 'mobilenetv2', 'mobilenet_v2': 'mobilenetv2', 'vit': 'vit_tiny', 'vit_tiny_cifar': 'vit_tiny'}
    return aliases.get(value, value)

def build_backbone(name: str='resnet18', img_size: int=32) -> nn.Module:
    name = normalize_backbone_name(name)
    if name == 'resnet18':
        return resnet18_cbam()
    if name == 'mobilenetv2':
        return MobileNetV2FeatureExtractor()
    if name == 'vit_tiny':
        return CIFARViTFeatureExtractor(img_size=int(img_size), patch_size=4, embed_dim=192, depth=6, num_heads=3)
    raise ValueError(f'Unsupported backbone: {name}')
