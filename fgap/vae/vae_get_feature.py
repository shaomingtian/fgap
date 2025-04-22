#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# 定义与训练时一致的 Encoder 结构：包含卷积部分和 fc_mu 层
class Encoder(nn.Module):
    def __init__(self, latent_dim=256):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            # 输入：64x64, 单通道
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # -> (32, 32, 32)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),   # -> (64, 16, 16)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # -> (128, 8, 8)
            nn.ReLU(),
            nn.Flatten()  # -> (128*8*8)
        )
        self.fc_mu = nn.Linear(128 * 8 * 8, latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        return mu

# 为避免重复加载，全局缓存 encoder 实例
_encoder = None

def load_encoder(model_path="encoder.pth", device=None):
    global _encoder
    if _encoder is None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Encoder(latent_dim=256).to(device)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Encoder 模型文件未找到：{model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        # 加载卷积部分和 fc_mu 参数
        model.encoder.load_state_dict(checkpoint['encoder'])
        model.fc_mu.load_state_dict(checkpoint['fc_mu'])
        model.eval()
        _encoder = (model, device)
    return _encoder

def get_vae_feature(image_path, encoder_model_path="/mnt/e/Dr/FGAP/code/log_del/final/vae/encoder.pth"):
    """
    输入：
      image_path: 图片的路径
      encoder_model_path: encoder 模型参数的路径，默认为当前目录下的 "encoder.pth"
    输出：
      256 维的特征向量，类型为 torch.Tensor
    """
    model, device = load_encoder(model_path=encoder_model_path)

    # 图片预处理：转换为灰度图，调整尺寸到 64x64，并转为 tensor
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert('L')
    image = transform(image).unsqueeze(0).to(device)  # 形状: (1, 1, 64, 64)

    with torch.no_grad():
        feature = model(image)
    # 返回一维 tensor (256,) 的特征向量
    return feature.squeeze(0)