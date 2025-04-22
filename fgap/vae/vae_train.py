#!/usr/bin/env python3
import os
import glob
import argparse
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# 自定义数据集：读取指定目录下所有png图片
class LineChartDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_paths = glob.glob(os.path.join(image_dir, "*.png"))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert('L')  # 转换为灰度图
        if self.transform:
            image = self.transform(image)
        return image

# 定义 VAE 模型
class VAE(nn.Module):
    def __init__(self, latent_dim=256):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        # Encoder 部分（卷积部分）
        self.encoder = nn.Sequential(
            # 输入：(batch, 1, 64, 64)
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # -> (batch, 32, 32, 32)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # -> (batch, 64, 16, 16)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # -> (batch, 128, 8, 8)
            nn.ReLU(),
            nn.Flatten()  # -> (batch, 128*8*8)
        )
        # fc_mu: 将卷积部分输出映射到 latent 维度
        self.fc_mu = nn.Linear(128 * 8 * 8, latent_dim)
        # fc_logvar 在重参数化时使用，但此处不用于特征提取
        self.fc_logvar = nn.Linear(128 * 8 * 8, latent_dim)

        # Decoder 部分
        self.decoder_input = nn.Linear(latent_dim, 128 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.Unflatten(dim=1, unflattened_size=(128, 8, 8)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # -> (batch, 64, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # -> (batch, 32, 32, 32)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),     # -> (batch, 1, 64, 64)
            nn.Sigmoid()  # 输出范围[0,1]
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        z = self.reparameterize(mu, logvar)
        dec_in = self.decoder_input(z)
        reconstruction = self.decoder(dec_in)
        return reconstruction, mu, logvar

def loss_function(recon_x, x, mu, logvar):
    # 重构损失采用 BCE（对二值化图片比较合适，如不合适可改为 MSELoss）
    bce = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    # KL 散度
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kld

def main(args):
    # 数据预处理：调整尺寸并转化为tensor
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    dataset = LineChartDataset(args.data_dir, transform=transform)
    if len(dataset) == 0:
        print("未检测到png格式图片，请检查指定目录：", args.data_dir)
        return

    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vae = VAE(latent_dim=256).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)

    num_epochs = 50  # 根据数据量设置合适的 epoch 数
    vae.train()
    for epoch in range(1, num_epochs + 1):
        train_loss = 0
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            reconstruction, mu, logvar = vae(batch)
            loss = loss_function(reconstruction, batch, mu, logvar)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"Epoch {epoch}, Average Loss: {train_loss / len(dataset):.4f}")

    # 修改保存方式：保存编码器的完整部分（卷积和 fc_mu）
    encoder_path = "/mnt/e/Dr/FGAP/code/log_del/final/vae/encoder.pth"
    torch.save({
        'encoder': vae.encoder.state_dict(),
        'fc_mu': vae.fc_mu.state_dict()
    }, encoder_path)
    print(f"Encoder network 已保存到 {encoder_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VAE on line chart images.")
    parser.add_argument("data_dir", type=str, help="包含png图片的目录")
    args = parser.parse_args()
    main(args)