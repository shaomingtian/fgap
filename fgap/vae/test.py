#!/usr/bin/env python3
import argparse
from vae_get_feature import get_vae_feature

def main(args):
    feature = get_vae_feature(args.image_path)
    # 将特征 tensor 转换为列表并输出
    feature_list = feature.cpu().numpy().flatten()
    print("提取到的 256 维特征向量：")
    print(feature_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用预训练的 encoder 提取图片特征")
    parser.add_argument("image_path", type=str, help="待测试的 png 图片路径")
    args = parser.parse_args()
    main(args)