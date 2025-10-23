import os
import csv
import argparse
import torch
import numpy as np
import json
from PIL import Image, ImageFile
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler, Dataset
from torchvision import transforms

# --- 导入ViT模型 ---
from model import VisionTransformer, CONFIGS

# --- 允许 PIL 加载可能被截断的图像 ---
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ================================================================
# Dataset 类
# ================================================================
class TestDataset(Dataset):
    def __init__(self, data_dir, transform):
        self.data_dir = data_dir
        self.transform = transform
        IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
        try:
            all_files = os.listdir(data_dir)
            self.image_files = sorted([
                f for f in all_files
                if os.path.isfile(os.path.join(data_dir, f)) and f.lower().endswith(IMG_EXTENSIONS)
            ])
            if not self.image_files:
                print(f"警告: 在目录 {data_dir} 中未找到任何有效的图片文件。")
        except Exception as e:
            print(f"无法读取测试目录 {data_dir}。错误: {e}")
            self.image_files = []

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        try:
            img_name = self.image_files[idx]
            img_path = os.path.join(self.data_dir, img_name)
            with Image.open(img_path) as image:
                image = image.convert('RGB')
            if self.transform:
                image_tensor = self.transform(image)
            else:
                image_tensor = transforms.ToTensor()(image)
            return image_tensor, img_name
        except Exception as e:
            print(f"警告：处理文件 {self.image_files[idx]} 时出错。将跳过此文件。错误: {e}")
            return None, None


# ================================================================
# Collate Function
# ================================================================
def robust_collate_fn(batch):
    valid_batch = [b for b in batch if b[0] is not None]
    if not valid_batch:
        return torch.tensor([]), []
    images = torch.stack([b[0] for b in valid_batch])
    filenames = [b[1] for b in valid_batch]
    return images, filenames


# ================================================================
# 主函数
# ================================================================
def main():
    parser = argparse.ArgumentParser(description="ViT 花卉分类预测脚本（最终版）")
    parser.add_argument("test_data_dir", type=str, help="测试集图片文件夹的路径")
    parser.add_argument("output_csv_path", type=str, help="保存 submission.csv 的路径")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用设备: {device}")

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        script_dir = os.getcwd()

    model_dir = os.path.join(script_dir, '..', 'model')
    config_path = os.path.join(model_dir, 'config.json')
    checkpoint_path = os.path.join(model_dir, 'best_model.pth')

    # --- 加载正确映射表---
    mapping_json_path = os.path.join(model_dir, 'class_to_idx.json')
    try:
        with open(mapping_json_path, 'r') as f:
            class_to_idx = json.load(f)
        idx_to_class = {v: int(k) for k, v in class_to_idx.items()}
        num_classes = len(idx_to_class)
        print(f"成功加载 {num_classes} 个类别映射。")
    except FileNotFoundError:
        print(f"在 {mapping_json_path} 未找到 class_to_idx.json。")
        return
    except Exception as e:
        print(f"加载 class_to_idx.json 时出错: {e}")
        return

    # --- 加载 ViT 的 config.json ---
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
    except FileNotFoundError:
        print(f"在路径 '{config_path}' 未找到 config.json 文件。")
        return

    model_type = config_data.get("model_type", "ViT-B_16")
    img_size = config_data.get("img_size", 384)
    print(f"模型类型: {model_type}, 图像大小: {img_size}")

    # --- 数据预处理 ---
    transform_test = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # --- DataLoader ---
    test_dataset = TestDataset(data_dir=args.test_data_dir, transform=transform_test)
    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=32,
        num_workers=0,
        pin_memory=True,
        collate_fn=robust_collate_fn
    )
    print(f"测试集已加载: 找到 {len(test_dataset)} 张有效图片。")

    # --- 模型加载 ---
    try:
        config = CONFIGS[model_type]
        model = VisionTransformer(config, img_size, zero_head=True, num_classes=num_classes)

        print(f"正在从 {checkpoint_path} 加载模型权重...")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
            print("模型权重（从 'model' 键）加载成功。")
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("模型权重（从 'model_state_dict' 键）加载成功。")
        else:
            model.load_state_dict(checkpoint)
            print("警告: 未找到 'model' 键，已尝试直接加载。")

        model.to(device)
        model.eval()
        print("模型已成功加载")

    except Exception as e:
        print(f"加载模型错误: {e}")
        return

    # --- 运行推理 ---
    predictions = []
    try:
        with torch.no_grad():
            for images, filenames in tqdm(test_loader, desc="正在预测"):
                if not filenames:
                    continue

                images = images.to(device)
                logits = model(images)[0]

                probabilities = torch.softmax(logits, dim=1)
                confidence, pred_indices = torch.max(probabilities, dim=1)

                # --- 使用正确映射表 ---
                for filename, pred_idx, conf in zip(filenames, pred_indices, confidence):
                    real_category_id = idx_to_class[pred_idx.item()]
                    predictions.append([filename, real_category_id, conf.item()])

    except Exception as e:
        print(f"推理发生错误: {e}")
        return

    # --- 保存结果 ---
    try:
        os.makedirs(os.path.dirname(args.output_csv_path), exist_ok=True)
        headers = ['filename', 'category_id', 'confidence']

        with open(args.output_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(predictions)

        print(f"\n预测完成！提交文件已保存至: {args.output_csv_path}")
        print(f"总共预测了 {len(predictions)} 张图片。")

    except Exception as e:
        print(f"保存结果到 CSV 时出错: {e}")
        return


if __name__ == "__main__":
    main()