import os
import csv
import argparse
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler
from torchvision import transforms
from sklearn.metrics import f1_score
from torchvision.datasets import ImageFolder

# 导入模型相关
from models.modeling import VisionTransformer, CONFIGS


def calculate_metrics(probs, labels, num_classes):
    """计算TOP1、TOP5准确率和F1分数"""
    # TOP1准确率
    top1_preds = np.argmax(probs, axis=1)
    top1_acc = np.mean(top1_preds == labels)

    # TOP5准确率
    top5_preds = np.argsort(probs, axis=1)[:, -5:]  # 获取每个样本的前5个预测
    top5_acc = np.mean([label in top5 for label, top5 in zip(labels, top5_preds)])

    # F1分数（宏平均）
    f1 = f1_score(labels, top1_preds, average='macro', labels=range(num_classes), zero_division=0)

    return {
        'top1_acc': top1_acc,
        'top5_acc': top5_acc,
        'f1_score': f1
    }


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16", "ViT-L_32", "R50-ViT-B_16"],
                        default="ViT-B_16", help="模型类型（需与训练时一致）")
    parser.add_argument("--checkpoint_path", required=True, help="训练好的模型权重路径")
    parser.add_argument("--valid_dir", required=True, help="验证集目录（需包含类别子文件夹）")
    parser.add_argument("--output_csv", default="valid_metrics.csv", help="评估结果保存路径")
    parser.add_argument("--img_size", default=224, type=int, help="图像尺寸（需与训练时一致）")
    parser.add_argument("--batch_size", default=32, type=int, help="评估批次大小")
    parser.add_argument("--num_classes", default=100, type=int, help="类别数（花卉数据集通常为102）")
    parser.add_argument("--device", default="cuda", help="使用的设备（cuda或cpu）")
    args = parser.parse_args()

    # 设备设置
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 数据预处理（与训练时的验证集处理一致）
    transform_valid = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载验证集（使用ImageFolder，自动获取标签）
    valid_dataset = ImageFolder(
        root=args.valid_dir,
        transform=transform_valid
    )
    valid_loader = DataLoader(
        valid_dataset,
        sampler=SequentialSampler(valid_dataset),
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True
    )
    print(f"验证集加载完成: {len(valid_dataset)} 张图片，{len(valid_dataset.classes)} 个类别")

    # 初始化模型
    config = CONFIGS[args.model_type]
    model = VisionTransformer(
        config,
        args.img_size,
        zero_head=True,
        num_classes=args.num_classes
    )

    # 加载模型权重
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()  # 切换到评估模式
    print(f"模型权重加载完成: {args.checkpoint_path}")

    # 开始评估
    all_probs = []
    all_labels = []

    with torch.no_grad():  # 禁用梯度计算，加速评估
        for batch in tqdm(valid_loader, desc="评估中..."):
            images, labels = batch
            images = images.to(device)

            # 模型前向传播
            logits = model(images)[0]  # 获取模型输出的logits
            probs = torch.softmax(logits, dim=-1)  # 转换为概率

            # 收集结果
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 计算指标
    metrics = calculate_metrics(
        np.array(all_probs),
        np.array(all_labels),
        args.num_classes
    )

    # 打印结果
    print("\n===== 验证集评估结果 =====")
    print(f"TOP1准确率: {metrics['top1_acc']:.4f} ({metrics['top1_acc'] * 100:.2f}%)")
    print(f"TOP5准确率: {metrics['top5_acc']:.4f} ({metrics['top5_acc'] * 100:.2f}%)")
    print(f"宏平均F1分数: {metrics['f1_score']:.4f}")

    # 保存结果到CSV
    with open(args.output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value', 'percentage'])
        writer.writerow(['TOP1准确率', metrics['top1_acc'], f"{metrics['top1_acc'] * 100:.2f}%"])
        writer.writerow(['TOP5准确率', metrics['top5_acc'], f"{metrics['top5_acc'] * 100:.2f}%"])
        writer.writerow(['宏平均F1分数', metrics['f1_score'], f"{metrics['f1_score'] * 100:.2f}%"])

    print(f"\n评估结果已保存到: {args.output_csv}")


if __name__ == "__main__":
    main()
