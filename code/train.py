# coding=utf-8
import logging
import argparse
import os
import random
import numpy as np
from datetime import timedelta
import csv

import torch
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score

from model import VisionTransformer, CONFIGS
from utils import WarmupLinearSchedule, WarmupCosineSchedule
from utils import get_loader

logger = logging.getLogger(__name__)

"""
训练参数示例（花卉数据集）：
--name oxford-flower-vit 
--dataset flower 
--model_type ViT-B_16 
--pretrained_dir ../checkpoint/ViT-B_16.npz
--data_path ../dataset  # 花卉数据集根目录
--img_size 384 
--train_batch_size 16 
--eval_batch_size 64 
--learning_rate 3e-2 
--num_epochs 50 
--warmup_epochs 5 
--eval_every_epoch 1
--loss_save_path ./loss_logs.csv  # 训练/验证loss保存路径
"""


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calculate_metrics(preds, labels, num_classes=100):
    """Top1/Top5准确率 + F1分数"""
    # 1. Top1准确率
    top1_preds = torch.argmax(preds, dim=-1)
    top1_acc = (top1_preds == labels).float().mean().item()

    # 2. Top5准确率
    _, top5_indices = torch.topk(preds, k=5, dim=-1)
    labels_expanded = labels.unsqueeze(dim=-1).expand_as(top5_indices)
    top5_acc = (top5_indices == labels_expanded).any(dim=-1).float().mean().item()

    # 3. 宏平均F1分数
    top1_preds_np = top1_preds.cpu().numpy()
    labels_np = labels.cpu().numpy()
    f1 = f1_score(
        labels_np,
        top1_preds_np,
        average="macro",
        labels=range(num_classes),
        zero_division=0
    )

    return top1_acc, top5_acc, f1


def save_model(args, model, best_val_metrics, current_epoch):
    """保存模型"""
    model_to_save = model.module if hasattr(model, 'module') else model
    os.makedirs(args.output_dir, exist_ok=True)
    model_checkpoint = os.path.join(
        args.output_dir,
        f"best_model.pth"
    )
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info(f"Saved best model (epoch {current_epoch}) to: {model_checkpoint}")


def save_loss_to_csv(args, loss_data):
    """将训练/验证loss按epoch保存到CSV文件"""
    os.makedirs(os.path.dirname(args.loss_save_path), exist_ok=True)
    file_exists = os.path.isfile(args.loss_save_path)
    with open(args.loss_save_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
        if not file_exists:
            writer.writeheader()
        writer.writerow(loss_data)


def setup(args):
    """初始化 ViT 模型（花卉数据集固定 100 类）"""
    config = CONFIGS[args.model_type]
    num_classes = 100

    # 1. 初始化模型
    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes)

    # 2. 加载权重
    if args.load_checkpoint:
        logger.info(f"正在从 .pth 检查点加载模型: {args.load_checkpoint}")
        try:
            state_dict = torch.load(args.load_checkpoint, map_location=args.device)


            if 'transformer.embeddings.position_embeddings' in state_dict:
                pos_embed_loaded = state_dict['transformer.embeddings.position_embeddings']
                pos_embed_new = model.transformer.embeddings.position_embeddings

                if pos_embed_loaded.size() != pos_embed_new.size():
                    logger.info(f"正在插值位置编码: {pos_embed_loaded.size()} -> {pos_embed_new.size()}")

                    # (这段逻辑改编自 modeling.py)
                    ntok_new = pos_embed_new.size(1)
                    posemb_tok, posemb_grid = pos_embed_loaded[:, :1], pos_embed_loaded[0, 1:]
                    ntok_new -= 1

                    gs_old = int(np.sqrt(posemb_grid.size(0)))
                    gs_new = int(np.sqrt(ntok_new))
                    logger.info(f"网格尺寸: {gs_old} -> {gs_new}")

                    # (B, N, C) -> (B, C, N) -> (B, C, H, W)
                    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
                    # 使用双线性插值
                    posemb_grid = F.interpolate(posemb_grid, size=(gs_new, gs_new), mode='bilinear',
                                                align_corners=False)
                    # (B, C, H_new, W_new) -> (B, H_new, W_new, C) -> (B, N_new, C)
                    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)

                    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
                    state_dict['transformer.embeddings.position_embeddings'] = posemb

            # strict=False 允许我们只加载匹配的键
            model.load_state_dict(state_dict, strict=False)
            logger.info("模型加载成功。")

        except Exception as e:
            logger.error(f"加载 .bin 检查点失败: {e}")
            logger.error("将使用 --pretrained_dir 的 .npz 文件从头开始。")
            model.load_from(np.load(args.pretrained_dir))  # 加载失败时的后备方案

    else:
        # --- 原始逻辑：加载 .npz 文件 ---
        logger.info(f"正在从 .npz 加载预训练权重: {args.pretrained_dir}")
        model.load_from(np.load(args.pretrained_dir))

    model.to(args.device)

    # 统计可训练参数数量
    num_params = count_parameters(model)
    logger.info("Model Config: {}".format(config))
    logger.info("Training Parameters: %s", args)
    logger.info("Total Trainable Parameters: \t%2.1fM" % num_params)
    print(f"Total Trainable Parameters: {num_params:.1f}M")
    return args, model

def count_parameters(model):
    """统计可训练参数数量（单位：M）"""
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1000000


def set_seed(args):
    """固定随机种子以确保可复现性"""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def validate(args, model, writer, val_loader, current_epoch, num_classes=100):
    """验证函数（按epoch执行，返回val损失和指标）"""
    eval_losses = AverageMeter()
    val_top1 = AverageMeter()
    val_top5 = AverageMeter()
    val_f1 = AverageMeter()

    logger.info(f"\n***** Running Validation (Epoch {current_epoch}) *****")
    logger.info("  Num Batches = %d", len(val_loader))
    logger.info("  Batch Size = %d", args.eval_batch_size)

    model.eval()
    all_preds_logits = []
    all_labels = []
    epoch_iterator = tqdm(
        val_loader,
        desc=f"Validating (Epoch {current_epoch})... (loss=X.X, top1=X.X)",
        bar_format="{l_bar}{r_bar}",
        dynamic_ncols=True,
        disable=args.local_rank not in [-1, 0]
    )

    loss_fct = torch.nn.CrossEntropyLoss()

    for step, batch in enumerate(epoch_iterator):
        x, y = tuple(t.to(args.device) for t in batch)
        batch_size = x.size(0)

        with torch.no_grad():
            logits = model(x)[0]
            eval_loss = loss_fct(logits, y)
            top1_acc, top5_acc, f1 = calculate_metrics(logits, y, num_classes)

        # 更新统计器
        eval_losses.update(eval_loss.item(), batch_size)
        val_top1.update(top1_acc, batch_size)
        val_top5.update(top5_acc, batch_size)
        val_f1.update(f1, batch_size)

        # 收集全局结果
        all_preds_logits.append(logits.detach().cpu())
        all_labels.append(y.detach().cpu())

        # 更新进度条
        epoch_iterator.set_description(
            f"Validating (Epoch {current_epoch})... (loss={eval_losses.val:.5f}, top1={val_top1.val:.5f})"
        )

    # 计算全局指标
    all_logits = torch.cat(all_preds_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    overall_top1, overall_top5, overall_f1 = calculate_metrics(all_logits, all_labels, num_classes)

    # 打印验证结果
    logger.info("\nValidation (Epoch {}) Results".format(current_epoch))
    logger.info("Val Loss:       %2.5f" % eval_losses.avg)
    logger.info("Val Top1 Accuracy: %2.5f" % overall_top1)
    logger.info("Val Top5 Accuracy: %2.5f" % overall_top5)
    logger.info("Val Macro F1 Score: %2.5f" % overall_f1)

    # 写入TensorBoard（按epoch记录）
    if args.local_rank in [-1, 0]:
        writer.add_scalar("val/loss", eval_losses.avg, current_epoch)
        writer.add_scalar("val/top1_accuracy", overall_top1, current_epoch)
        writer.add_scalar("val/top5_accuracy", overall_top5, current_epoch)
        writer.add_scalar("val/macro_f1", overall_f1, current_epoch)

    # 返回val损失和指标字典
    return eval_losses.avg, {"top1": overall_top1, "top5": overall_top5, "f1": overall_f1}


def train(args, model):
    """核心训练函数（按epoch训练，保存loss并优化日志）"""
    # 初始化日志和TensorBoard
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join("logs", args.name))
        logger.info(f"TensorBoard日志路径: logs/{args.name}")

    # 加载数据集
    train_loader, val_loader = get_loader(args)
    logger.info(f"\n数据集加载完成:")
    logger.info(f"  训练集: {len(train_loader.dataset)} 样本, {len(train_loader)} 批次/epoch")
    logger.info(f"  验证集: {len(val_loader.dataset)} 样本, {len(val_loader)} 批次")

    # 计算学习率调度器参数
    total_train_batches = len(train_loader)
    total_steps = total_train_batches // args.gradient_accumulation_steps * args.num_epochs
    warmup_steps = total_train_batches // args.gradient_accumulation_steps * args.warmup_epochs
    logger.info(f"学习率调度: 总步数={total_steps}, 热身步数={warmup_steps}")

# --- 差分学习率修改 ---

    # 1. 识别出“头”和“身体”的参数
    # "头" = BDC模块 + 最终的线性分类层
    head_parameters = list(model.bdc_module.parameters()) + list(model.head.parameters())

    # "身体" = 原始ViT的Transformer主干
    backbone_parameters = model.transformer.parameters()

    # 2. 创建参数组
    # 为 "头" 设置高学习率 (使用命令行传入的 --learning_rate)
    # 为 "身体" 设置一个低10倍的学习率
    head_lr = args.learning_rate
    backbone_lr = args.learning_rate * 0.1  # 关键：为Backbone设置一个低得多的LR

    optimizer_grouped_parameters = [
        {'params': head_parameters, 'lr': head_lr},
        {'params': backbone_parameters, 'lr': backbone_lr}
    ]

    logger.info("差分学习率已启用:")
    logger.info(f"  Head (BDC + Linear) LR: {head_lr}")
    logger.info(f"  Backbone (Transformer) LR: {backbone_lr}")

    # 3. 初始化优化器
    optimizer = torch.optim.SGD(
        optimizer_grouped_parameters,  # 使用新的参数组
        momentum=0.9,
        weight_decay=args.weight_decay
    )

    # --- 修改结束 ---

    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=warmup_steps, t_total=total_steps)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=total_steps)

    # 分布式训练配置
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank
        )
        logger.info(f"分布式训练启用: Local Rank = {args.local_rank}")

    # 训练状态初始化
    best_val_metrics = {"top1": 0.0, "top5": 0.0, "f1": 0.0}
    global_step = 0
    loss_log = []  # 缓存loss数据，支持中途保存

    # 训练主循环（按epoch迭代）
    logger.info("\n===== 开始训练 =====")
    for current_epoch in range(1, args.num_epochs + 1):
        # 分布式训练：每个epoch打乱数据
        if args.local_rank != -1:
            train_loader.sampler.set_epoch(current_epoch)

        model.train()
        train_loss_meter = AverageMeter()  # 本epoch训练损失
        epoch_iterator = tqdm(
            train_loader,
            desc=f"Epoch {current_epoch}/{args.num_epochs} | 训练损失: 0.00000",
            bar_format="{l_bar}{r_bar}",
            dynamic_ncols=True,
            disable=args.local_rank not in [-1, 0]
        )

        # 单epoch内的batch迭代
        for step, batch in enumerate(epoch_iterator):
            x, y = tuple(t.to(args.device) for t in batch)
            batch_size = x.size(0)

            # 前向传播计算损失
            loss = model(x, y)

            # 梯度累积处理
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            # 反向传播
            loss.backward()

            # 累积到指定步数后更新参数
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # 更新损失统计
                train_loss_meter.update(loss.item() * args.gradient_accumulation_steps, batch_size)

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                # 参数更新与学习率调度
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # 更新进度条
                epoch_iterator.set_description(
                    f"Epoch {current_epoch}/{args.num_epochs} | 训练损失: {train_loss_meter.val:.5f}"
                )

                # 写入TensorBoard（每10步记录一次）
                if args.local_rank in [-1, 0] and global_step % 10 == 0:
                    writer.add_scalar("train/step_loss", train_loss_meter.val, global_step)
                    writer.add_scalar("train/learning_rate", scheduler.get_lr()[0], global_step)

        # 本epoch训练结束
        logger.info(f"\nEpoch {current_epoch} 训练完成 | 平均训练损失: {train_loss_meter.avg:.5f}")
        if args.local_rank in [-1, 0]:
            writer.add_scalar("train/epoch_loss", train_loss_meter.avg, current_epoch)

        # 验证逻辑（每N个epoch执行）
        if current_epoch % args.eval_every_epoch == 0 and args.local_rank in [-1, 0]:
            val_loss, current_val_metrics = validate(
                args, model, writer, val_loader, current_epoch
            )

            # 缓存并保存loss数据
            loss_data = {
                "epoch": current_epoch,
                "train_loss": round(train_loss_meter.avg, 5),
                "val_loss": round(val_loss, 5)
            }
            loss_log.append(loss_data)
            save_loss_to_csv(args, loss_data)
            logger.info(f"Epoch {current_epoch} 损失已保存至: {args.loss_save_path}")

            # 更新最佳模型
            if current_val_metrics["top1"] > best_val_metrics["top1"]:
                best_val_metrics = current_val_metrics
                save_model(args, model, best_val_metrics, current_epoch)
                logger.info(
                    f"更新最佳模型 (Epoch {current_epoch}): "
                    f"Top1={best_val_metrics['top1']:.4f}, "
                    f"Top5={best_val_metrics['top5']:.4f}, "
                    f"F1={best_val_metrics['f1']:.4f}"
                )

    # 训练完全结束
    if args.local_rank in [-1, 0]:
        writer.close()
        logger.info("\n===== 训练全部完成 =====")
        logger.info(f"最终最佳验证指标:")
        logger.info(f"  Top1准确率: {best_val_metrics['top1']:.4f}")
        logger.info(f"  Top5准确率: {best_val_metrics['top5']:.4f}")
        logger.info(f"  宏平均F1: {best_val_metrics['f1']:.4f}")
        logger.info(f"最佳模型保存路径: {args.output_dir}")
        logger.info(f"损失日志保存路径: {args.loss_save_path}")


def main():
    """入口函数：解析参数、初始化环境、启动训练"""
    parser = argparse.ArgumentParser(description="ViT花卉分类训练（按epoch）")

    # 必选参数
    parser.add_argument("--name", required=True, help="训练任务名称")
    parser.add_argument("--data_path", required=True, help="数据集根目录（含train/val）")
    parser.add_argument("--pretrained_dir", default="checkpoint/ViT-B_16.npz", help="预训练权重路径")
    # --- 添加下面这一行 ---
    parser.add_argument("--load_checkpoint", type=str, default=None, help="要加载的 .pth 模型检查点路径 (用于微调)")
    # --- 添加结束 ---
    # 模型与数据集参数
    parser.add_argument("--dataset", choices=["flower"], default="flower", help="数据集类型")
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16", "ViT-L_32", "R50-ViT-B_16"],
                        default="ViT-B_16", help="ViT模型变体")
    parser.add_argument("--img_size", default=384, type=int, help="输入图像尺寸")

    # 训练参数（按epoch调整）
    parser.add_argument("--num_epochs", required=True, type=int, help="总训练epoch数",default=5)
    parser.add_argument("--warmup_epochs", default=5, type=int, help="学习率热身epoch数")
    parser.add_argument("--train_batch_size", default=16, type=int, help="训练批次大小")
    parser.add_argument("--eval_batch_size", default=64, type=int, help="验证批次大小")
    parser.add_argument("--eval_every_epoch", default=1, type=int, help="每N个epoch验证一次")
    parser.add_argument("--num_workers", default=4, type=int, help="数据加载进程数")

    # 优化器参数
    parser.add_argument("--learning_rate", default=3e-2, type=float, help="初始学习率")
    parser.add_argument("--weight_decay", default=1e-4, type=float, help="权重衰减系数")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine", help="学习率衰减方式")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="梯度裁剪最大范数")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="梯度累积步数")

    # 输出参数
    parser.add_argument("--output_dir", default="model", help="模型保存目录")
    parser.add_argument("--loss_save_path", default="./loss_log.csv", help="损失日志保存路径")

    # 分布式与稳定性参数
    parser.add_argument("--local_rank", type=int, default=-1, help="分布式训练本地Rank（-1为单卡）")
    parser.add_argument("--seed", default=42, type=int, help="随机种子")

    args = parser.parse_args()

    # 设备初始化
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
        logger.info(f"单卡训练: 设备={device}, GPU数量={args.n_gpu}")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", timeout=timedelta(minutes=60))
        args.n_gpu = 1
        logger.info(f"分布式训练: Local Rank={args.local_rank}, 设备={device}")
    args.device = device

    # 日志配置
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN
    )
    logger.info(f"初始化完成: Rank={args.local_rank}, 设备={args.device}")

    # 固定随机种子
    set_seed(args)
    logger.info(f"随机种子已固定: {args.seed}")

    # 初始化模型并启动训练
    args, model = setup(args)
    train(args, model)


if __name__ == "__main__":
    main()
