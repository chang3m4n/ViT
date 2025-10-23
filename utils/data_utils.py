import logging
import os
from PIL import Image
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler, Dataset


logger = logging.getLogger(__name__)


def get_loader(args):
    # 分布式训练同步（主进程优先加载）
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    # 1. 数据预处理
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # 验证集用带标签的预处理（需要和训练集对应）
    transform_val = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    # 路径定义
    data_dirs = {
          "train": os.path.join(args.data_path, "train"),  # 有类别文件夹（带标签）
          "valid": os.path.join(args.data_path, "val"),      # 有类别文件夹（带标签）
     }
    # 验证路径存在
    for split, dir_path in data_dirs.items():
         if not os.path.exists(dir_path):
               raise FileNotFoundError(f"{split}文件夹不存在：{dir_path}")

    # 加载带标签的训练集和验证集（用ImageFolder，自动获取标签）
    trainset = datasets.ImageFolder(
        root=data_dirs["train"],
         transform=transform_train
    )
    valset = datasets.ImageFolder(
         root=data_dirs["valid"],
          transform=transform_val
      ) if args.local_rank in [-1, 0] else None


    logger.info(f"数据集加载完成："
              f"训练集 {len(trainset)} 样本，"
             f"验证集 {len(valset) if valset else 0} 样本，")

    # 分布式同步
    if args.local_rank == 0:
        torch.distributed.barrier()

    # 3. 定义采样器
    train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
    val_sampler = SequentialSampler(valset) if valset is not None else None

    # 4. 创建数据加载器
    train_loader = DataLoader(
        trainset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        valset,
        sampler=val_sampler,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    ) if valset is not None else None


    return train_loader, val_loader