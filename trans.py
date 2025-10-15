import os
import shutil
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image  # 用于检查图片完整性

# ---------------------- 1. 配置路径（请修改为你的实际路径） ----------------------
csv_path = "train_labels.csv"  # 你的CSV文件路径
image_folder = "./train/train"  # 原始图片所在文件夹路径
root_folder = "dataset"  # 最终生成train/val的根文件夹

# 处理PIL的图片截断警告
Image.MAX_IMAGE_PIXELS = None  # 取消最大像素限制
os.environ["PIL_IMAGE_MAX_MEMORY"] = "1073741824"  # 1GB内存限制


def is_image_valid(img_path):
    """检查图片文件是否完整有效"""
    try:
        with Image.open(img_path) as img:
            img.verify()  # 验证文件完整性
            return True
    except (IOError, SyntaxError, OSError) as e:
        print(f"无效图片: {img_path}, 错误: {str(e)}")
        return False


# ---------------------- 2. 读取CSV并整理图片-类别对应关系 ----------------------
df = pd.read_csv(csv_path, header=None, names=["image_id", "class_id"])
df = df.drop_duplicates(subset=["image_id"])
class_to_images = df.groupby("class_id")["image_id"].apply(list).to_dict()

# ---------------------- 3. 创建文件夹结构 ----------------------
os.makedirs(root_folder, exist_ok=True)
for class_id in class_to_images.keys():
    os.makedirs(os.path.join(root_folder, "train", str(class_id)), exist_ok=True)
    os.makedirs(os.path.join(root_folder, "val", str(class_id)), exist_ok=True)

# ---------------------- 4. 按8:2拆分并移动图片 ----------------------
random.seed(42)  # 固定随机种子

for class_id, image_ids in class_to_images.items():
    # 过滤不存在或损坏的图片
    valid_images = []
    for img in image_ids:
        img_path = os.path.join(image_folder, img)
        if os.path.exists(img_path) and is_image_valid(img_path):
            valid_images.append(img)

    if not valid_images:
        print(f"警告：类别{class_id}下无有效图片，跳过该类别")
        continue

    # 拆分训练集和验证集
    train_imgs, val_imgs = train_test_split(
        valid_images, test_size=0.2, shuffle=True
    )

    # 移动训练集图片
    train_dst = os.path.join(root_folder, "train", str(class_id))
    for img in train_imgs:
        src = os.path.join(image_folder, img)
        dst = os.path.join(train_dst, img)
        shutil.copy(src, dst)

    # 移动验证集图片
    val_dst = os.path.join(root_folder, "val", str(class_id))
    for img in val_imgs:
        src = os.path.join(image_folder, img)
        dst = os.path.join(val_dst, img)
        shutil.copy(src, dst)

print("图片分类与拆分完成！")
