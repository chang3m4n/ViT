import torchvision.datasets as datasets
import json
import os

# --- !!! 修改这里 !!! ---
# 1. 确保这个路径指向你本地的 "train" 文件夹
#    (根据你的 trans.py，它应该在 "dataset/train")
TRAIN_DATA_PATH = r"D:\competition\NCCCU\ViT\ViT\dataset\train"

# 2. 确保这个路径指向你最终要提交的 submission/model 文件夹
OUTPUT_JSON_PATH = r"D:\competition\NCCCU\ViT\ViT\submission\model\class_to_idx.json"
# --- !!! 修改结束 !!! ---

print(f"正在从 {TRAIN_DATA_PATH} 读取训练数据...")
try:
    # 这一步会完全模拟你训练时 ImageFolder 的行为
    dataset = datasets.ImageFolder(root=TRAIN_DATA_PATH)

    # 提取那个按“字母顺序”生成的宝贵映射
    class_to_idx = dataset.class_to_idx

    if not class_to_idx or len(class_to_idx) != 100:
        print(f"错误：只找到了 {len(class_to_idx)} 个类别，请确保 TRAIN_DATA_PATH 正确。")
    else:
        print(f"成功提取了 {len(class_to_idx)} 个类别的映射！")

        # 确保输出目录存在
        os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)

        # 将映射保存为 json 文件
        with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(class_to_idx, f, indent=4)

        print(f"成功！已将正确的映射表保存到: {OUTPUT_JSON_PATH}")
        print("\n请检查该文件，然后用这个 submission 文件夹打包提交。")

except FileNotFoundError:
    print(f"错误：找不到路径 {TRAIN_DATA_PATH}")
    print("请确保上面的 TRAIN_DATA_PATH 变量指向你本地的 train 文件夹。")
except Exception as e:
    print(f"发生错误: {e}")