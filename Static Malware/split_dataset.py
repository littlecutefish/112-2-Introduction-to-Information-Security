import os
import random
import shutil

# 定義數據集根目錄
dataset_root = "/Users/liuliyu/Desktop/malware/all"

# 定義分割後的目錄
train_dir = "/Users/liuliyu/Desktop/malware/train"
val_dir = "/Users/liuliyu/Desktop/malware/val"
test_dir = "/Users/liuliyu/Desktop/malware/test"

# 定義分割比例
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# 遍歷數據集根目錄中的每個類別文件夾
for class_name in os.listdir(dataset_root):
    class_dir = os.path.join(dataset_root, class_name)
    # 如果是文件夾
    if os.path.isdir(class_dir):
        # 獲取當前類別文件夾下的所有圖像文件
        image_files = [f for f in os.listdir(class_dir) if f.endswith('.png')]
        # 隨機打亂圖像文件列表
        random.shuffle(image_files)
        # 計算訓練集、驗證集和測試集的數量
        num_images = len(image_files)
        num_train = int(num_images * train_ratio)
        num_val = int(num_images * val_ratio)
        num_test = num_images - num_train - num_val

        # 創建保存訓練集、驗證集和測試集的目錄
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)

        # 將圖像文件分配給訓練集、驗證集和測試集
        for i, image_file in enumerate(image_files):
            src_path = os.path.join(class_dir, image_file)
            if i < num_train:
                dst_path = os.path.join(train_class_dir, image_file)
            elif i < num_train + num_val:
                dst_path = os.path.join(val_class_dir, image_file)
            else:
                dst_path = os.path.join(test_class_dir, image_file)
            # 將圖像文件復制到相應的目錄
            shutil.copy(src_path, dst_path)

print("Dataset split completed successfully.")
