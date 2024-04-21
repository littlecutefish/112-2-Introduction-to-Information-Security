import os
import json
import shutil

def organize_images_from_jsonl(jsonl_file, image_folder, output_folder, count):
    # 打開JSONL文件
    with open(jsonl_file, 'r') as file:
        for line in file:
            item = json.loads(line)
            # 獲取檔案名（reported_hash）
            reported_hash = item['reported_hash']
            # 構建輸入圖片檔案路徑
            image_filename = f"{reported_hash}.png"
            image_path = os.path.join(image_folder, image_filename)
            # 獲取reported_family
            reported_family = item['reported_family']
            # 構建目標家族資料夾路徑
            family_folder = os.path.join(output_folder, reported_family)
            # 如果目標資料夾不存在，則創建它
            if not os.path.exists(family_folder):
                os.makedirs(family_folder)
            # 檢查圖片是否存在，並將圖片複製到目標家族資料夾中
            if os.path.exists(image_path):
                shutil.copy(image_path, family_folder)
                # print(f"已將 {image_filename} 複製到 {family_folder}")
                count += 1
            else:
                print(f"找不到 {image_filename}, family name:{reported_family}")

# 設置路徑和檔案名稱
jsonl_file = r'C:\Users\jolie\Desktop\MOTIF\dataset\motif_dataset.jsonl'  # 請替換為你的JSONL檔案路徑
image_folder = r'C:\Users\jolie\Desktop\MOTIF\images'  # 請替換為您存放圖像檔案的資料夾路徑
output_folder = r'C:\Users\jolie\Desktop\MOTIF\seg_train'  # 請替換為您想要的輸出資料夾路徑


count = 0
# 整理圖像檔案
organize_images_from_jsonl(jsonl_file, image_folder, output_folder, count)

print(f"total: {count}")
