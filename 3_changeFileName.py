import os
import json
import shutil

# 讀取.jsonl檔案，建立一個mb5到reported_hash的映射表
def create_mapping(jsonl_file):
    mapping = {}
    with open(jsonl_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            mb5 = data.get('md5')
            reported_hash = data.get('reported_hash')
            if mb5 and reported_hash:
                mapping[mb5] = reported_hash
    return mapping

# 將每張圖像檔案的名字改為對應的reported_hash值
def rename_images(image_folder, new_image_folder, mapping):
    # 確保新資料夾存在，如果不存在則創建它
    if not os.path.exists(new_image_folder):
        os.makedirs(new_image_folder)

    for filename in os.listdir(image_folder):
        mb5 = os.path.splitext(filename)[0]  # 從檔案名稱中提取 mb5
        reported_hash = mapping.get(mb5)
        if reported_hash:
            # 建立舊檔案和新檔案的路徑
            old_path = os.path.join(image_folder, filename)
            new_filename = reported_hash + os.path.splitext(filename)[1]  # 保留原始的副檔名
            new_path = os.path.join(new_image_folder, new_filename)
            # 移動圖像檔案到新的資料夾
            shutil.move(old_path, new_path)
            print(f"已將 {filename} 移動到 {new_path}")
        else:
            print(f"找不到 {filename} 對應的 reported_hash 值")

# 設置路徑和檔案名稱
jsonl_file = r'C:\Users\jolie\Desktop\MOTIF\dataset\motif_dataset_copy.jsonl'
image_folder = r'C:\Users\jolie\Desktop\MOTIF\images'
new_image_folder = r'C:\Users\jolie\Desktop\MOTIF\images_new'

# 建立mb5到reported_hash的映射表
mapping = create_mapping(jsonl_file)

# 將圖像檔案重新命名並移動到新的資料夾
rename_images(image_folder, new_image_folder, mapping)
