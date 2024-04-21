import os
import numpy as np
from PIL import Image

# 函數：根據檔案大小決定影像寬度
def get_width(file_size):
    width = 32
    
    if file_size > 1024000:
        width = 1024
    elif file_size > 512000:
        width = 768
    elif file_size > 204800:
        width = 512
    elif file_size > 102400:
        width = 384
    elif file_size > 61440:
        width = 256
    elif file_size > 30720:
        width = 128
        
    return width

def save_image(vector_data, width, file_name):
    width = int(width / 8)
    
    # 計算所需像素數量
    remainder = len(vector_data) % width
    if remainder != 0:
        padding = width - remainder
        vector_data.extend([0] * padding)  # 填充為0

    # 將向量轉換為像素矩陣
    pixel_matrix = np.array(vector_data).reshape(-1, width)
    
    # 創建影像對象
    img = Image.fromarray(np.uint8(pixel_matrix), mode='L')
    
    # 定義保存影像的路徑
    # 去掉檔名中的 VirusShare_
    file_name = file_name.replace('VirusShare_', '')
    save_path = os.path.join("C:\\Users\\jolie\\Desktop\\MOTIF\\virusShare_images", file_name + ".png")
    img.save(save_path)

# 讀取資料夾中的檔案名稱
file_names = os.listdir(r"C:\Users\jolie\Desktop\VirusShare\VirusShare_00273")

# 紀錄處理的檔案數量
count = 0

# 迭代處理每個檔案
for file_name in file_names:
    # 構建完整檔案路徑
    file_path = os.path.join(r"C:\Users\jolie\Desktop\VirusShare\VirusShare_00273", file_name)

    # 檢查檔案是否存在
    if not os.path.exists(file_path):
        print(f"檔案 {file_name} 不存在。跳過處理...")
        continue

    try:
        # 獲取檔案大小
        file_size = os.path.getsize(file_path)

        # 讀取檔案內容
        with open(file_path, "rb") as file:
            content = file.read()

        # 將位元組轉換為整數列表
        integer_list = [byte for byte in content]

        # 轉換整數列表為灰階影像並保存
        save_image(integer_list, get_width(file_size), file_name)

        count += 1

    except OSError as e:
        print(f" {file_path} 出錯")
        continue

print("所有影像已保存。總數:", count)
