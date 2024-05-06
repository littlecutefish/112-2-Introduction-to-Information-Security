import os
import shutil

# 檢查目錄是否存在，如果不存在則創建它
target_dir = os.path.expanduser("~/文件/Malware/1clickdownload")
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# 源目錄
source_dir = os.path.expanduser("~/下載/all/1clickdownload")

# 檢查源目錄是否存在
if not os.path.exists(source_dir):
    print("源目錄不存在。")
    exit()

# 初始化索引
index = 1

# 遍歷源目錄中的JSON檔案
for filename in os.listdir(source_dir):
    if filename.endswith(".json"):
        # 提取檔案名稱（不包含副檔名）
        folder_name = str(index) + "." + os.path.splitext(filename)[0]

        # 新建目標資料夾路徑
        target_folder_path = os.path.join(target_dir, folder_name)

        # 創建目標資料夾
        os.makedirs(target_folder_path, exist_ok=True)

        # 構建源檔案的完整路徑
        source_file_path = os.path.join(source_dir, filename)

        # 更新索引
        index += 1

print("完成資料夾的建立和JSON檔案的複製。")
