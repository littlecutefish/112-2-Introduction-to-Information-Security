import os
import shutil

# 指定原始資料夾和目標資料夾
source_folder = r"C:\Users\User\Desktop\Malware\Hw2_PCAP\datasets\airinstaller\output_images"
target_folder = r"C:\Users\User\Desktop\Malware\Hw2_PCAP\datasets\all\airinstaller"

# 遍歷原始資料夾中的每個資料夾
for folder_name in os.listdir(source_folder):
    folder_path = os.path.join(source_folder, folder_name)

    # 檢查是否為資料夾
    if os.path.isdir(folder_path):
        # 遍歷資料夾中的每個檔案
        for filename in os.listdir(folder_path):
            source_file_path = os.path.join(folder_path, filename)
            target_file_path = os.path.join(target_folder, filename)

            # 移動檔案到目標資料夾
            shutil.move(source_file_path, target_file_path)
            print(f"Moved {filename} to {target_folder}")

print("All files moved successfully.")
