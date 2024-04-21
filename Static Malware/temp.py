import os

def count_files(folder_path):
    total_files = 0
    # 遍歷資料夾中的所有子資料夾
    for root, dirs, files in os.walk(folder_path):
        # 將每個子資料夾中的檔案數量相加
        total_files += len(files)
    return total_files

folder_path = r'C:\Users\jolie\Desktop\MOTIF\seg_train'
total_files = count_files(folder_path)
print(f"Total number of files in {folder_path} and its subfolders: {total_files}")
