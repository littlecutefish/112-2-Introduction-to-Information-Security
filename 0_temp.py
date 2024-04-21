import os
import shutil

# 指定主資料夾路徑
main_folder_path = r"C:\Users\jolie\Desktop\MOTIF\seg_train"

# 新資料夾路徑
new_folder_path = r"C:\Users\jolie\Desktop\MOTIF\seg_train_new"

# 獲取主資料夾下的所有子資料夾
sub_folders = [f.path for f in os.scandir(main_folder_path) if f.is_dir()]

# 遍歷每個子資料夾，檢查檔案數量是否大於 10，如果是，則複製到新資料夾
for sub_folder in sub_folders:
    file_count = len([f for f in os.listdir(sub_folder) if os.path.isfile(os.path.join(sub_folder, f))])
    if file_count > 10:
        try:
            # 建立新資料夾（如果不存在）
            if not os.path.exists(new_folder_path):
                os.makedirs(new_folder_path)
            # 複製資料夾
            shutil.copytree(sub_folder, os.path.join(new_folder_path, os.path.basename(sub_folder)))
            print(f"資料夾 {sub_folder} 已複製到新資料夾")
        except OSError as e:
            print(f"複製資料夾 {sub_folder} 時發生錯誤：{e}")
