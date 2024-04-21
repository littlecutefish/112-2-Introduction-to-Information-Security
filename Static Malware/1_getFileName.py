import os

# 資料夾路徑
folder_path = r"C:\Users\jolie\Desktop\MOTIF\MOTIF_defanged"

# 讀取資料夾內的檔案名稱
file_names = os.listdir(folder_path)

# 要儲存字串的 txt 檔路徑
output_file_path = "strings_v1.txt"

# 遍歷每個檔案，將其字串寫入 txt 檔
with open(output_file_path, "w") as f:
    for file_name in file_names:
        # 提取 "MOTIF_" 後面的字串
        motif_string = file_name.split("MOTIF_")[1]
        
        # 構建完整的檔案路徑
        file_path = os.path.join(folder_path, file_name)

        # 檢查檔案是否存在，若不存在則跳過
        if not os.path.exists(file_path):
            print("File does not exist:", file_path)
            continue
        
        # 寫入 txt 檔
        f.write(motif_string + "\n")
        # print("Processed file:", motif_string)

print("Strings have been saved to", output_file_path)
