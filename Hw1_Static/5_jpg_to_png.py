from PIL import Image
import os

def jpg_to_png(input_folder, output_folder):
    # 確保輸出資料夾存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍歷輸入資料夾中的所有檔案
    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg'):
            # 打開jpg檔案
            img = Image.open(os.path.join(input_folder, filename))
            # 構造輸出檔案路徑
            output_path = os.path.join(output_folder, filename[:-4] + '.png')
            # 轉換並保存為png檔案
            img.save(output_path, 'PNG')

if __name__ == "__main__":
    input_folder = r'C:\Users\jolie\Desktop\MOTIF\images_new'  # 輸入資料夾路徑
    output_folder = r'C:\Users\jolie\Desktop\MOTIF\images'  # 輸出資料夾路徑
    jpg_to_png(input_folder, output_folder)
