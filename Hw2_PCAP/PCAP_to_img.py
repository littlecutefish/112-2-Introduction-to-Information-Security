from scapy.all import *
from PIL import Image
import numpy as np
import os

# 指定資料夾路徑 adclicer airinstaller
malware_folder = r"C:\Users\User\Desktop\Malware\Hw2_PCAP\datasets\airinstaller"
output_folder = os.path.join(malware_folder, "output_images")

# 創建輸出資料夾（如果不存在）
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍歷每個VirusShare資料夾
for folder_name in os.listdir(malware_folder):
    folder_path = os.path.join(malware_folder, folder_name)

    # 檢查是否為VirusShare資料夾
    if os.path.isdir(folder_path) and folder_name.startswith("VirusShare_"):
        # 構建PCAP檔案和輸出圖像資料夾路徑
        pcap_folder = os.path.join(folder_path, "output_pcap_files")
        output_subfolder = os.path.join(output_folder, folder_name)

        # 創建子資料夾（如果不存在）
        if not os.path.exists(output_subfolder):
            os.makedirs(output_subfolder)

        # 計數器
        image_counter = 1

        # 遍歷PCAP檔案夾中的每個PCAP檔案
        for filename in os.listdir(pcap_folder):
            # 讀取PCAP檔案
            packets = rdpcap(os.path.join(pcap_folder, filename))

            # 提取第一個封包的二進制數據
            first_packet_data = bytes(packets[0])

            # 將二進制數據轉換為 numpy 數組
            data = np.frombuffer(first_packet_data, dtype=np.uint8)

            # 如果數據長度不足，補 0
            width, height = 28, 28  # 指定圖像寬度和高度
            if len(data) < width * height:
                data = np.concatenate([data, np.zeros(width * height - len(data), dtype=np.uint8)])

            # 將數據調整為指定的圖像尺寸
            data = np.reshape(data[:width * height], (height, width))

            # 創建灰度圖像
            img = Image.fromarray(data, mode='L')

            # 獲取圖像中所有像素的灰度值
            pixels = list(img.getdata())

            # 檢查是否所有像素都是黑色（0）
            if all(pixel == 0 for pixel in pixels):
                # 跳過
                continue
            else:
                # 保存圖像
                img_filename = os.path.join(output_subfolder, f"{folder_name}_{image_counter}.png")
                img.save(img_filename)

                print(f"PCAP文件 {filename} 的圖像已保存為: {img_filename}")

            # 更新計數器
            image_counter += 1

print("所有圖像已保存完成。")
