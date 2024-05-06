from scapy.all import *
from PIL import Image
import numpy as np
import os

# 指定資料夾路徑
malware_folder = r"C:\Users\User\Desktop\Malware\adclicer"
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
        pcap_folder = os.path.join(folder_path, "output_pcap_files_sanitized")
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

            # 將所有數據包的Payload拼接起來
            payload = b""
            for packet in packets:
                if Raw in packet:
                    payload += bytes(packet[Raw].load)

            # 裁剪或填充payload以確保長度為784（28*28）
            if len(payload) > 784:
                payload = payload[:784]
            elif len(payload) < 784:
                payload += b"\x00" * (784 - len(payload))

            # 將payload轉換為numpy數組
            hex_array = np.frombuffer(payload, dtype=np.uint8)

            # 將數據重新形狀為28x28的矩陣
            hex_array = hex_array.reshape(28, 28)

            # 創建灰度圖像
            img = Image.fromarray(hex_array, mode='L')

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
