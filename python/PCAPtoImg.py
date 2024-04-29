from scapy.all import *
from PIL import Image
import numpy as np
import os

# 指定PCAP文件夹和输出文件夹
pcap_folder = r"C:\Users\User\Desktop\Malware\1clickdownload\2.VirusShare_2fcdac1ecf51ba3690b68a241c1aa740\output_pcap_files_sanitized"
output_folder = r"C:\Users\User\Desktop\Malware\1clickdownload\2.VirusShare_2fcdac1ecf51ba3690b68a241c1aa740\output_images"

# 创建输出文件夹（如果不存在）
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历PCAP文件夹中的每个PCAP文件
for filename in os.listdir(pcap_folder):
    # 读取PCAP文件
    packets = rdpcap(os.path.join(pcap_folder, filename))

    # 将所有数据包的Payload拼接起来
    payload = b""
    for packet in packets:
        if Raw in packet:
            payload += bytes(packet[Raw].load)

    # 裁剪或填充payload以确保长度为784（28*28）
    if len(payload) > 784:
        payload = payload[:784]
    elif len(payload) < 784:
        payload += b"\x00" * (784 - len(payload))

    # 打印hex bytes的长度
    print(f"PCAP文件 {filename} 的Hex Bytes长度为: {len(payload)}")

    # 将payload转换为numpy数组
    hex_array = np.frombuffer(payload, dtype=np.uint8)

    # 将数据重新形状为28x28的矩阵
    hex_array = hex_array.reshape(28, 28)

    # 创建灰度图像
    img = Image.fromarray(hex_array, mode='L')

    # 保存图像
    img_filename = os.path.join(output_folder, os.path.splitext(filename)[0] + ".png")
    img.save(img_filename)

    print(f"PCAP文件 {filename} 的图像已保存为: {img_filename}")
