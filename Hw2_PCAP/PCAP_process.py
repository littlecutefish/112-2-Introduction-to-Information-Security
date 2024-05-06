import os
import random
import hashlib
from scapy.all import *


def process_pcap(input_pcap_file, output_folder):
    # 讀取原始 PCAP 檔案
    packets = rdpcap(input_pcap_file)

    # 創建一個字典用於存儲每個 Flow 的資料包列表
    flows = {}

    # 遍歷資料包並進行 IP 位址隨機化處理
    for packet in packets:
        # 檢查資料包是否包含 IP 層
        if IP in packet:
            # 隨機生成新的來源 IP 和目標 IP
            new_src_ip = ".".join(str(random.randint(0, 255)) for _ in range(4))
            new_dst_ip = ".".join(str(random.randint(0, 255)) for _ in range(4))

            # 將原始資料包中的來源 IP 和目標 IP 替換為隨機生成的 IP
            packet[IP].src = new_src_ip
            packet[IP].dst = new_dst_ip

            # 獲取 Flow 的關鍵訊息
            flow_key = (
                packet[IP].src,
                packet[TCP].sport if TCP in packet else (packet[UDP].sport if UDP in packet else None),
                packet[IP].dst,
                packet[TCP].dport if TCP in packet else (packet[UDP].dport if UDP in packet else None),
                'TCP' if TCP in packet else ('UDP' if UDP in packet else None)
            )

            # 將資料包添加到相應 Flow 的列表中
            if flow_key[1] is not None and flow_key[3] is not None and flow_key[4] is not None:
                if flow_key not in flows:
                    flows[flow_key] = []
                flows[flow_key].append(packet)

    # 刪除空文件
    empty_files = 0
    for flow_key, flow_packets in flows.items():
        if len(flow_packets) == 0:
            empty_files += 1
            continue

    # 刪除重複文件
    duplicate_files = 0
    for flow_key, flow_packets in flows.items():
        if len(flow_packets) > 1:
            # 計算資料包的hash
            packet_hashes = [hashlib.md5(packet.__repr__()).digest() for packet in flow_packets]
            # 如果有重複的hash，則刪除多餘的文件
            unique_hashes = set(packet_hashes)
            if len(unique_hashes) < len(packet_hashes):
                # 刪除重複的資料包
                flows[flow_key] = [flow_packets[i] for i, h in enumerate(packet_hashes) if h in unique_hashes]
                duplicate_files += len(packet_hashes) - len(unique_hashes)

    # 遍歷每個 Flow，並將其資料包寫入單獨的 PCAP 文件
    for index, (flow_key, flow_packets) in enumerate(flows.items()):
        # 構造文件名
        filename = os.path.join(output_folder, f"flow_{index}.pcap")
        # 寫入 PCAP 文件
        wrpcap(filename, flow_packets)
        print(f"Flow {index} written to {filename}")

    print(f"Deleted {empty_files} empty files.")
    print(f"Deleted {duplicate_files} duplicate files.")


# 遍歷指定目錄下的所有資料夾
malware_folder = r"C:\Users\User\Desktop\Malware\adclicer"
for folder_name in os.listdir(malware_folder):
    folder_path = os.path.join(malware_folder, folder_name)

    # 檢查是否為資料夾
    if os.path.isdir(folder_path):
        # 構建原始 PCAP 檔案路徑
        input_pcap_file = os.path.join(folder_path, "dump.pcap")

        # 檢查原始 PCAP 檔案是否存在
        if os.path.exists(input_pcap_file):
            # 構建輸出目錄
            output_folder = os.path.join(folder_path, "output_pcap_files_sanitized")

            # 創建目標資料夾（如果不存在）
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            # 進行處理
            print(f"Processing {input_pcap_file}...")
            process_pcap(input_pcap_file, output_folder)
            print("Processing completed.")
        else:
            print(f"No dump.pcap file found in {folder_path}. Skipping.")
