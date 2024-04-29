from scapy.all import *
import random
import os
import hashlib

# 指定原始PCAP文件和目标文件夹
input_pcap_file = r"C:\Users\User\Desktop\Malware\1clickdownload\2.VirusShare_2fcdac1ecf51ba3690b68a241c1aa740\dump_sorted.pcap"
output_folder = r"C:\Users\User\Desktop\Malware\1clickdownload\2.VirusShare_2fcdac1ecf51ba3690b68a241c1aa740\output_pcap_files_sanitized"

# 创建目标文件夹（如果不存在）
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 读取原始PCAP文件
packets = rdpcap(input_pcap_file)

# 创建一个字典用于存储每个Flow的数据包列表
flows = {}

# 遍历数据包并进行IP地址随机化处理
for packet in packets:
    # 检查数据包是否包含IP层
    if IP in packet:
        # 随机生成新的源IP和目标IP
        new_src_ip = ".".join(str(random.randint(0, 255)) for _ in range(4))
        new_dst_ip = ".".join(str(random.randint(0, 255)) for _ in range(4))

        # 将原始数据包中的源IP和目标IP替换为随机生成的IP
        packet[IP].src = new_src_ip
        packet[IP].dst = new_dst_ip

        # 获取Flow的关键信息
        flow_key = (
            packet[IP].src,
            packet[TCP].sport if TCP in packet else (packet[UDP].sport if UDP in packet else None),
            packet[IP].dst,
            packet[TCP].dport if TCP in packet else (packet[UDP].dport if UDP in packet else None),
            'TCP' if TCP in packet else ('UDP' if UDP in packet else None)
        )

        # 将数据包添加到相应Flow的列表中
        if flow_key[1] is not None and flow_key[3] is not None and flow_key[4] is not None:
            if flow_key not in flows:
                flows[flow_key] = []
            flows[flow_key].append(packet)

# 删除空文件
empty_files = 0
for flow_key, flow_packets in flows.items():
    if len(flow_packets) == 0:
        empty_files += 1
        continue

# 删除重复文件
duplicate_files = 0
for flow_key, flow_packets in flows.items():
    if len(flow_packets) > 1:
        # 计算数据包的哈希值
        packet_hashes = [hashlib.md5(packet.__repr__()).digest() for packet in flow_packets]
        # 如果有重复的哈希值，则删除多余的文件
        unique_hashes = set(packet_hashes)
        if len(unique_hashes) < len(packet_hashes):
            # 删除重复的数据包
            flows[flow_key] = [flow_packets[i] for i, h in enumerate(packet_hashes) if h in unique_hashes]
            duplicate_files += len(packet_hashes) - len(unique_hashes)

# 遍历每个Flow，并将其数据包写入单独的PCAP文件
for index, (flow_key, flow_packets) in enumerate(flows.items()):
    # 构造文件名
    filename = os.path.join(output_folder, f"flow_{index}.pcap")
    # 写入PCAP文件
    wrpcap(filename, flow_packets)
    print(f"Flow {index} written to {filename}")

print(f"Deleted {empty_files} empty files.")
print(f"Deleted {duplicate_files} duplicate files.")