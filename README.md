# 112-2-Introduction-to-Information-Security
hw for 112-2 Introduction to Information Security

### Hw1: Static Malware
摸索中，分成比較多檔案，步步執行

- 使用resnet訓練，使用4種病毒檔案類別

缺點: 檔案數過少，訓練資料少，準確率只有60~70多...

- Hw1_Static
  - ResNet: train, predict pictures
  - datasets
  - 0_check_folder.py
  - 1_get_file_name.py
  - 2_read_file.py: 主要讀取檔案後轉成灰階影像
  - 3_change_file_name.py
  - 4_move_img_to_folder.py
  - 5_jpg_to_png.py

![image](https://github.com/littlecutefish/112-2-Introduction-to-Information-Security/assets/90677074/6ee8d081-bc37-47f9-ae07-5f34c0c02c7f)

---
### Midterm: Dynamic Malware
剛好遇到電腦裝不起Cuckoo的問題，花了比較久的時間，因此只有把一個檔案的report拿來生成結果的圖片檔

TODO: 使用cuckoo腳本把report檔案抓下來後再train

- Midterm_Dynamic
  - datasets
  - tools: 當時在摸索顏色對應表格的小工具
  - color_mapping.xlsx: 顏色對應表格
  - json_to_excel.py: 把report的json檔案整理在excel上
  - ods_to_jpg.py: 把 excel/ods檔 轉成照片檔案

---
### Hw2: Network Packet Analysis
1. 有成功使用腳本上傳病毒檔並下載pcap檔案
2. 使用resnet來訓練圖片檔，目前只使用4種資料來訓練

缺點: 準確率只有30多，還未思考如何優化訓練模型

- Hw2_PCAP
  - ResNet: train, predict pictures
  - datasets
  - split_dataset
  - tools: 一些py的小工具
  - PCAP_process.py: 先處理PCAP檔案
  - PCAP_to_img.py: 把所有PCAP的小檔案轉成灰階圖
  - take_out_imgs.py: 移動檔案的小小工具(不重要)

![image](https://github.com/littlecutefish/112-2-Introduction-to-Information-Security/assets/90677074/64d44dd8-7d87-4327-b5d9-d889c295f346)
