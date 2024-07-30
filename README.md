# 112-2-Introduction-to-Information-Security
hw for 112-2 Introduction to Information Security

### Hw1: Static Malware
reference: https://medium.com/@vipinra79/image-classification-using-transfer-learning-with-inceptionv3-in-google-colab-d89c34ba438f

- 使用InceptionV3訓練，使用6種病毒檔案類別

結果: 準確率只有80多...

- Hw1_Static
  - 0_check_folder.py
  - 1_get_file_name.py
  - 2_read_file.py: 主要讀取檔案後轉成灰階影像
  - 3_change_file_name.py
  - 4_move_img_to_folder.py
  - 5_jpg_to_png.py
  - cnn_train.ipynb

![image](https://github.com/user-attachments/assets/dcb14754-7ef6-4a9c-acd6-e05c1978e371)


---
### Midterm: Dynamic Malware

TODO: training model 要再找好一點

缺點：datasets太少，所以無法訓練到準確率太高

- Midterm_Dynamic
  - datasets
  - tools: 當時在摸索顏色對應表格的小工具
  - color_mapping.xlsx: 顏色對應表格
  - json_to_excel.py: 把report的json檔案整理在excel上
  - ods_to_jpg.py: 把 excel/ods檔 轉成照片檔案

<img width="454" alt="截圖 2024-05-23 下午2 25 25" src="https://github.com/littlecutefish/112-2-Introduction-to-Information-Security/assets/90677074/9247c2f3-ce8b-48a1-9524-5d8d985faf2c">

---
### Hw2: Network Packet Analysis
1. 有成功使用腳本上傳病毒檔並下載pcap檔案
2. 使用resnet來訓練圖片檔，目前使用5種資料來訓練

- Hw2_PCAP
  - cnn: train, predict pictures
  - datasets
  - split_dataset
  - tools: 一些py的小工具
  - PCAP_process.py: 先處理PCAP檔案
  - PCAP_to_img.py: 把所有PCAP的小檔案轉成灰階圖
  - take_out_imgs.py: 移動檔案的小小工具(不重要)

準確率提升:30 -> 48
![image](https://github.com/littlecutefish/112-2-Introduction-to-Information-Security/assets/90677074/8443986a-7617-44f8-9348-039d2f5748ce)

---
### Hw3: Federated Learning

- 兩個檔案(一個有differential privacy, 一個沒有)

使用colab以及五個datasets訓練

Federated.ipynb:(accurancy:68)

![image](https://github.com/user-attachments/assets/4490e2a4-34d5-41b6-8b68-0751318da121)

加入noise: 0.1後:(accurancy:22-25左右)

![image](https://github.com/littlecutefish/112-2-Introduction-to-Information-Security/assets/90677074/f1b596e0-fc46-4914-bbae-45b87bbfe5d7)

---
### Final: Adversarial Attack

使用Hw1的資料集來處理圖片並擾亂預測

epsilon = 0.1, accuracy = 25

epsilon = 0.3, accuracy = 12.5

epsilon = 0.5, accuracy = 0
