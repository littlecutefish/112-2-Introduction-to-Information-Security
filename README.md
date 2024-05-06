# 112-2-Introduction-to-Information-Security
hw for 112-2 Introduction to Information Security

### Hw1: Static Malware
摸索中，分成比較多檔案，步步執行
- 使用resnet訓練，使用5種病毒檔案類別
缺點: 檔案數過少，訓練資料少，準確率只有60~70多...
![image](https://github.com/littlecutefish/112-2-Introduction-to-Information-Security/assets/90677074/6ee8d081-bc37-47f9-ae07-5f34c0c02c7f)

---
### Midterm: Dynamic Malware
剛好遇到電腦裝不起Cuckoo的問題，花了比較久的時間，因此只有把一個檔案的report拿來生成結果的圖片檔
TODO: 使用cuckoo腳本把report檔案抓下來後再train

---
### Hw2: Network Packet Analysis
有成功使用腳本上傳病毒檔並下載pcap檔案
使用resnet來訓練圖片檔，目前只使用4種資料來訓練
缺點: 準確率只有30多，還未思考如何優化訓練模型
![image](https://github.com/littlecutefish/112-2-Introduction-to-Information-Security/assets/90677074/64d44dd8-7d87-4327-b5d9-d889c295f346)

