import json
import pandas as pd
import os


def extract_calls_data(data):
    calls_data = []
    if isinstance(data, list):
        for item in data:
            calls_data.extend(extract_calls_data(item))
    elif isinstance(data, dict):
        if 'category' in data:  # 檢查是否存在'category'欄位
            call_info = {
                'status': data.get('status'),
                'category': data.get('category'),  # 提取'category'欄位
                'api': data.get('api'),
                'time': data.get('time')
            }
            # 如果任一欄位為空則跳過
            if all(call_info.values()):
                calls_data.append(call_info)
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                calls_data.extend(extract_calls_data(value))
    return calls_data


# JSON檔案目錄路徑
json_dir_path = r"C:\Users\User\Desktop\Malware\Midterm_Dynamic\reports\adclicer"

# 檢查目錄是否存在
if not os.path.exists(json_dir_path):
    print("指定的目錄不存在。")
    exit()

# 處理每個JSON檔案
for filename in os.listdir(json_dir_path):
    if filename.endswith('.json'):
        json_file_path = os.path.join(json_dir_path, filename)
        print("正在處理檔案:", json_file_path)

        # 載入JSON資料
        with open(json_file_path, 'r') as file:
            data = json.load(file)

        # 提取呼叫資料
        calls_data = extract_calls_data(data)

        # 轉換為 DataFrame
        calls_df = pd.DataFrame(calls_data)

        # 儲存為 Excel 檔案
        excel_filename = os.path.splitext(filename)[0] + '.xlsx'
        excel_file_path = os.path.join(json_dir_path, excel_filename)
        calls_df.to_excel(excel_file_path, index=False)
        print("已儲存為 Excel 檔案:", excel_file_path)
