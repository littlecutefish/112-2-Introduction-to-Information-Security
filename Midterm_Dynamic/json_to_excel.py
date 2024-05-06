import json
import pandas as pd

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
            calls_data.append(call_info)
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                calls_data.extend(extract_calls_data(value))
    return calls_data

# JSON檔案路徑
json_file_path = r"C:\Users\User\Desktop\Malware\Midterm_Dynamic\report.json"

# 載入JSON資料
with open(json_file_path, 'r') as file:
    data = json.load(file)

# 提取呼叫資料
calls_data = extract_calls_data(data)

# 轉換為 DataFrame
calls_df = pd.DataFrame(calls_data)

# 儲存為 Excel 檔案
calls_df.to_excel('calls_data.xlsx', index=False)
