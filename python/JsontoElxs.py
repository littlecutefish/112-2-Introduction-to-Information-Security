import json
import pandas as pd

def extract_calls_data(data):
    calls_data = []
    if isinstance(data, list):
        for item in data:
            calls_data.extend(extract_calls_data(item))
    elif isinstance(data, dict):
        if 'category' in data:  # 检查是否存在'category'字段
            call_info = {
                'status': data.get('status'),
                'category': data.get('category'),  # 提取'category'字段
                'api': data.get('api'),
                'time': data.get('time')
            }
            calls_data.append(call_info)
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                calls_data.extend(extract_calls_data(value))
    return calls_data

# JSON文件路径
json_file_path = '/home/jolie/Downloads/report.json'

# Load JSON data
with open(json_file_path, 'r') as file:
    data = json.load(file)

# Extract calls data
calls_data = extract_calls_data(data)

# Convert to DataFrame
calls_df = pd.DataFrame(calls_data)

# Save to Excel
calls_df.to_excel('calls_data.xlsx', index=False)
