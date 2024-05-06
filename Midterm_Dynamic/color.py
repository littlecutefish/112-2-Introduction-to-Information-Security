import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# 讀取Excel文件
df = pd.read_excel("/Users/liuliyu/Desktop/malware/dynamic/color_mapping.xlsx", index_col=0)

# 創建一個16x16的subplot
fig, ax = plt.subplots(figsize=(8, 8))

# 逐行進行迭代，並將色碼填充到網格中
for i in range(len(df)):
    for j in range(len(df.columns)):
        color = df.iloc[i, j]
        # print(i, j, color)
        ax.add_patch(plt.Rectangle((j, i), 1, 1, color=color))

# 指定要查找的數值
given_value = 100
# 找到特定單詞所在行的對應列的值
specific_word = "networking"

# 遍歷第一行，找到包含給定數值的範圍所在的列索引
column_index = None
for col in df.columns[1:].astype(str):  # 從第二列開始遍歷，跳過文字描述
    range_values = col.strip('()').strip(']').split(',')  # 刪除右括號後再進行分割
    if len(range_values) == 2:  # 檢查範圍值列表是否有兩個元素
        lower_bound = float(range_values[0])
        upper_bound = float(range_values[1])
        if lower_bound < given_value <= upper_bound:
            column_index = df.columns.get_loc(col)
            break
    else:
        print(f"Invalid range format for column {col}")

if column_index is not None:
    if specific_word in df.index:
        value_in_specified_column = df.iloc[df.index.get_loc(specific_word), column_index]
        print(f"The value of '{specific_word}' in the specified column is: {value_in_specified_column}")
    else:
        print(f"'{specific_word}' not found in the index.")
else:
    print(f"No valid range found for the given value {given_value}.")


# 設置刻度標籤
ax.set_xticks(range(len(df.columns)))
ax.set_xticklabels(df.columns)
ax.set_yticks(range(len(df)))
ax.set_yticklabels(df.index)

# 設置x軸和y軸的方向
ax.xaxis.tick_top()

# 顯示圖像
plt.show()
