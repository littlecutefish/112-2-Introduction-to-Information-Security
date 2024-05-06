import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 读取 Excel 文件
excel_file = "/Users/liuliyu/Desktop/malware/dynamic/calls_data.xlsx_0.ods"
df = pd.read_excel(excel_file)

# 取得 D1 和 D 的最後一個值
D1 = df.loc[0, 'time']
D_last = df.loc[len(df) - 1, 'time']

# 计算时间范围
time_range = D_last - D1

# 计算每个时间段的时间间隔
time_interval = time_range / 16

# 创建一个新的列来表示每个时间点所在的段
df['time_segment'] = ((df['time'] - D1) // time_interval).astype(int) + 1

# 整理 category 名称成统一的分类
category_mapping = {
    'network': 'networking',
    'process': 'process and thread',
    'system': 'system',
    'password dumping': 'password dumping and password hash',
    'password hash': 'password dumping and password hash',
    'anti-debugging': 'anti-debugging and anti-reversing',
    'anti-reversing': 'anti-debugging and anti-reversing'
    # 可以根据需要添加其他的映射
}

# 设置 y 轴的标签
categories = [
    'networking', 'register', 'service', 'file', 'hardware and system',
    'message', 'process and thread', 'system', 'Shellcode', 'Keylogging',
    'Obfuscation', 'password dumping and password hash',
    'anti-debugging and anti-reversing', 'handle manipulation', 'high risk',
    'other'
]

# 将 category 数据标准化
def map_category(category):
    if category not in categories[:-1]:  # 如果 category 不在前面的 15 个类别中，则归类为 'other'
        return 'other'
    else:
        return category

df['category'] = df['category'].apply(lambda x: category_mapping.get(x, x))  # 将没有映射的保持原样
df['category'] = df['category'].apply(map_category)

# 使用 groupby 计算每个 category 在各个时间段内出现的次数
category_counts = df.groupby(['time_segment', 'category']).size().unstack(fill_value=0)

# 确保有 16 个时间段和 16 个 category
category_counts = category_counts.reindex(index=range(1, 17), columns=categories, fill_value=0)

# 转置 category_counts
category_counts = category_counts.T

category_counts.insert(0, 'category', categories)

# 将 DataFrame 转换为 numpy 数组
array_data = category_counts.values

# print(array_data)

# ====分隔線====
# 創建一個空的二維矩陣
color_array_data = np.empty((16, 16), dtype=object)

# 讀取Excel文件
df = pd.read_excel("/Users/liuliyu/Desktop/malware/dynamic/color_mapping.xlsx", index_col=0)

for i in range(len(array_data)):
    specific_word = ""
    given_value = 0

    # 存储 specific_word
    specific_word = array_data[i][0]
    # color_array_data[i][0] = specific_word

    for j in range(1, len(array_data[i])):
        # 指定要查找的數值
        given_value = array_data[i][j]

        # 遍历第一行，找到包含给定数值的范围所在的列索引
        column_index = None
        for col in df.columns[0:]:
            range_values = col.strip('()').strip(']').split(',')
            if len(range_values) == 2:
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
                color_array_data[i][j-1] = value_in_specified_column
            else:
                print(f"'{specific_word}' not found in the index.")
        else:
            print(f"No valid range found for the given value {given_value}.")

print(color_array_data)

# ==== 分隔線 ====
fig = plt.figure(figsize=(16, 16))
ax = fig.add_axes([0, 0, 1, 1])

# 計算單元格的大小
cell_size = 1/16

# 遍歷網格
for i in range(16):
    for j in range(16):
        # 如果是顏色值，則填充顏色
        if isinstance(color_array_data[i][j], str) and color_array_data[i][j].startswith("#"):
            p = patches.Rectangle(
                (j * cell_size, 1 - (i + 1) * cell_size),  # (left, bottom)
                cell_size, cell_size,  # width, height
                fill=True, color=color_array_data[i][j], clip_on=False, transform=ax.transAxes
            )
            ax.add_patch(p)
        else:
            # 否則填入文字
            ax.text(
                j * cell_size + 0.5 * cell_size,  # x-coordinate
                1 - (i * cell_size + 0.5 * cell_size),  # y-coordinate
                str(color_array_data[i][j]),  # 文字內容
                ha='center', va='center',  # 對齊方式
                fontsize=10, color='black',  # 文字大小和顏色
                transform=ax.transAxes
            )

# 隱藏座標軸
ax.set_axis_off()

# 將圖片保存為jpg檔案
plt.savefig('grid_image.jpg', format='jpg', dpi=300)

# 顯示圖片
plt.show()
