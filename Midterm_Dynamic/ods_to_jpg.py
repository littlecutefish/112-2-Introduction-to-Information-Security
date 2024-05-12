import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_grid_image(excel_file_path):
    # 讀取 Excel 文件
    df = pd.read_excel(excel_file_path)

    print(excel_file_path)
    # 取得 D1 和 D 的最後一個值
    D1 = df.loc[0, 'time']
    D_last = df.loc[len(df) - 1, 'time']

    # 計算時間範圍
    time_range = D_last - D1

    # 計算每個時間段的時間間隔
    time_interval = time_range / 16

    # 創建一個新的列來表示每個時間點所在的段
    df['time_segment'] = ((df['time'] - D1) // time_interval).astype(int) + 1

    # 整理 category 名稱成統一的分類
    category_mapping = {
        'network': 'networking',
        'process': 'process and thread',
        'system': 'system',
        'password dumping': 'password dumping and password hash',
        'password hash': 'password dumping and password hash',
        'anti-debugging': 'anti-debugging and anti-reversing',
        'anti-reversing': 'anti-debugging and anti-reversing'
        # 可以根據需要添加其他的映射
    }

    # 設定 y 軸的標籤
    categories = [
        'networking', 'register', 'service', 'file', 'hardware and system',
        'message', 'process and thread', 'system', 'Shellcode', 'Keylogging',
        'Obfuscation', 'password dumping and password hash',
        'anti-debugging and anti-reversing', 'handle manipulation', 'high risk',
        'other'
    ]

    # 將 category 數據標準化
    def map_category(category):
        if category not in categories[:-1]:  # 如果 category 不在前面的 15 個類別中，則歸類為 'other'
            return 'other'
        else:
            return category

    df['category'] = df['category'].apply(lambda x: category_mapping.get(x, x))  # 將沒有映射的保持原樣
    df['category'] = df['category'].apply(map_category)

    # 使用 groupby 計算每個 category 在各個時間段內出現的次數
    category_counts = df.groupby(['time_segment', 'category']).size().unstack(fill_value=0)

    # 確保有 16 個時間段和 16 個 category
    category_counts = category_counts.reindex(index=range(1, 17), columns=categories, fill_value=0)

    # 轉置 category_counts
    category_counts = category_counts.T

    category_counts.insert(0, 'category', categories)

    # 將 DataFrame 轉換為 numpy 數組
    array_data = category_counts.values

    # 創建一個空的二維矩陣
    color_array_data = np.empty((16, 16), dtype=object)

    # 讀取顏色映射數據
    color_mapping_df = pd.read_excel(r"C:\Users\User\Desktop\Malware\Midterm_Dynamic\color_mapping.xlsx", index_col=0)

    for i in range(len(array_data)):
        specific_word = ""
        given_value = 0

        # 存儲 specific_word
        specific_word = array_data[i][0]

        for j in range(1, len(array_data[i])):
            # 指定要查找的數值
            given_value = array_data[i][j]

            # 遍歷第一行，找到包含給定數值的範圍所在的列索引
            column_index = None
            for col in color_mapping_df.columns[0:]:
                range_values = col.strip('()').strip(']').split(',')
                if len(range_values) == 2:
                    lower_bound = float(range_values[0])
                    upper_bound = float(range_values[1])
                    if lower_bound < given_value <= upper_bound:
                        column_index = color_mapping_df.columns.get_loc(col)
                        break
                else:
                    print(f"Invalid range format for column {col}")

            if column_index is not None:
                if specific_word in color_mapping_df.index:
                    value_in_specified_column = color_mapping_df.iloc[color_mapping_df.index.get_loc(specific_word), column_index]
                    color_array_data[i][j-1] = value_in_specified_column
                else:
                    print(f"'{specific_word}' not found in the index.")
            else:
                print(f"No valid range found for the given value {given_value}.")

    # 創建圖像
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

    # 隱藏坐標軸
    ax.set_axis_off()

    # 將圖片保存到指定的資料夾內
    output_file_name = os.path.splitext(os.path.basename(excel_file_path))[0] + ".jpg"
    output_path = os.path.join(output_folder, output_file_name)
    plt.savefig(output_path, format='jpg', dpi=300)


# 資料夾路徑
folder_path = r"C:\Users\User\Desktop\Malware\Midterm_Dynamic\reports_v2\reports\acda"
output_folder = r"C:\Users\User\Desktop\Malware\Midterm_Dynamic\datasets\acda"

# 獲取資料夾內所有Excel文件
for file in os.listdir(folder_path):
    if file.endswith(".xlsx"):
        generate_grid_image(os.path.join(folder_path, file))
