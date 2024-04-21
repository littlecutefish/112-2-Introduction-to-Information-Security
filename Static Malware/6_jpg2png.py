from PIL import Image
import os

def jpg_to_png(input_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.jpg'):
            # 打开jpg文件
            img = Image.open(os.path.join(input_folder, filename))
            # 构造输出文件路径
            output_path = os.path.join(output_folder, filename[:-4] + '.png')
            # 转换并保存为png文件
            img.save(output_path, 'PNG')

if __name__ == "__main__":
    input_folder = r'C:\Users\jolie\Desktop\MOTIF\images_new'  # 输入文件夹路径
    output_folder = r'C:\Users\jolie\Desktop\MOTIF\images'  # 输出文件夹路径
    jpg_to_png(input_folder, output_folder)
