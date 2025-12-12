import os
import cupy as cp
from PIL import Image, UnidentifiedImageError

# 原始图片文件夹路径
input_folder = r'C:\Pycharm\Projects\DONGFANG\superalloy_data\heat_exposure_photos\tem_time-0C'
# 镜像图片文件夹路径
output_folder = r'C:\Pycharm\Projects\DONGFANG\superalloy_data\heat_exposure_photos\tem_time-0C0'

# 创建输出文件夹（如果不存在）
os.makedirs(output_folder, exist_ok=True)

# 裁剪的正方形大小
square_size = 512

# 存储无法处理的文件信息
error_files = []

# 遍历文件夹中的所有图片文件
for root, _, files in os.walk(input_folder):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):  # 支持的图片格式
            image_path = os.path.join(root, file)
            try:
                image = Image.open(image_path)
                image.verify()  # 验证图像是否可以读取
                image = Image.open(image_path)  # 重新打开图像以获取大小

                width, height = image.size

                # 将图片数据转换为GPU数组
                image_array = cp.array(image)

                # 分割图片
                for i in range(2):  # 行
                    top = i * square_size
                    bottom = top + square_size
                    if bottom > height:
                        top = height - square_size
                        bottom = height

                    for j in range(3):  # 列
                        left = j * square_size
                        right = left + square_size
                        if right > width:
                            left = width - square_size
                            right = width

                        # 裁剪图片
                        cropped_image_array = image_array[top:bottom, left:right]

                        # 转回PIL图像
                        cropped_image = Image.fromarray(cp.asnumpy(cropped_image_array))
                        if cropped_image.mode != 'RGB':
                            cropped_image = cropped_image.convert('RGB')

                        # 创建对应的输出文件夹路径
                        relative_path = os.path.relpath(root, input_folder)
                        output_subfolder = os.path.join(output_folder, relative_path)
                        os.makedirs(output_subfolder, exist_ok=True)

                        # 构建输出文件路径，确保为jpg格式
                        output_image_path = os.path.join(output_subfolder,
                                                         f"{os.path.splitext(file)[0]}_cropped_{i * 3 + j + 1}.jpg")
                        cropped_image.save(output_image_path, format='JPEG')

                        # 打印处理信息
                        print(f"已处理文件 {file}，生成分割图像 {output_image_path}")

            except (UnidentifiedImageError, OSError) as e:
                error_files.append(image_path)  # 收集无法处理的文件信息

# 输出无法处理的文件信息
if error_files:
    print("以下文件无法处理：")
    for error_file in error_files:
        print(error_file)
else:
    print("所有文件均成功处理！")

print("图片分割完成！")
