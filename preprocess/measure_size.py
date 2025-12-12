import os
from PIL import Image


def find_images_with_unexpected_height(folder_path):
    # 用于统计不符合高度要求的图片
    unexpected_height_images = []

    # 遍历文件夹及其子文件夹
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 检查文件是否是图片
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.gif')):
                image_path = os.path.join(root, file)

                try:
                    # 打开图片
                    with Image.open(image_path) as img:
                        width, height = img.size

                        # 检查图片高度是否符合要求
                        if height not in [1094, 1024]:
                            unexpected_height_images.append((image_path, width, height))

                except Exception as e:
                    print(f"无法打开图片: {image_path}. 错误信息: {e}")

    return unexpected_height_images


# 指定文件夹路径
folder_path = r"C:\Users\SUN\Desktop\科研\东方电气\图片预处理2"  # 替换成你的文件夹路径
result = find_images_with_unexpected_height(folder_path)

if result:
    print("以下图片的高度不为 1094 像素和 1024 像素:")
    for image_path, width, height in result:
        print(f"路径: {image_path}, 宽度: {width}, 高度: {height}")
else:
    print("所有图片的高度都符合要求。")
