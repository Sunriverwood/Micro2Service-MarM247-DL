import os
import pandas as pd

def count_images_in_folders(folder_path):
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif','.tif', '.bmp', '.tiff'}
    image_counts = {}

    for root, dirs, files in os.walk(folder_path):
        if root != folder_path:  # 忽略顶层文件夹
            count = sum(1 for file in files if os.path.splitext(file)[1].lower() in image_extensions)
            if count > 0:  # 仅统计含有图片的文件夹
                image_counts[os.path.basename(root)] = count

    return image_counts

folder_path = r'C:\Users\SUN\Desktop\科研\东方电气\图片预处理3'
image_counts = count_images_in_folders(folder_path)

# 输出为Excel
df = pd.DataFrame(list(image_counts.items()), columns=['文件夹', '图片数量'])
df.to_excel('image_counts.xlsx', index=False)

print('统计结果已输出到 image_counts.xlsx')
