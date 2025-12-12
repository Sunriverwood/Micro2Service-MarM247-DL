import os
import shutil
#先将tif移动到临时文件夹，再改一下将jpg移动回去

def move_tif_images(source_dir, dest_dir):
    # 遍历源文件夹及其子文件夹
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith('.jpg'):#'.tif'
                # 构建源文件的完整路径
                source_file = os.path.join(root, file)

                # 计算目标文件夹的相对路径并创建
                relative_path = os.path.relpath(root, source_dir)
                target_folder = os.path.join(dest_dir, relative_path)

                # 如果目标子文件夹不存在，则创建
                if not os.path.exists(target_folder):
                    os.makedirs(target_folder)

                # 剪切文件到目标文件夹
                dest_file = os.path.join(target_folder, file)
                shutil.move(source_file, dest_file)  #复制将move改为copy2
                print(f'已移动: {source_file} 到 {dest_file}')

# 使用示例
source_directory = r'C:\Users\SUN\Desktop\科研\东方电气\图片预处理4'  # 替换为你的源文件夹路径
destination_directory = r'C:\Users\SUN\Desktop\科研\东方电气\图片预处理2'  # 替换为你的目标文件夹路径
move_tif_images(source_directory, destination_directory)
