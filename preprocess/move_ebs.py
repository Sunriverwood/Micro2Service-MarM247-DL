import os
import shutil


def move_images_with_ebs(source_folder, target_folder):
    # 遍历源文件夹及其子文件夹
    for dirpath, _, filenames in os.walk(source_folder):
        for filename in filenames:
            # 检查文件名中是否包含 "ebs"
            if "ebs" in filename:
                # 构建源文件的完整路径
                source_file = os.path.join(dirpath, filename)

                # 计算目标文件夹的相对路径
                relative_path = os.path.relpath(dirpath, source_folder)
                target_subfolder = os.path.join(target_folder, relative_path)

                # 创建目标子文件夹（如果不存在）
                os.makedirs(target_subfolder, exist_ok=True)

                # 移动文件到目标子文件夹
                shutil.move(source_file, os.path.join(target_subfolder, filename))
                print(f"移动文件: {source_file} 到 {target_subfolder}")


# 使用示例
source_folder = r"C:\Users\SUN\Desktop\科研\东方电气\图片预处理\裁剪前"  # 替换为实际源文件夹路径
target_folder = r"C:\Users\SUN\Desktop\科研\东方电气\图片预处理\裁剪前\ebs"  # 替换为实际目标文件夹路径
move_images_with_ebs(source_folder, target_folder)
