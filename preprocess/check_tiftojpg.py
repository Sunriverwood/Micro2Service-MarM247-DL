import os

def check_jpg_in_subfolders(parent_folder):
    # 用于存储结果
    results = {}

    for subdir, dirs, files in os.walk(parent_folder):
        # 如果该子文件夹不包含文件，则跳过
        if not files:
            continue

        jpg_found = False
        for file in files:
            if file.lower().endswith('.jpg'):
                jpg_found = True
                break

        results[subdir] = jpg_found

    # 打印每个子文件夹的检查结果
    for subdir, has_jpg in results.items():
        if has_jpg:
            print(f"子文件夹 '{subdir}' 中存在 JPG 图片。")
        else:
            print(f"子文件夹 '{subdir}' 中不存在 JPG 图片。")

    # 检查是否所有有图片的子文件夹都包含 JPG 图片
    no_jpg_folders = [subdir for subdir, has_jpg in results.items() if not has_jpg]
    if no_jpg_folders:
        print("\n以下子文件夹中没有 JPG 图片：")
        for folder in no_jpg_folders:
            print(f" - {folder}")
    else:
        print("\n所有有图片的子文件夹中都有 JPG 图片。")


# 替换为你的父级文件夹路径
parent_folder = r'C:\Users\SUN\Desktop\科研\东方电气\图片预处理4'
check_jpg_in_subfolders(parent_folder)
