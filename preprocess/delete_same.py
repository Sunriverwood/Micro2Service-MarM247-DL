import os
import numpy as np
import pandas as pd
from PIL import Image
import imagehash
from skimage.metrics import structural_similarity as ssim
import torch
import concurrent.futures

#51行修改哈希值和汉明距离改变判定两张图片是否相同的严格程度，哈希越大，相似度越小，判定越松

# 确保使用 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calculate_hashes(image_tensor):
    """计算图片的多种哈希值"""
    img = image_tensor.cpu().numpy()  # 将张量转换为NumPy数组
    ahash = imagehash.average_hash(Image.fromarray(img))
    phash = imagehash.phash(Image.fromarray(img))
    dhash = imagehash.dhash(Image.fromarray(img))
    whash = imagehash.whash(Image.fromarray(img))
    return (ahash, phash, dhash, whash)

def load_and_process_image(image_path):
    """加载和处理图像并返回张量"""
    with Image.open(image_path) as img:
        img = img.convert('L').resize((256, 256))
        img_tensor = torch.tensor(np.array(img)).float().to(device)  # 转换为张量并移动到GPU
    return img_tensor

def calculate_ssim(image1_tensor, image2_tensor):
    """计算两张图片的SSIM相似度"""
    img1_np = image1_tensor.cpu().numpy()  # 转换为NumPy数组
    img2_np = image2_tensor.cpu().numpy()
    return ssim(img1_np, img2_np, data_range=255)  # 添加data_range参数

def process_image(file_path):
    """处理单张图片并计算哈希值"""
    try:
        img_tensor = load_and_process_image(file_path)
        hashes = calculate_hashes(img_tensor)
        return file_path, hashes
    except Exception as e:
        print(f"无法处理文件 {file_path}: {e}")
        return file_path, None

def hamming_distance(hash1, hash2):
    """计算两个哈希值之间的汉明距离"""
    return sum(c1 != c2 for c1, c2 in zip(str(hash1), str(hash2)))

def find_and_remove_duplicates(folder_path, hash_threshold=10, ssim_threshold=0.95):
    """遍历文件夹，找出所有相同但命名不同的图片，并删除较小的那一张"""
    image_hashes = {}
    duplicate_groups = []

    # 使用多线程处理图片
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_image, os.path.join(root, file)): (root, file)
                   for root, _, files in os.walk(folder_path)
                   for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.tif','.webp'))}

        for future in concurrent.futures.as_completed(futures):
            file_path, result = future.result()
            if result is None:
                continue
            hashes = result

            found_duplicate = False
            for existing_hashes, paths in image_hashes.items():
                distances = [hamming_distance(h1, h2) for h1, h2 in zip(hashes, existing_hashes)]
                if all(dist <= hash_threshold for dist in distances):
                    # 如果哈希值匹配，再通过SSIM确认
                    img_tensor = load_and_process_image(file_path)
                    if any(calculate_ssim(img_tensor, load_and_process_image(existing_file)) >= ssim_threshold for existing_file in paths):
                        paths.append(file_path)
                        found_duplicate = True
                        break

            if not found_duplicate:
                image_hashes[hashes] = [file_path]

    # 找出命名不同但内容相似的图片，并处理
    for duplicate_group in image_hashes.values():
        if len(duplicate_group) > 1:
            if len(duplicate_group) == 2:
                # 对于重复的两张图片，按文件大小从大到小排序，删除较小的
                duplicate_group.sort(key=lambda x: os.path.getsize(x), reverse=True)
                for file_to_delete in duplicate_group[1:]:
                    try:
                        os.remove(file_to_delete)
                        print(f"删除文件: {file_to_delete} (文件较小，已删除)")
                    except Exception as e:
                        print(f"无法删除文件 {file_to_delete}: {e}")
            else:
                # 对于重复的多于两张的图片，记录文件路径
                duplicate_groups.append(duplicate_group)

    # 将多于两张相同图片的路径写入Excel文件
    if duplicate_groups:
        output_file = os.path.join(folder_path, "duplicate_images.xlsx")
        all_duplicates = [path for group in duplicate_groups for path in group]
        df = pd.DataFrame(all_duplicates, columns=["Duplicate Image Paths"])
        df.to_excel(output_file, index=False)
        print(f"重复的多张图片路径已写入: {output_file}")

def main():
    folder_path = input("请输入要查找图片的文件夹路径: ")
    find_and_remove_duplicates(folder_path)
    print("处理完成！所有重复图片已处理。")

if __name__ == "__main__":
    main()
