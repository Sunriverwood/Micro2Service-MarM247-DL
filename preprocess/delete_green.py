import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_image(image_path):
    """使用Torch加载图片并转化为Tensor"""
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")  # 确保图片是RGB模式
            transform = transforms.ToTensor()  # 转化为Tensor
            image_tensor = transform(img).to(device)
            return image_tensor
    except Exception as e:
        print(f"无法处理图片 {image_path}: {e}")
        return None


def detect_green_text(image_tensor, green_threshold=0.05):
    """
    检测图片中是否有绿色标记
    green_threshold: 判断是否删除图片的绿色像素比例阈值
    """
    # 转换为 NumPy 数组
    image_np = image_tensor.permute(1, 2, 0).cpu().numpy()

    # 将RGB转换为HSV
    img_hsv = np.zeros_like(image_np)
    img_hsv[..., 0] = np.arctan2(np.sqrt(3) * (image_np[..., 1] - image_np[..., 2]),
                                 2 * image_np[..., 0] - image_np[..., 1] - image_np[..., 2])
    img_hsv[..., 1] = np.max(image_np, axis=2) - np.min(image_np, axis=2)
    img_hsv[..., 2] = np.mean(image_np, axis=2)

    # 识别绿色部分（Hue 约为 0.3-0.5, 即 [35, 85] 度范围）
    green_mask = (img_hsv[..., 0] >= 0.3) & (img_hsv[..., 0] <= 0.5)

    # 判断绿色部分占图片的比例
    green_ratio = np.sum(green_mask) / (image_np.shape[0] * image_np.shape[1])

    return green_ratio > green_threshold


def find_and_remove_green_text_images(folder_path, green_threshold=0.00001):
    """遍历文件夹并删除包含绿色文字标记的图片"""
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.tif','.webp')):
                file_path = os.path.join(root, file)
                image_tensor = load_image(file_path)

                if image_tensor is None:
                    continue

                try:
                    if detect_green_text(image_tensor, green_threshold):
                        os.remove(file_path)
                        print(f"删除包含绿色文字标记的图片: {file_path}")
                except Exception as e:
                    print(f"无法处理文件 {file_path}: {e}")


def main():
    folder_path = input("请输入要查找图片的文件夹路径: ")
    find_and_remove_green_text_images(folder_path)
    print("处理完成！所有包含绿色文字标记的图片已处理。")


if __name__ == "__main__":
    main()
