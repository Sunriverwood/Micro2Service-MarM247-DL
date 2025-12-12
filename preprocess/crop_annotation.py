import os
import torch
from PIL import Image

def crop_images(folder_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.tif')):
                file_path = os.path.join(root, file)

                try:
                    with Image.open(file_path) as img:
                        width, height = img.size
                        if height == 1094:
                            cropped_img = img.crop((0, 0, width, height - 70))
                            cropped_img.save(file_path)
                            print(f"Processed: {file_path}")

                except Exception as e:
                    print(f"Error processing image {file_path}: {e}")

# Example usage
folder_path = r"C:\Users\SUN\Desktop\科研\东方电气\图片预处理2"  # Replace with your folder path
crop_images(folder_path)
