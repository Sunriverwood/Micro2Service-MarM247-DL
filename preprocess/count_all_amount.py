import os

def count_images_in_directory(directory):
    # Define common image file extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff','.tif', '.webp', '.heic'}

    image_count = 0

    # Walk through all folders and subfolders
    for root, _, files in os.walk(directory):
        for file in files:
            # Check if the file has an image extension
            if os.path.splitext(file)[1].lower() in image_extensions:
                image_count += 1

    return image_count

# Specify the directory path you want to check
directory_path = r'C:\Users\SUN\Desktop\科研\东方电气\图片预处理'

print(f"Total number of images: {count_images_in_directory(directory_path)}")
