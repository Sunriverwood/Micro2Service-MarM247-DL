import os
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class ImageDataset(Dataset):
    """自定义数据集，用于加载测试集图片"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.labels = []
        self._prepare_dataset()

    def _prepare_dataset(self):
        # 遍历所有子文件夹
        for folder_name in os.listdir(self.root_dir):
            folder_path = os.path.join(self.root_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue
            # 解析文件夹名，提取时间
            temp_time = folder_name.split("_")
            try:
                time_exposed = float(temp_time[1].replace("h", ""))
            except ValueError:
                print(f"Skipping invalid folder name: {folder_name}")
                continue
            # 遍历文件夹中的图片
            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                if img_path.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):  # 检查是否为图片
                    self.samples.append(img_path)
                    self.labels.append([time_exposed])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]  # 真实值 (时间)
        img = Image.open(img_path)

        if self.transform:
            img = self.transform(img)

        return img, label, img_path


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 创建模型
    from Alex_model import AlexNet  # 确保 AlexNet 的输出为 1（仅时间）
    model = AlexNet(num_outputs=1).to(device)  # 修改为仅预测时间

    # 加载模型权重
    weights_path = "AlexNet_regression.pth"
    assert os.path.exists(weights_path), "file: '{}' does not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, weights_only=True, map_location=device))
    model.eval()

    # 加载测试集
    test_root = '../../superalloy_data/heat_exposure_photos/test'
    assert os.path.exists(test_root), "file: '{}' does not exist.".format(test_root)

    dataset = ImageDataset(test_root, transform=data_transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 记录预测值和真实值
    all_preds = []
    all_labels = []
    error_images = []  # 记录预测偏差较大的图片路径
    total = 0

    # 预测
    with torch.no_grad():
        for img, label, img_path in dataloader:
            img = img.to(device)
            label = torch.tensor(label, dtype=torch.float32).to(device)

            # 预测
            output = model(img).cpu().numpy()[0]  # 输出预测值
            true_label = label.cpu().numpy()  # 真实值

            # 保存预测值和真实值
            all_preds.append(output)
            all_labels.append(true_label)

            # 如果偏差较大，记录图片路径
            error = np.abs(output - true_label)
            if error > 1000:  # 假设时间偏差 > 1000
                error_images.append(img_path[0])  # 记录图片路径

            total += 1

    # 转换为 NumPy 数组
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 评估指标（仅针对时间）
    mse_time = mean_squared_error(all_labels, all_preds)
    mae_time = mean_absolute_error(all_labels, all_preds)
    r2_time = r2_score(all_labels, all_preds)

    print(f'Total images: {total}')
    print(f'Time - MSE: {mse_time:.3f}, MAE: {mae_time:.3f}, R²: {r2_time:.3f}')

    # 可视化预测与真实值的对比（时间）
    plt.figure(figsize=(10, 5))
    plt.plot(all_labels[:, 0], label='True Time', color='green', linestyle='dashed')
    plt.plot(all_preds, label='Predicted Time', color='orange')
    plt.legend()
    plt.title('True vs Predicted Time')
    plt.xlabel('Sample Index')
    plt.ylabel('Time (h)')
    plt.savefig('time_comparison.png')
    plt.show()

    # # 输出预测偏差较大的图片路径
    # if error_images:
    #     print("Images with large prediction errors:")
    #     for img_path in error_images:
    #         print(img_path)
    # else:
    #     print("No significant prediction errors found.")


if __name__ == '__main__':
    main()
