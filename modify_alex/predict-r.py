import os
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from model import AlexNet
import pandas as pd


class ImageDataset(Dataset):
    """自定义数据集，用于加载测试集图片，并从文件夹名称解析温度和时间"""

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
            # 解析文件夹名，提取温度和时间
            temp_time = folder_name.split("_")
            try:
                temperature = float(temp_time[0].replace("℃", ""))
                time_exposed = float(temp_time[1].replace("h", ""))
            except ValueError:
                print(f"Skipping invalid folder name: {folder_name}")
                continue
            # 遍历文件夹中的图片
            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                if img_path.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):  # 检查是否为图片
                    self.samples.append(img_path)
                    self.labels.append([temperature, time_exposed])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]  # 真实值 (温度, 时间)
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label, img_path

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 创建模型，确保 num_outputs=2 以分别预测温度和时间
    model = AlexNet(num_outputs=2).to(device)

    weights_path = "parameter/all-r.pth"
    assert os.path.exists(weights_path), "file: '{}' does not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, weights_only=True, map_location=device))
    model.eval()

    # 加载测试集
    test_root = '../superalloy_data/heat_exposure_photos/test'
    assert os.path.exists(test_root), f"Test data folder not found: '{test_root}'"

    dataset = ImageDataset(test_root, transform=data_transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 记录所有预测值和真实值
    all_preds = []
    all_labels = []

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

    # 转换为 NumPy 数组
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 分别提取温度和时间的数据
    true_temps = all_labels[:, 0]
    pred_temps = all_preds[:, 0]
    true_times = all_labels[:, 1]
    pred_times = all_preds[:, 1]

    # --- 评估温度预测 ---
    mse_temp = mean_squared_error(true_temps, pred_temps)
    mae_temp = mean_absolute_error(true_temps, pred_temps)
    r2_temp = r2_score(true_temps, pred_temps)
    print(f'--- Temperature Prediction Evaluation ---')
    print(f'MSE: {mse_temp:.3f}, MAE: {mae_temp:.3f}, R²: {r2_temp:.3f}\n')

    # --- 评估时间预测 ---
    mse_time = mean_squared_error(true_times, pred_times)
    mae_time = mean_absolute_error(true_times, pred_times)
    r2_time = r2_score(true_times, pred_times)
    print(f'--- Time Prediction Evaluation ---')
    print(f'MSE: {mse_time:.3f}, MAE: {mae_time:.3f}, R²: {r2_time:.3f}\n')

    # 将预测结果和真实值保存到DataFrame中
    df = pd.DataFrame({
        'True Temperature': [label[0] for label in all_labels],  # 假设第一个值是温度
        'True Time': [label[1] for label in all_labels],  # 假设第二个值是时间
        'Predicted Temperature': [pred[0] for pred in all_preds],  # 假设第一个值是温度
        'Predicted Time': [pred[1] for pred in all_preds]  # 假设第二个值是时间
    })
    # 保存到 csv 文件
    output_path = "reg/all-filter.csv"
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


if __name__ == '__main__':
    main()