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
            try:
                # 解析文件夹名，提取时间标签
                time_exposed = float(folder_name.replace("h", ""))  # 文件夹名应表示时间，例如 "10h"
            except ValueError:
                print(f"Skipping invalid folder name: {folder_name}")
                continue

            # 遍历文件夹中的图片文件
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):  # 检查图片格式
                    self.samples.append(file_path)  # 添加图片路径
                    self.labels.append([time_exposed])  # 添加时间标签

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]  # 真实值 (时间)
        img = Image.open(img_path)

        if self.transform:
            img = self.transform(img)

        return img, label, img_path


def calculate_confidence_interval_with_min_variance(predictions, confidence_level=0.70):
    """
    计算包含给定置信度（如70%）的区间，并确保区间内的方差最小
    :param predictions: 所有预测值（对某个真实值的多次预测结果）
    :param confidence_level: 置信度水平（默认70%）
    :return: 置信区间的上下限
    """
    predictions_sorted = np.sort(predictions)  # 对预测结果进行排序

    # 计算包含70%数据的区间长度
    n = len(predictions_sorted)
    interval_length = int(n * confidence_level)

    # 初始化最小方差和最优区间
    min_variance = float('inf')
    best_lower_bound = best_upper_bound = None

    # 遍历所有可能的区间
    for i in range(n - interval_length + 1):
        lower_bound = predictions_sorted[i]
        upper_bound = predictions_sorted[i + interval_length - 1]

        # 计算该区间内的方差
        variance = np.var(predictions_sorted[i:i + interval_length])

        # 如果方差更小，更新最小方差和对应区间
        if variance < min_variance:
            min_variance = variance
            best_lower_bound = lower_bound
            best_upper_bound = upper_bound

    return best_lower_bound, best_upper_bound


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 创建模型
    from Alex_model import AlexNet  # 确保 AlexNet 的输出为 1（仅时间）
    model = AlexNet(num_outputs=1).to(device)  # 修改为仅预测时间

    # 加载模型权重
    weights_path = "AlexNet_regression_3.pth"
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
    maximum_error=0
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
            if error > 500:
                error_images.append(img_path[0])  # 记录图片路径
            if error>maximum_error:
                maximum_error=error.item()

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

    # 计算每个真实值的置信区间（70%范围）
    for true_value in np.unique(all_labels):
        # 获取所有预测值
        predictions_for_value = all_preds[all_labels.flatten() == true_value]
        lower, upper = calculate_confidence_interval_with_min_variance(predictions_for_value)
        print(f"Real Time: {true_value.item():.2f}, Confidence Interval (70%): ({lower.item():.2f}, {upper.item():.2f}),Negative Deviation:{(true_value-lower).item():.2f},Positive Deviation:{(upper-true_value).item():.2f}")

    # 可视化预测与真实值的对比（时间）
    plt.figure(figsize=(10, 5))
    plt.plot(all_labels[:, 0], label='True Time', color='green', linestyle='dashed')
    plt.plot(all_preds, label='Predicted Time', color='orange')
    plt.legend()
    plt.title('True vs Predicted Time')
    plt.xlabel('Sample Index')
    plt.ylabel('Time (h)')
    plt.savefig('time_comparison_3.png')
    plt.show()

    # 输出预测偏差较大的图片路径
    if error_images:
        print("Images with large prediction errors:")
        for img_path in error_images:
            print(img_path)
    else:
        print("No significant prediction errors found.")

    print(f"Maximum error: {maximum_error:.2f}")

if __name__ == '__main__':
    main()