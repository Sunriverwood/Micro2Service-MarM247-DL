import os
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy.stats import f
from matplotlib.patches import Ellipse
from Alex_model3 import AlexNet
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def normalize_data(all_preds, all_labels):
    """数据归一化处理"""
    # 温度的标准化
    scaler_temp = StandardScaler()
    temp_preds = all_preds[:, 0].reshape(-1, 1)
    temp_labels = all_labels[:, 0].reshape(-1, 1)
    temp_preds = scaler_temp.fit_transform(temp_preds)
    temp_labels = scaler_temp.transform(temp_labels)

    # 时间的最小-最大归一化
    scaler_time = MinMaxScaler()
    time_preds = all_preds[:, 1].reshape(-1, 1)
    time_labels = all_labels[:, 1].reshape(-1, 1)
    time_preds = scaler_time.fit_transform(time_preds)
    time_labels = scaler_time.transform(time_labels)

    # 返回归一化后的数据
    return np.hstack((temp_preds, time_preds)), np.hstack((temp_labels, time_labels))


# def plot_confidence_ellipse(coefs, covariance_matrix, ax=None, n_std=1.96, color='blue'):
#     """
#     绘制回归系数的置信椭圆图。
#     :param coefs: 回归系数（包含截距项）
#     :param covariance_matrix: 回归系数的协方差矩阵
#     :param ax: Matplotlib的坐标轴对象，如果没有提供，则创建新的
#     :param n_std: 置信区间的标准差倍数（默认为 1.96，表示 95% 置信区间）
#     :param color: 椭圆的颜色
#     """
#     # 如果没有提供坐标轴对象，则创建一个新的
#     if ax is None:
#         fig, ax = plt.subplots()
#
#     # 确保协方差矩阵与回归系数的维度一致
#     if covariance_matrix.shape[0] > 2:
#         covariance_matrix = covariance_matrix[:2, :2]
#
#     # 计算协方差矩阵的特征值和特征向量
#     eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
#
#     # 计算椭圆的主轴长度（根据协方差矩阵的特征值）
#     width, height = 2 * n_std * np.sqrt(eigenvalues)
#
#     # 旋转角度的正确计算方向
#     angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))  # 直接计算xy方向的角度
#
#     # 创建椭圆并绘制
#     ellipse = Ellipse(xy=coefs, width=width, height=height, angle=angle, color=color, alpha=0.5)
#     ax.add_patch(ellipse)
#
#     # 设置图形的范围和标签
#     ax.set_xlim(coefs[0] - 1.5 * width, coefs[0] + 1.5 * width)
#     ax.set_ylim(coefs[1] - 1.5 * height, coefs[1] + 1.5 * height)
#     ax.set_xlabel('Coefficient of Temperature')
#     ax.set_ylabel('Coefficient of Time')
#     ax.set_title('Confidence Ellipse for Time(95%)')
#
#     # 显示图形
#     plt.grid(True)
#     plt.show()

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 创建模型
    model = AlexNet(num_outputs=2).to(device)

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
    all_img_paths = []  # 记录图片路径
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
            all_img_paths.append(img_path[0])  # 保存图片路径

            # # 如果偏差较大，记录图片路径
            # error = np.abs(output - true_label)
            # if np.any(error > [50, 1000]):  # 假设温度偏差 > 50 或时间偏差 > 1000
            #     error_images.append(img_path[0])  # 记录图片路径

            total += 1

    # 将预测结果和真实值保存到DataFrame中
    df = pd.DataFrame({
        'Image Path': all_img_paths,
        'True Temperature': [label[0] for label in all_labels],  # 假设第一个值是温度
        'True Time': [label[1] for label in all_labels],  # 假设第二个值是时间
        'Predicted Temperature': [pred[0] for pred in all_preds],  # 假设第一个值是温度
        'Predicted Time': [pred[1] for pred in all_preds]  # 假设第二个值是时间
    })
    # 保存到 Excel 文件
    output_path = "predictions_results.xlsx"
    df.to_excel(output_path, index=False, engine='openpyxl')
    print(f"Results saved to {output_path}")

    # # 转换为 NumPy 数组
    # all_preds = np.array(all_preds)
    # all_labels = np.array(all_labels)

    # pred_temp = all_preds[:, 0]
    # pred_time = all_preds[:, 1]
    #
    # true_temp = all_labels[:, 0]
    # true_time = all_labels[:, 1]
    #
    # # 绘制散点图
    # plt.figure(figsize=(14, 6))
    #
    # # 温度与时间对比
    # plt.scatter(true_temp, true_time, color='blue', label='real', s=30, marker='o')
    # plt.scatter(pred_temp, pred_time, color='red', label='pred', s=30, marker='x')
    # plt.title('Real VS Predict')
    # plt.xlabel('Temperature(℃)')
    # plt.ylabel('Time (h)')
    #
    # plt.legend()
    #
    # plt.tight_layout()
    # plt.show()

    # # 归一化数据
    # normalized_preds, normalized_labels = normalize_data(all_preds, all_labels)
    #
    # # 线性回归
    # X = normalized_preds  # 归一化后的预测值
    # y = normalized_labels[:, 1]  # 归一化后的温度
    #
    # # 训练回归模型
    # regressor = LinearRegression()
    # regressor.fit(X, y)
    #
    # # 计算回归系数的置信区间（置信度 95%）
    # X_with_intercept = sm.add_constant(X)  # 添加常数项（截距）
    # model_sm = sm.OLS(y, X_with_intercept)  # 使用 statsmodels 的 OLS 模型
    # results = model_sm.fit()  # 拟合模型
    # conf_interval = results.conf_int(alpha=0.05)  # 计算95%置信区间
    #
    # print("回归系数置信区间 (95%):")
    # print(conf_interval)
    #
    # # 获取回归系数
    # coefs = results.params  # 截距和回归系数
    #
    # # 获取协方差矩阵
    # covariance_matrix = results.cov_params()  # 获取协方差矩阵
    #
    # # 绘制回归系数的置信椭圆图
    # plot_confidence_ellipse(coefs, covariance_matrix)

    # # 可视化预测与真实值的对比（温度）
    # plt.figure(figsize=(10, 5))
    # plt.plot(all_labels[:, 0], label='True Temperature', color='blue', linestyle='dashed')
    # plt.plot(all_preds[:, 0], label='Predicted Temperature', color='red')
    # plt.legend()
    # plt.title('True vs Predicted Temperature')
    # plt.xlabel('Sample Index')
    # plt.ylabel('Temperature (℃)')
    # plt.savefig('temperature_comparison_3.png')
    # plt.show()
    #
    # # 可视化预测与真实值的对比（时间）
    # plt.figure(figsize=(10, 5))
    # plt.plot(all_labels[:, 1], label='True Time', color='green', linestyle='dashed')
    # plt.plot(all_preds[:, 1], label='Predicted Time', color='orange')
    # plt.legend()
    # plt.title('True vs Predicted Time')
    # plt.xlabel('Sample Index')
    # plt.ylabel('Time (h)')
    # plt.savefig('time_comparison_3.png')
    # plt.show()

    # # 输出预测偏差较大的图片路径
    # if error_images:
    #     print("Images with large prediction errors:")
    #     for img_path in error_images:
    #         print(img_path)
    # else:
    #     print("No significant prediction errors found.")


if __name__ == '__main__':
    main()
