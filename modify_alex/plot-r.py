import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def create_frequency_spaced_plot(ax, data, true_col, pred_col, unit):
    """
    创建一个特殊的散点图，其X轴的间距由数据点的频率决定。

    :param ax: Matplotlib的Axes对象，用于绘图。
    :param data: 包含数据的DataFrame。
    :param true_col: 真实值列的名称。
    :param pred_col: 预测值列的名称。
    :param unit: 坐标轴标签的单位（如 'C' 或 'h'）。
    """
    # 1. 计算每个真实值的频率
    value_counts = data[true_col].value_counts().sort_index()

    # 2. 计算每个真实值在新X轴上的位置和宽度
    tick_positions = {}
    tick_widths = {}
    current_pos = 0
    for value, count in value_counts.items():
        # 每个区域的宽度等于其样本数
        width = count
        # 刻度标签放在每个区域的中心
        tick_positions[value] = current_pos + width / 2
        tick_widths[value] = width
        current_pos += width

    # 3. 为每个数据点计算其在新的X轴上的坐标
    def calculate_new_x(row):
        true_value = row[true_col]
        # 获取该真实值对应的区域的起始位置
        start_pos = tick_positions[true_value] - tick_widths[true_value] / 2
        # 在该区域内添加随机抖动，以避免点重叠
        return start_pos + np.random.rand() * tick_widths[true_value]

    data['new_x'] = data.apply(calculate_new_x, axis=1)

    # 4. 绘图
    # 绘制散点图
    ax.scatter(data['new_x'], data[pred_col], alpha=0.6, s=50, label='Predicted Values')

    # 绘制 "预测=真实" 的参考线（阶梯状）
    for value, pos in tick_positions.items():
        start_pos = pos - tick_widths[value] / 2
        end_pos = pos + tick_widths[value] / 2
        ax.hlines(y=value, xmin=start_pos, xmax=end_pos, color='red', linestyle='-', linewidth=2, label='True' if value == value_counts.index[0] else "")

    # 5. 设置坐标轴
    # 设置X轴刻度和标签
    ax.set_xticks(list(tick_positions.values()))
    ax.set_xticklabels([f"{int(val)}{unit}" for val in tick_positions.keys()], rotation=45, ha='right')

    # 设置标题和Y轴标签
    title_name = 'Temperature' if 'Temp' in true_col else 'Time'
    ax.set_title(f'Predicted vs. True {title_name} (Frequency Spaced)', fontsize=16, pad=15)
    ax.set_ylabel(f'Predicted Value ({unit})', fontsize=12)
    ax.set_xlabel(f'True Value ({unit}) - Spaced by Sample Count', fontsize=12)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)


# --- 主程序 ---
# 1. 加载数据
try:
    df = pd.read_csv("reg/predictions_results.csv")
except FileNotFoundError:
    print("错误: 'reg-0C/predictions_results.csv' 文件未找到。")
    exit()

# 2. 创建图表布局
fig, axes = plt.subplots(2, 1, figsize=(16, 18))
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 3. 调用函数分别绘制温度和时间的图表
print("正在生成频率间隔的温度图...")
create_frequency_spaced_plot(axes[0], df.copy(), 'True Temperature', 'Predicted Temperature', 'C')

print("正在生成频率间隔的时间图...")
create_frequency_spaced_plot(axes[1], df.copy(), 'True Time', 'Predicted Time', 'h')

# 4. 调整布局并保存
plt.tight_layout(pad=4.0)
output_image_path = "prediction_vs_true.png"
# plt.savefig(output_image_path, dpi=1200)
plt.show()

print(f"图表已成功保存为 '{output_image_path}'")