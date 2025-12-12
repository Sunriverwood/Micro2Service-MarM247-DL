import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


# --- 1. 加载数据 ---
try:
    df_cm = pd.read_csv('classify/all-all.csv', index_col=0)
except FileNotFoundError:
    print("错误: 'classify/all-all.csv' 文件未找到。请确保文件已上传或在正确路径下。")
    # 创建一个示例DataFrame以防文件不存在
    data = {'700℃_100h': [10, 1], '850℃_100h': [2, 15], '700℃_300h': [0, 20]}
    df_cm = pd.DataFrame(data, index=['700℃_100h', '850℃_100h', '700℃_300h'])


# --- 2. 准备数据用于分组 (同时提取温度和时间) ---
try:
    # 从索引（类别标签）中提取温度和时间数值
    temperatures = df_cm.index.str.extract(r'(\d+)℃')[0].astype(int)
    times = df_cm.index.str.extract(r'_(\d+)h')[0].astype(int)
except (AttributeError, IndexError, TypeError):
    print("错误：无法从CSV的索引中解析出温度和时间。请检查索引格式是否为 'XXX℃_YYYh'。")
    exit()

# 将正确预测数和总样本数作为新列添加到DataFrame中
cm_values = df_cm.to_numpy()
df_agg = pd.DataFrame({
    'Temperature': temperatures,
    'Time': times,
    'Correct_Predictions': np.diag(cm_values),
    'Total_Samples': cm_values.sum(axis=1)
})


# ==============================================================================
# --- 按温度分组绘图 (第一个图) ---
# ==============================================================================

# 按温度分组并计算准确率
accuracy_by_temp = df_agg.groupby('Temperature').sum()
accuracy_by_temp['Accuracy'] = accuracy_by_temp['Correct_Predictions'] / accuracy_by_temp['Total_Samples'].replace(0, np.nan)
accuracy_by_temp.dropna(inplace=True)
accuracy_by_temp = accuracy_by_temp.sort_index()

# 绘图
plt.figure(figsize=(12, 8))
plt.rcParams.update({'font.size': 24})

x_labels_temp = [f"{temp}℃" for temp in accuracy_by_temp.index]

custom_cmap = LinearSegmentedColormap.from_list('custom', ['#F48892', '#FF5E00'])
normed_acc = (accuracy_by_temp['Accuracy'] - 0.7) / (1 - 0.7)  # 0为最小，1为最大
bars_temp = plt.bar(x_labels_temp, accuracy_by_temp['Accuracy'], color=custom_cmap(normed_acc))

plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.title('Prediction Accuracy by Temperature Group')
plt.xlabel('Temperature')
plt.ylabel('Group Accuracy')
plt.ylim(0.7, 1.02)

for bar in bars_temp:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2%}', ha='center', va='bottom', fontsize=20)

plt.tight_layout()
output_filename_temp = 'temperature_group_accuracy.png'
# plt.savefig(output_filename_temp, dpi=1200)

print("--- 按温度分组的准确率 ---")
print(accuracy_by_temp)
print(f"\n图表已保存为 '{output_filename_temp}'")


# ==============================================================================
# --- 按时间分组绘图 (新增的第二个图) ---
# ==============================================================================

# 按时间分组并计算准确率
accuracy_by_time = df_agg.groupby('Time').sum()
accuracy_by_time['Accuracy'] = accuracy_by_time['Correct_Predictions'] / accuracy_by_time['Total_Samples'].replace(0, np.nan)
accuracy_by_time.dropna(inplace=True)
accuracy_by_time = accuracy_by_time.sort_index()

# 绘图
plt.figure(figsize=(16, 8)) # 时间跨度可能更大，用更宽的图
plt.rcParams.update({'font.size': 24})

x_labels_time = [f"{time}h" for time in accuracy_by_time.index]

custom_cmap_time = LinearSegmentedColormap.from_list('custom_time', ['#91CAE8', '#A2C9AE'])
normed_acc_time = (accuracy_by_time['Accuracy'] - 0.7) / (1 - 0.7)  # 0为最小，1为最大
bars_time = plt.bar(x_labels_time, accuracy_by_time['Accuracy'], color=custom_cmap_time(normed_acc_time))


plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.title('Prediction Accuracy by Time Group')
plt.xlabel('Time')
plt.ylabel('Group Accuracy')
plt.xticks(rotation=15) # 时间标签可能需要旋转
plt.ylim(0.7, 1.02)

for bar in bars_time:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2%}', ha='center', va='bottom', fontsize=20)

plt.tight_layout()
output_filename_time = 'time_group_accuracy.png'
# plt.savefig(output_filename_time, dpi=1200)

print("\n--- 按时间分组的准确率 ---")
print(accuracy_by_time)
print(f"\n图表已保存为 '{output_filename_time}'")

# 同时显示所有生成的图表
plt.show()