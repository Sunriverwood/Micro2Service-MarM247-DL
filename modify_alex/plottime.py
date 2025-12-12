import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from matplotlib.colors import LinearSegmentedColormap
import sys

# --- 全局参数设置 ---
# 设置全局字体大小为 24
plt.rcParams.update({'font.size': 24})

# *** 修改点 1: 将参数从温度改为时间 ***
# 在这里设定您想要筛选的特定时间 (单位: 小时)
target_time = 5000  # 例如, 设置为 5000 来筛选出所有5000h的数据

# --- 颜色定义 (保持不变) ---
target_color_hex = '#91CAE8'
custom_cmap = LinearSegmentedColormap.from_list(
    name='custom_transparent_blue',
    colors=[(*plt.cm.colors.to_rgb(target_color_hex), 0),
            (*plt.cm.colors.to_rgb(target_color_hex), 1)]
)

# --- 排序逻辑 (保持不变，依然有效) ---
def get_sort_key(label):
    match = re.match(r'(\d+)℃_(\d+)h', label)
    if match:
        temp = int(match.group(1))
        hours = int(match.group(2))
        return (temp, hours)
    return (float('inf'), float('inf'))

# --- 数据加载和筛选 ---
try:
    df_full = pd.read_csv('classify/filter-all.csv', index_col=0)
except FileNotFoundError:
    print("错误: 'classify/filter-all.csv' 文件未找到。请检查文件路径。")
    sys.exit()

# *** 修改点 2: 筛选逻辑，从检查前缀改为检查后缀 ***
time_suffix = f'_{target_time}h'
filtered_labels = [label for label in df_full.index if label.endswith(time_suffix)]

# 检查是否找到了任何匹配的标签
if not filtered_labels:
    print(f"错误: 在CSV文件中没有找到任何与时间 {target_time}h 相关的数据。")
    sys.exit()

# 使用 get_sort_key 对筛选出的标签按温度排序
sorted_labels = sorted(filtered_labels, key=get_sort_key)
# 从原始DataFrame中提取数据
df_filtered = df_full.loc[sorted_labels, sorted_labels]

# --- 提取数据用于绘图 ---
class_labels = df_filtered.index.values
cm = df_filtered.values

# 归一化
cm_sum = cm.sum(axis=1)[:, np.newaxis]
cm_norm = cm.astype('float') / (cm_sum + 1e-9)

# --- 绘图 ---
plt.figure(figsize=(16, 14))
plt.imshow(cm_norm, interpolation='nearest', cmap=custom_cmap)
plt.colorbar()

# *** 修改点 3: 创建仅包含温度部分的短标签 ***
# 例如, 将 '850℃_5000h' 转换为 '850℃'
short_labels = [label.split('_')[0] for label in class_labels]
tick_marks = np.arange(len(class_labels))

# 使用短标签设置坐标轴
plt.xticks(tick_marks, short_labels, rotation=90)
plt.yticks(tick_marks, short_labels)

# *** 修改点 4: 将时间信息作为标题 ***
plt.title(f'Condition: {target_time}h', pad=25)

plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# 添加单元格文本
thresh = cm_norm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        if cm[i, j] > 0:
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm_norm[i, j] > thresh else "black")

plt.tight_layout()
# *** 修改点 5: 更新输出文件名 ***
output_filename = f'classify/filter-{target_time}h.png'
# plt.savefig(output_filename, dpi=1200)
plt.show()

print(f"为 {target_time}h 数据生成的混淆矩阵已保存为 '{output_filename}'")