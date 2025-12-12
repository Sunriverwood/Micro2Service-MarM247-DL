import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from matplotlib.colors import LinearSegmentedColormap
import sys

# --- 全局参数设置 ---
# *** 修改点: 设置全局字体大小为 24 ***
plt.rcParams.update({'font.size': 24})

# 在这里设定您想要筛选的温度
target_temperature = 900

# --- 颜色定义 ---
target_color_hex = '#91CAE8'#F48892
custom_cmap = LinearSegmentedColormap.from_list(
    name='custom_transparent_blue',
    colors=[(*plt.cm.colors.to_rgb(target_color_hex), 0),
            (*plt.cm.colors.to_rgb(target_color_hex), 1)]
)

# --- 排序逻辑 ---
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
    print("错误: 文件未找到。请检查文件路径。")
    sys.exit()

temp_prefix = f'{target_temperature}℃'
filtered_labels = [label for label in df_full.index if label.startswith(temp_prefix)]

if not filtered_labels:
    print(f"错误: 在CSV文件中没有找到任何与温度 {target_temperature}℃ 相关的数据。")
    sys.exit()

sorted_labels = sorted(filtered_labels, key=get_sort_key)
df_filtered = df_full.loc[sorted_labels, sorted_labels]

# --- 提取数据用于绘图 ---
class_labels = df_filtered.index.values
cm = df_filtered.values

# 归一化
cm_sum = cm.sum(axis=1)[:, np.newaxis]
cm_norm = cm.astype('float') / (cm_sum + 1e-9)

# --- 绘图 ---
plt.figure(figsize=(16, 14)) # 适当增大图像尺寸以容纳更大的字体
plt.imshow(cm_norm, interpolation='nearest', cmap=custom_cmap)
plt.colorbar()

# 创建仅包含时间部分的短标签
short_labels = [label.split('_')[1] for label in class_labels]
tick_marks = np.arange(len(class_labels))

# 使用短标签设置坐标轴
plt.xticks(tick_marks, short_labels, rotation=90)
plt.yticks(tick_marks, short_labels)

# 将温度信息作为标题
plt.title(f'Condition: {target_temperature}°C', pad=25) # 增加标题与图的间距

# 设置坐标轴标题
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
output_filename = f'classify/filter-all-{target_temperature}C.png'
plt.savefig(output_filename, dpi=1200)
plt.show()

print(f"大字体版本的混淆矩阵已保存为 '{output_filename}'")