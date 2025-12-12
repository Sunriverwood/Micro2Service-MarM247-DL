import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from matplotlib.colors import LinearSegmentedColormap

# --- 颜色定义开始 ---

# 1. 定义你的目标颜色
target_color_hex = '#F48892'
# 91CAE8  蓝
# F48892  红
# 2. 创建一个从完全透明到完全不透明的颜色列表
#    - 起始颜色: 目标颜色，但alpha(透明度)为0 (完全透明)
#    - 结束颜色: 目标颜色，但alpha(透明度)为1 (完全不透明)
#    Matplotlib的 from_list 函数会在这两个点之间创建一个平滑的渐变
custom_cmap = LinearSegmentedColormap.from_list(
    name='custom_transparent_red',
    colors=[(*plt.cm.colors.to_rgb(target_color_hex), 0),  # (R, G, B, Alpha=0)
            (*plt.cm.colors.to_rgb(target_color_hex), 1)]  # (R, G, B, Alpha=1)
)

# --- 颜色定义结束 ---


# --- 排序逻辑 ---
def get_sort_key(label):
    match = re.match(r'(\d+)℃_(\d+)h', label)
    if match:
        temp = int(match.group(1))
        hours = int(match.group(2))
        return (temp, hours)
    return (9999, 9999)

# 从CSV文件加载混淆矩阵
try:
    df = pd.read_csv('classify/all-all.csv', index_col=0)
except FileNotFoundError:
    print("错误: 'confusion_matrix.csv' 文件未找到。")
    exit()

# 获取原始标签并排序
original_labels = df.index.tolist()
sorted_labels = sorted(original_labels, key=get_sort_key)

# 重排DataFrame
df_sorted = df.reindex(index=sorted_labels, columns=sorted_labels)

# 提取数据
class_labels = df_sorted.index.values
cm = df_sorted.values

# 归一化
cm_sum = cm.sum(axis=1)[:, np.newaxis]
cm_norm = cm.astype('float') / (cm_sum + 1e-8)

# --- 绘图 ---
plt.figure(figsize=(14, 12))

# 3. 在imshow中使用我们自定义的颜色映射 custom_cmap
plt.imshow(cm_norm, interpolation='nearest', cmap=custom_cmap) # <-- 这里应用了自定义颜色

plt.title('Original Confusion Matrix', fontsize=20)
plt.colorbar()

# 设置坐标轴
tick_marks = np.arange(len(class_labels))
plt.xticks(tick_marks, class_labels, rotation=90, fontsize=10)
plt.yticks(tick_marks, class_labels, fontsize=10)
plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('True Label', fontsize=14)

# 添加单元格文本
thresh = cm_norm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        if cm[i, j] > 0:
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     fontsize=7,
                     color="white" if cm_norm[i, j] > thresh else "black")

plt.tight_layout()
plt.savefig('classify/confusion_matrix.png', dpi=1200)
plt.show()

print("自定义颜色和透明度的混淆矩阵已保存为 'confusion_matrix.png'")