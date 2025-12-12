import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# --- 1. 加载数据 ---
# 从CSV文件加载混淆矩阵
try:
    df_cm = pd.read_csv('classify/filter-all.csv', index_col=0)
except FileNotFoundError:
    print("错误: 'all-all.csv' 文件未找到。请确保文件已上传。")

# --- 2. 计算每个类别的准确率 ---
# 预测正确的数量在对角线上
correct_predictions_per_class = np.diag(df_cm)
# 每个类别的总样本数是该行的所有值之和
total_per_class = df_cm.sum(axis=1)

# 计算准确率
accuracy_per_class = correct_predictions_per_class / total_per_class.replace(0, np.nan)

# 将准确率和类别标签合并到一个新的DataFrame中，以便于排序和绘图
accuracy_df = pd.DataFrame({
    'Class': df_cm.index,
    'Accuracy': accuracy_per_class
}).dropna()

# 按准确率从高到低排序
accuracy_df_sorted = accuracy_df.sort_values(by='Accuracy', ascending=False)


# --- 3. 计算总体准确率 ---
# 所有正确的预测数（对角线之和）
total_correct = np.sum(correct_predictions_per_class)
# 数据集总样本数（矩阵所有元素之和）
total_samples = df_cm.to_numpy().sum()
# 计算总体准确率
overall_accuracy = total_correct / total_samples


# --- 4. 绘制柱状图和准确率线 ---
plt.figure(figsize=(18, 11)) # 稍微调整尺寸以适应图例

# 设置全局字体大小
plt.rcParams.update({'font.size': 24})

# 创建柱状图
bars = plt.bar(accuracy_df_sorted['Class'], accuracy_df_sorted['Accuracy'], color=plt.cm.viridis(accuracy_df_sorted['Accuracy']), label='Class Accuracy')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# *** 新增：绘制表示总体准确率的水平线 ***
plt.axhline(y=overall_accuracy, color='red', linestyle='--', linewidth=2.5, label=f'Overall Accuracy: {overall_accuracy:.2%}')

# 添加标题和坐标轴标签
plt.title('Prediction Accuracy per Class vs. Overall Accuracy')
plt.xlabel('Class')
plt.ylabel('Accuracy')
plt.xticks(rotation=90,fontsize=16)

# 设置y轴的范围
plt.ylim(0.7, 1.02)

plt.legend(loc='upper right')
plt.tight_layout()

# --- 5. 保存和输出 ---
plt.savefig('classify/filter-all-zhu.png', dpi=1200)

plt.show()

print(f"计算出的总体准确率为: {overall_accuracy:.4f} ({overall_accuracy:.2%})")
print("\n每个类别的预测准确率柱状图（含总体准确率线）已生成并保存为 'class_vs_overall_accuracy.png'")