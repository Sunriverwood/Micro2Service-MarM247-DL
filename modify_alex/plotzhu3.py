import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

# --- 1. 全局设置 ---

plt.rcParams.update({
    'font.size': 32,  # Set font size
    'font.family': 'Arial'
})
# 定义要处理的文件和它们在图例中的标签
# 请确保这些文件都在 'classify/' 目录下
file_info = {
    'all-all': 'all-all',
    'all-filter': 'all-filter',
    'filter-all': 'filter-all',
    'filter-filter': 'filter-filter'
}
csv_files = {label: f"classify/{filename}.csv" for label, filename in file_info.items()}


# --- 2. 定义准确率计算函数 ---

def calculate_accuracy(filepath):
    """从单个CSV文件计算按温度和时间分组的准确率"""
    try:
        df_cm = pd.read_csv(filepath, index_col=0)
    except FileNotFoundError:
        print(f"警告: 文件 {filepath} 未找到，将跳过。")
        return None, None

    try:
        temperatures = df_cm.index.str.extract(r'(\d+)℃')[0].astype(int)
        times = df_cm.index.str.extract(r'_(\d+)h')[0].astype(int)
    except (AttributeError, IndexError, TypeError):
        print(f"警告: 无法解析文件 {filepath} 的索引，格式可能不正确。将跳过。")
        return None, None

    cm_values = df_cm.to_numpy()
    df_agg = pd.DataFrame({
        'Temperature': temperatures,
        'Time': times,
        'Correct_Predictions': np.diag(cm_values),
        'Total_Samples': cm_values.sum(axis=1)
    })

    # 按温度计算
    acc_temp = df_agg.groupby('Temperature').sum()
    acc_temp['Accuracy'] = acc_temp['Correct_Predictions'] / acc_temp['Total_Samples'].replace(0, np.nan)

    # 按时间计算
    acc_time = df_agg.groupby('Time').sum()
    acc_time['Accuracy'] = acc_time['Correct_Predictions'] / acc_time['Total_Samples'].replace(0, np.nan)

    return acc_temp[['Accuracy']].dropna(), acc_time[['Accuracy']].dropna()


# --- 3. 循环处理所有文件并合并数据 ---

temp_results = []
time_results = []

for label, filepath in csv_files.items():
    acc_temp, acc_time = calculate_accuracy(filepath)
    if acc_temp is not None:
        acc_temp['source'] = label  # 添加来源标签
        temp_results.append(acc_temp)
    if acc_time is not None:
        acc_time['source'] = label  # 添加来源标签
        time_results.append(acc_time)

# 将所有结果合并到一个DataFrame中
combined_temp = pd.concat(temp_results).reset_index()
combined_time = pd.concat(time_results).reset_index()

# --- 4. 重构数据以进行分组绘图 (Pivot) ---
# 将'source'列的值作为新的列，'Accuracy'作为值
temp_pivot = combined_temp.pivot(index='Temperature', columns='source', values='Accuracy')
time_pivot = combined_time.pivot(index='Time', columns='source', values='Accuracy')

# 确保列的顺序是我们定义的顺序
temp_pivot = temp_pivot[file_info.keys()]
time_pivot = time_pivot[file_info.keys()]


# --- 5. 定义分组柱状图的绘图函数 ---

def plot_grouped_bar_chart(pivot_df, title, xlabel, ylabel, output_filename, unit='°C'):
    """一个通用的函数，用于绘制分组柱状图"""
    # colors = ['#91CAE8', '#A2C9AE', '#F48892', '#F5B3A5']
    colors = ['#8EC8ED', '#AED594', '#D693BE', '#F5B3A5']
    n_groups = len(pivot_df.index)
    n_bars = len(pivot_df.columns)

    fig, ax = plt.subplots(figsize=(18, 10))

    bar_width = 0.8 / n_bars
    index = np.arange(n_groups)

    # 循环为每个来源(CSV文件)绘制一组柱子
    for i, column in enumerate(pivot_df.columns):
        # 计算每组柱子的位置
        offset = (i - (n_bars - 1) / 2) * bar_width
        bar_positions = index + offset

        ax.bar(bar_positions, pivot_df[column].fillna(0), bar_width, label=column, color=colors[i])
        # 不显示柱子上的数值

    # 在坐标轴标题中写入单位
    ax.set_xlabel(f"{xlabel} ({unit})", fontsize=32)
    ax.set_ylabel(ylabel, fontsize=32)

    # 设置X轴刻度标签在组的中心
    ax.set_xticks(index)
    ax.set_xticklabels([f"{int(i)}" for i in pivot_df.index], rotation=45, ha='center')

    ax.legend(fontsize=32, loc='lower left', bbox_to_anchor=(0, 0), frameon=True, borderaxespad=0.5)
    # 不显示网格线
    ax.grid(axis='y', linestyle=(0,(12,6)), alpha=0.5)
    ax.set_ylim(0.5, 1.02)

    fig.tight_layout()
    plt.savefig(f'classify/{output_filename}', dpi=600)
    print(f"\n分组柱状图已保存为 '{output_filename}'")


# --- 6. 调用绘图函数生成两个图表 ---

# 绘制按温度分组的图
plot_grouped_bar_chart(
    pivot_df=temp_pivot,
    title='Comparison of Accuracy by Temperature across different sources',
    xlabel='Temperature',
    ylabel='Group Accuracy',
    output_filename='grouped_accuracy_by_temperature-1.png',
    unit='°C'
)

# 绘制按时间分组的图
plot_grouped_bar_chart(
    pivot_df=time_pivot,
    title='Comparison of Accuracy by Time across different sources',
    xlabel='Time',
    ylabel='Group Accuracy',
    output_filename='grouped_accuracy_by_time-1.png',
    unit='h'
)

plt.show()