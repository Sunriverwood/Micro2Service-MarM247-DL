import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- 全局绘图设置 ---
plt.rcParams.update({'font.size': 24})

df = pd.read_csv('inception-predict.csv')

# --- 绘制小提琴图 ---
plt.figure(figsize=(16, 10))

# 对类别进行排序
df = df.sort_values(by='True Stress')

# 设置渐变色
start_color = "#91CAE8"
end_color = "#A2C9AE"
n_colors = len(df['True Stress'].unique())
stress_palette = sns.color_palette(f"blend:{start_color},{end_color}", n_colors=n_colors)

# 绘制小提琴图
ax = sns.violinplot(
    x=df['True Stress'].astype(int),
    y='Predicted Stress',
    data=df,
    hue='True Stress',
    palette=stress_palette,
    legend=False
)

# 添加真实值标记
unique_stress = sorted(df['True Stress'].unique())
plt.scatter(
    x=range(len(unique_stress)),
    y=unique_stress,
    color='red',
    marker='*',
    s=250,
    zorder=3,
    label='True Value'
)

plt.legend()
plt.xlabel('True Stress')
plt.ylabel('Predicted Stress')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('inc-predict.png')
plt.show()