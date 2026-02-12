import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 读取选中特征的数据集
df = pd.read_excel('d:\中风指标\elderly_hypertension_selected_alpha005.xlsx')

print('='*70)
print('7个选中指标的相关性分析')
print('='*70)

# 7个选中的特征
selected_features = ['Dyslipidemia', 'PHR', 'Uric_acid', 'MCV', 'Exercise', 'Age', 'AIP']

print(f'\n分析的变量: {selected_features}')
print(f'样本量: {len(df)} 人')

# 计算相关性矩阵
corr_matrix = df[selected_features].corr()

print('\n相关性矩阵:')
print(corr_matrix.round(3))

# 创建热图
plt.figure(figsize=(10, 8))

# 使用seaborn绘制热图
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # 只显示下三角
sns.heatmap(
    corr_matrix,
    annot=True,           # 显示数值
    fmt='.2f',           # 保留2位小数
    cmap='RdBu_r',       # 红蓝配色
    center=0,            # 以0为中心
    square=True,         # 正方形
    linewidths=0.5,      # 线宽
    cbar_kws={"shrink": 0.8},  # 颜色条大小
    annot_kws={'size': 12}     # 数值字体大小
)

# No title
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.yticks(rotation=0, fontsize=11)
plt.tight_layout()

# 保存热图
heatmap_file = 'd:\中风指标\correlation_heatmap_7features.png'
plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
print(f'\n热图已保存到: {heatmap_file}')

plt.show()

# 找出强相关性（|r| > 0.5）
print('\n' + '='*70)
print('强相关性分析 (|r| > 0.5)')
print('='*70)

strong_corr = []
for i in range(len(selected_features)):
    for j in range(i+1, len(selected_features)):
        corr_val = corr_matrix.iloc[i, j]
        if abs(corr_val) > 0.5:
            strong_corr.append({
                'Var1': selected_features[i],
                'Var2': selected_features[j],
                'Correlation': corr_val
            })

if strong_corr:
    print('\n发现强相关性:')
    for item in strong_corr:
        print(f"  {item['Var1']} ↔ {item['Var2']}: {item['Correlation']:.3f}")
else:
    print('\n未发现强相关性 (|r| > 0.5)，各指标相对独立')

# 与Stroke的相关性
print('\n' + '='*70)
print('各指标与中风(Stroke)的相关性')
print('='*70)

stroke_corr = []
for feature in selected_features:
    corr_val = df[feature].corr(df['Stroke'])
    stroke_corr.append({
        'Feature': feature,
        'Correlation': corr_val,
        'Abs_Correlation': abs(corr_val)
    })

# 按绝对值排序
stroke_corr_df = pd.DataFrame(stroke_corr).sort_values('Abs_Correlation', ascending=False)

print('\n排名:')
for idx, row in stroke_corr_df.iterrows():
    direction = "↑ 正相关" if row['Correlation'] > 0 else "↓ 负相关"
    print(f"  {row['Feature']:<15}: {row['Correlation']:>7.4f}  {direction}")

print('\n' + '='*70)
