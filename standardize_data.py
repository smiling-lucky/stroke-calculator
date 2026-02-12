import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# 读取数据
df = pd.read_excel('d:\中风指标\elderly_hypertension_60plus_imputed.xlsx')

print('='*70)
print('连续变量标准化处理')
print('='*70)

print(f'\n原始数据形状: {df.shape[0]} 行, {df.shape[1]} 列')

# 分离ID列
id_col = df['ID'].copy()
df_features = df.drop(columns=['ID'])

# 识别连续变量（排除二分类变量）
binary_vars = ['Gender', 'Marry', 'Hearte', 'Stroke', 'Dyslipidemia', 'Livere', 
               'Alcohol_consumption', 'Smoken', 'Hypertension', 'Diabetes', 'Exercise']

continuous_vars = [col for col in df_features.columns if col not in binary_vars]

print(f'\n变量分类:')
print(f'  二分类变量 ({len(binary_vars)} 个): {binary_vars}')
print(f'  连续变量 ({len(continuous_vars)} 个): {continuous_vars}')

# 对连续变量进行标准化
print('\n正在进行标准化处理 (Z-score标准化)...')
scaler = StandardScaler()
df_features[continuous_vars] = scaler.fit_transform(df_features[continuous_vars])

# 将ID列加回来
df_standardized = pd.concat([id_col, df_features], axis=1)

# 确保列顺序与原始数据一致
df_standardized = df_standardized[df.columns]

print('\n标准化完成！')

# 显示标准化前后的对比
print('\n连续变量标准化前后对比:')
print(f"{'变量名':<20}{'标准化前均值':<15}{'标准化后均值':<15}{'标准化前标准差':<15}{'标准化后标准差':<15}")
print('-'*80)

# 重新读取原始数据进行对比
df_original = pd.read_excel('d:\中风指标\elderly_hypertension_60plus_imputed.xlsx')
for col in continuous_vars[:10]:  # 显示前10个连续变量
    before_mean = df_original[col].mean()
    after_mean = df_standardized[col].mean()
    before_std = df_original[col].std()
    after_std = df_standardized[col].std()
    print(f"{col:<20}{before_mean:<15.3f}{after_mean:<15.3f}{before_std:<15.3f}{after_std:<15.3f}")

# 验证二分类变量未被改变
print('\n二分类变量验证（应保持0/1）:')
for var in binary_vars[:5]:  # 显示前5个
    unique_vals = sorted(df_standardized[var].unique())
    print(f'  {var}: {unique_vals}')

# 验证ID列未被改变
print(f'\nID列验证:')
print(f'  原始ID前5个: {df_original["ID"].head().tolist()}')
print(f'  标准化后ID前5个: {df_standardized["ID"].head().tolist()}')

# 保存标准化后的数据集
output_file = 'd:\中风指标\elderly_hypertension_60plus_standardized.xlsx'
df_standardized.to_excel(output_file, index=False)

print(f'\n标准化后的数据集已保存到:')
print(f'  {output_file}')
print(f'\n数据集信息:')
print(f'  - 样本量: {len(df_standardized)} 人')
print(f'  - 变量数: {len(df_standardized.columns)} 个')
print(f'  - 连续变量已标准化 (均值≈0, 标准差≈1)')
print(f'  - 二分类变量保持0/1')
print(f'  - ID列保持不变')

print('\n' + '='*70)
