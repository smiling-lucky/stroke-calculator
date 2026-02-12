import pandas as pd

# 读取数据
df = pd.read_excel('d:\中风指标\wave2026.02.01_removed.xlsx')

print('='*60)
print('保存60岁以上高血压人群数据集')
print('='*60)

print(f'\n原始数据形状: {df.shape[0]} 行, {df.shape[1]} 列')

# 筛选条件: Age >= 60 且 Hypertension = 1
df_elderly_ht = df[(df['Age'] >= 60) & (df['Hypertension'] == 1)].copy()

print(f'\n筛选条件: Age >= 60 且 Hypertension = 1')
print(f'筛选后数据形状: {df_elderly_ht.shape[0]} 行, {df_elderly_ht.shape[1]} 列')

# 保存数据集
output_file = 'd:\中风指标\elderly_hypertension_60plus.xlsx'
df_elderly_ht.to_excel(output_file, index=False)

print(f'\n数据集已保存到: {output_file}')
print('\n数据集信息:')
print(f'  - 样本量: {len(df_elderly_ht)} 人')
print(f'  - 变量数: {len(df_elderly_ht.columns)} 个')
print(f'  - 年龄范围: {df_elderly_ht["Age"].min():.0f} - {df_elderly_ht["Age"].max():.0f} 岁')
print(f'  - 平均年龄: {df_elderly_ht["Age"].mean():.1f} 岁')
print(f'  - 男性比例: {(df_elderly_ht["Gender"] == 1).mean()*100:.1f}%')
print(f'  - 中风患病率: {df_elderly_ht["Stroke"].mean()*100:.1f}%')

print('\n变量列表:')
for i, var in enumerate(df_elderly_ht.columns, 1):
    print(f'  {i:2d}. {var}')

print('\n' + '='*60)
