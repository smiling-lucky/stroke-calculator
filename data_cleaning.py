import pandas as pd
import numpy as np

# 读取数据
df = pd.read_excel('d:\中风指标\wave2026.02.01_removed.xlsx')

print('='*70)
print('数据清洗操作')
print('='*70)

print(f'\n原始数据形状: {df.shape[0]} 行, {df.shape[1]} 列')

# 查看关键字段的基本信息
print('\n关键字段信息:')
print(f'  Kidneye - 唯一值: {sorted(df["Kidneye"].dropna().unique())}')
print(f'  Age - 范围: {df["Age"].min():.1f} ~ {df["Age"].max():.1f}')
print(f'  Stroke - 缺失值数量: {df["Stroke"].isna().sum()}')
print(f'  Stroke - 唯一值: {sorted(df["Stroke"].dropna().unique())}')

# 记录原始行数
original_count = len(df)

# 1. 删除Kidneye=1的记录
print('\n' + '-'*70)
print('步骤1: 删除 Kidneye = 1 的记录')
print('-'*70)
kidneye_1_count = (df['Kidneye'] == 1).sum()
print(f'  Kidneye=1 的记录数: {kidneye_1_count}')
df = df[df['Kidneye'] != 1]
print(f'  删除后剩余记录数: {len(df)}')

# 2. 删除Age<45岁的记录
print('\n' + '-'*70)
print('步骤2: 删除 Age < 45 岁的记录')
print('-'*70)
age_under_45_count = (df['Age'] < 45).sum()
print(f'  Age<45 的记录数: {age_under_45_count}')
df = df[df['Age'] >= 45]
print(f'  删除后剩余记录数: {len(df)}')

# 3. 删除Stroke字段存在缺失值的记录
print('\n' + '-'*70)
print('步骤3: 删除 Stroke 字段存在缺失值的记录')
print('-'*70)
# 检查各种形式的缺失值
stroke_na_count = df['Stroke'].isna().sum()
stroke_blank_count = (df['Stroke'].astype(str).str.strip() == '').sum()
print(f'  Stroke为NaN的记录数: {stroke_na_count}')
print(f'  Stroke为空字符串的记录数: {stroke_blank_count}')

# 删除缺失值（包括NaN和空白字符）
df = df[df['Stroke'].notna()]  # 删除NaN
df = df[df['Stroke'].astype(str).str.strip() != '']  # 删除空白字符
print(f'  删除后剩余记录数: {len(df)}')

# 最终统计
print('\n' + '='*70)
print('清洗完成统计')
print('='*70)
print(f'  原始记录数: {original_count}')
print(f'  最终记录数: {len(df)}')
print(f'  删除记录数: {original_count - len(df)}')
print(f'  保留比例: {len(df)/original_count*100:.2f}%')

# 验证清洗结果
print('\n' + '='*70)
print('验证清洗结果')
print('='*70)
print(f'  Kidneye = 1 的记录数: {(df["Kidneye"] == 1).sum()} (应为0)')
print(f'  Age < 45 的记录数: {(df["Age"] < 45).sum()} (应为0)')
print(f'  Stroke 缺失值数量: {df["Stroke"].isna().sum()} (应为0)')
print(f'  Age 范围: {df["Age"].min():.1f} ~ {df["Age"].max():.1f}')

# 保存清洗后的数据
output_file = 'd:\中风指标\wave2026.02.01_cleaned.xlsx'
df.to_excel(output_file, index=False)
print(f'\n清洗后的数据已保存到: {output_file}')
print('='*70)
