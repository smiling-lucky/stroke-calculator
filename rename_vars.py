import pandas as pd

# 读取数据
df = pd.read_excel('d:\中风指标\wave2026.02.01_removed.xlsx')

print('='*60)
print('重命名变量')
print('='*60)

print(f'\n当前数据形状: {df.shape[0]} 行, {df.shape[1]} 列')

# 重命名映射
rename_dict = {
    'Hypertension.1': 'Hypertension',
    'Diabetes_mellitus.1': 'Diabetes'
}

print('\n重命名计划:')
for old_name, new_name in rename_dict.items():
    if old_name in df.columns:
        print(f'  {old_name} -> {new_name}')
    else:
        print(f'  {old_name} -> {new_name} (原变量不存在)')

# 执行重命名
df_renamed = df.rename(columns=rename_dict)

print('\n重命名后的变量列表:')
for i, var in enumerate(df_renamed.columns, 1):
    print(f'  {i:2d}. {var}')

# 保存新表格（覆盖原文件）
output_file = 'd:\中风指标\wave2026.02.01_removed.xlsx'
df_renamed.to_excel(output_file, index=False)
print(f'\n更新后的表格已保存到: {output_file}')
print('='*60)
