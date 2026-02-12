import pandas as pd

# 读取原始数据
df = pd.read_excel('d:\中风指标\wave2026.02.01.xlsx')

print('='*60)
print('删除变量并保存新表格')
print('='*60)

print(f'\n原始数据形状: {df.shape[0]} 行, {df.shape[1]} 列')
print(f'原始变量数: {len(df.columns)}')

# 要删除的变量
vars_to_remove = [
    'Height', 'Weight', 'Waist',           # 人体测量
    'SBP', 'DBP',                           # 血压
    'TC', 'TG', 'HDL-C', 'LDL-C',          # 血脂
    'Fasting', 'Blood_glucose', 'HbA1c',   # 血糖
    'Scr', 'BUN'                            # 肾功能
]

print('\n要删除的变量:')
for i, var in enumerate(vars_to_remove, 1):
    status = "存在" if var in df.columns else "不存在"
    print(f'  {i:2d}. {var:<20} ({status})')

# 检查哪些变量实际存在于数据集中
existing_vars_to_remove = [var for var in vars_to_remove if var in df.columns]
not_found_vars = [var for var in vars_to_remove if var not in df.columns]

if not_found_vars:
    print(f'\n以下变量在数据集中未找到: {not_found_vars}')

# 删除变量
df_new = df.drop(columns=existing_vars_to_remove)

print(f'\n删除后数据形状: {df_new.shape[0]} 行, {df_new.shape[1]} 列')
print(f'剩余变量数: {len(df_new.columns)}')

# 显示剩余的变量
print('\n剩余变量列表:')
for i, var in enumerate(df_new.columns, 1):
    print(f'  {i:2d}. {var}')

# 保存新表格
output_file = 'd:\中风指标\wave2026.02.01_removed.xlsx'
df_new.to_excel(output_file, index=False)
print(f'\n新表格已保存到: {output_file}')
print('='*60)
