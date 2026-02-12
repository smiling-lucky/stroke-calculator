import pandas as pd

# 读取数据
df = pd.read_excel('d:\中风指标\wave2026.02.01_cleaned.xlsx')

print('='*60)
print('删除最后两列')
print('='*60)

print(f'\n当前数据形状: {df.shape[0]} 行, {df.shape[1]} 列')

# 获取所有列名
columns = df.columns.tolist()
print('\n当前所有变量:')
for i, col in enumerate(columns, 1):
    print(f'  {i:2d}. {col}')

# 获取最后两列
last_two_cols = columns[-2:]
print(f'\n要删除的最后两列:')
for i, col in enumerate(last_two_cols, 1):
    print(f'  {i}. {col}')

# 删除最后两列
df_new = df.drop(columns=last_two_cols)

print(f'\n删除后数据形状: {df_new.shape[0]} 行, {df_new.shape[1]} 列')
print(f'剩余变量数: {len(df_new.columns)}')

# 显示剩余的变量
print('\n剩余变量列表:')
for i, var in enumerate(df_new.columns, 1):
    print(f'  {i:2d}. {var}')

# 保存新表格（覆盖原文件）
output_file = 'd:\中风指标\wave2026.02.01_cleaned.xlsx'
df_new.to_excel(output_file, index=False)
print(f'\n更新后的表格已保存到: {output_file}')
print('='*60)
