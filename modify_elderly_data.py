import pandas as pd

# 读取数据
df = pd.read_excel('d:\中风指标\elderly_hypertension_60plus.xlsx')

print('='*60)
print('修改60岁以上高血压人群数据集')
print('='*60)

print(f'\n当前数据形状: {df.shape[0]} 行, {df.shape[1]} 列')
print(f'当前变量数: {len(df.columns)}')

# 1. 重命名Hemoglobin为Hb
print('\n1. 重命名变量:')
print('   Hemoglobin -> Hb')
df = df.rename(columns={'Hemoglobin': 'Hb'})

# 2. 删除Kidneye和Wave列
vars_to_delete = ['Kidneye', 'Wave']
print(f'\n2. 删除变量:')
for var in vars_to_delete:
    if var in df.columns:
        print(f'   {var} (已删除)')
    else:
        print(f'   {var} (不存在)')

df = df.drop(columns=[var for var in vars_to_delete if var in df.columns])

print(f'\n修改后数据形状: {df.shape[0]} 行, {df.shape[1]} 列')
print(f'剩余变量数: {len(df.columns)}')

# 显示剩余的变量
print('\n剩余变量列表:')
for i, var in enumerate(df.columns, 1):
    print(f'  {i:2d}. {var}')

# 保存新表格（覆盖原文件）
output_file = 'd:\中风指标\elderly_hypertension_60plus.xlsx'
df.to_excel(output_file, index=False)
print(f'\n更新后的表格已保存到: {output_file}')
print('='*60)
