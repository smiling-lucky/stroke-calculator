import pandas as pd
import numpy as np

# 读取数据
df = pd.read_excel('d:\中风指标\wave2026.02.01_removed.xlsx')

print('='*80)
print('缺失值分析报告 - wave2026.02.01_removed.xlsx')
print('='*80)

print(f'\n数据形状: {df.shape[0]} 行, {df.shape[1]} 列')
print(f'总记录数: {len(df)}')

# 计算每个变量的缺失值情况
missing_stats = []

for col in df.columns:
    missing_count = df[col].isna().sum()
    missing_percent = (missing_count / len(df)) * 100
    missing_stats.append({
        '变量名': col,
        '缺失数量': missing_count,
        '缺失比例(%)': round(missing_percent, 2)
    })

# 转换为DataFrame并排序
missing_df = pd.DataFrame(missing_stats)
missing_df = missing_df.sort_values('缺失比例(%)', ascending=False)

print('\n' + '-'*80)
print('各变量缺失值统计（按缺失比例降序排列）')
print('-'*80)

# 打印表头
print(f"{'序号':<6}{'变量名':<25}{'缺失数量':<15}{'缺失比例(%)':<15}")
print('-'*80)

# 打印数据
for idx, row in missing_df.iterrows():
    print(f"{idx+1:<6}{row['变量名']:<25}{row['缺失数量']:<15}{row['缺失比例(%)']:<15}")

# 统计摘要
print('\n' + '='*80)
print('缺失值统计摘要')
print('='*80)

# 完全无缺失的变量
no_missing = missing_df[missing_df['缺失数量'] == 0]
print(f"\n完全无缺失的变量: {len(no_missing)} 个")
if len(no_missing) > 0:
    print(f"  变量列表: {', '.join(no_missing['变量名'].tolist())}")

# 有缺失的变量
has_missing = missing_df[missing_df['缺失数量'] > 0]
print(f"\n存在缺失值的变量: {len(has_missing)} 个")

if len(has_missing) > 0:
    print(f"\n缺失值分布:")
    print(f"  缺失比例 < 5% 的变量: {len(has_missing[has_missing['缺失比例(%)'] < 5])} 个")
    print(f"  缺失比例 5%-20% 的变量: {len(has_missing[(has_missing['缺失比例(%)'] >= 5) & (has_missing['缺失比例(%)'] < 20)])} 个")
    print(f"  缺失比例 >= 20% 的变量: {len(has_missing[has_missing['缺失比例(%)'] >= 20])} 个")
    
    # 高缺失比例变量详情
    high_missing = has_missing[has_missing['缺失比例(%)'] >= 20]
    if len(high_missing) > 0:
        print(f"\n高缺失比例变量详情 (>=20%):")
        for idx, row in high_missing.iterrows():
            print(f"  - {row['变量名']}: {row['缺失数量']} ({row['缺失比例(%)']}%)")

# 总体缺失情况
total_cells = df.shape[0] * df.shape[1]
total_missing = df.isna().sum().sum()
overall_missing_percent = (total_missing / total_cells) * 100

print(f"\n总体缺失情况:")
print(f"  总数据单元格数: {total_cells:,}")
print(f"  总缺失单元格数: {total_missing:,}")
print(f"  总体缺失比例: {overall_missing_percent:.2f}%")

print('\n' + '='*80)
