import pandas as pd
import numpy as np

# 读取数据
df = pd.read_excel('d:\中风指标\wave2026.02.01_removed.xlsx')

print('='*80)
print('60岁以上且患高血压人群的缺失值分析')
print('='*80)

# 查看基本条件的数据情况
print(f'\n原始数据形状: {df.shape[0]} 行, {df.shape[1]} 列')
print(f'\nAge字段范围: {df["Age"].min():.1f} ~ {df["Age"].max():.1f}')
print(f'Hypertension字段唯一值: {sorted(df["Hypertension"].dropna().unique())}')

# 筛选条件: Age >= 60 且 Hypertension = 1
df_elderly_ht = df[(df['Age'] >= 60) & (df['Hypertension'] == 1)].copy()

print(f'\n筛选条件: Age >= 60 且 Hypertension = 1')
print(f'筛选后数据形状: {df_elderly_ht.shape[0]} 行, {df_elderly_ht.shape[1]} 列')
print(f'60岁以上高血压人群数量: {len(df_elderly_ht)}')

if len(df_elderly_ht) == 0:
    print('\n警告: 未找到符合条件的记录！')
else:
    # 计算每个变量的缺失值情况
    missing_stats = []
    
    for col in df_elderly_ht.columns:
        missing_count = df_elderly_ht[col].isna().sum()
        missing_percent = (missing_count / len(df_elderly_ht)) * 100
        missing_stats.append({
            '变量名': col,
            '缺失数量': missing_count,
            '缺失比例(%)': round(missing_percent, 2)
        })
    
    # 转换为DataFrame并排序
    missing_df = pd.DataFrame(missing_stats)
    missing_df = missing_df.sort_values('缺失比例(%)', ascending=False)
    
    print('\n' + '-'*80)
    print('60岁以上高血压人群各变量缺失值统计（按缺失比例降序排列）')
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
    total_cells = df_elderly_ht.shape[0] * df_elderly_ht.shape[1]
    total_missing = df_elderly_ht.isna().sum().sum()
    overall_missing_percent = (total_missing / total_cells) * 100
    
    print(f"\n总体缺失情况:")
    print(f"  总数据单元格数: {total_cells:,}")
    print(f"  总缺失单元格数: {total_missing:,}")
    print(f"  总体缺失比例: {overall_missing_percent:.2f}%")
    
    # 与整体数据、所有高血压人群的对比
    print('\n' + '='*80)
    print('与整体数据、所有高血压人群的缺失情况对比')
    print('='*80)
    
    # 计算所有高血压人群的缺失情况
    df_ht = df[df['Hypertension'] == 1].copy()
    
    # 计算整体数据的缺失情况
    overall_stats = []
    ht_stats = []
    elderly_ht_stats = []
    
    for col in df.columns:
        # 整体数据
        missing_count = df[col].isna().sum()
        missing_percent = (missing_count / len(df)) * 100
        overall_stats.append({'变量名': col, '整体缺失%': round(missing_percent, 2)})
        
        # 所有高血压人群
        missing_count = df_ht[col].isna().sum()
        missing_percent = (missing_count / len(df_ht)) * 100
        ht_stats.append({'变量名': col, '高血压缺失%': round(missing_percent, 2)})
        
        # 60岁以上高血压人群
        missing_count = df_elderly_ht[col].isna().sum()
        missing_percent = (missing_count / len(df_elderly_ht)) * 100
        elderly_ht_stats.append({'变量名': col, '老年高血压缺失%': round(missing_percent, 2)})
    
    overall_df = pd.DataFrame(overall_stats)
    ht_df = pd.DataFrame(ht_stats)
    elderly_ht_df = pd.DataFrame(elderly_ht_stats)
    
    # 合并对比
    comparison = missing_df.merge(overall_df, on='变量名')
    comparison = comparison.merge(ht_df, on='变量名')
    comparison = comparison.merge(elderly_ht_df, on='变量名')
    
    print(f"\n{'变量名':<22}{'老年高血压%':<12}{'所有高血压%':<12}{'整体%':<10}{'差异(vs整体)':<12}")
    print('-'*80)
    for idx, row in comparison.iterrows():
        diff = row['老年高血压缺失%'] - row['整体缺失%']
        diff_str = f"{diff:+.2f}"
        print(f"{row['变量名']:<22}{row['老年高血压缺失%']:<12}{row['高血压缺失%']:<12}{row['整体缺失%']:<10}{diff_str:<12}")

print('\n' + '='*80)
