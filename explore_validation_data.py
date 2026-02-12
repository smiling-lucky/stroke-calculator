import pandas as pd
import numpy as np

print("="*70)
print("验证集数据探索 - 分类变量与连续变量识别")
print("="*70)

# 读取验证集数据
file_path = r'd:\中风指标\验证集_clean_20260211.xlsx'
df = pd.read_excel(file_path)

print(f"\n数据集基本信息:")
print(f"数据形状: {df.shape}")
print(f"总行数: {df.shape[0]}")
print(f"总列数: {df.shape[1]}")

print(f"\n列名列表:")
for i, col in enumerate(df.columns, 1):
    print(f"{i:2d}. {col}")

print(f"\n数据类型:")
print(df.dtypes)

print(f"\n前几行数据预览:")
print(df.head())

print(f"\n数据统计摘要:")
print(df.describe(include='all'))

print(f"\n" + "="*70)
print("变量类型分类")
print("="*70)

# 识别分类变量和连续变量
categorical_vars = []
continuous_vars = []

for col in df.columns:
    unique_vals = df[col].nunique()
    dtype = df[col].dtype
    
    # 如果是数值类型但唯一值较少，可能是分类变量
    if pd.api.types.is_numeric_dtype(dtype):
        if unique_vals <= 10:  # 设定阈值，可根据实际情况调整
            categorical_vars.append(col)
            print(f"CATEGORICAL ({unique_vals} unique): {col}")
        else:
            continuous_vars.append(col)
            print(f"CONTINUOUS ({unique_vals} unique): {col}")
    else:
        # 非数值类型通常为分类变量
        categorical_vars.append(col)
        print(f"CATEGORICAL ({unique_vals} unique): {col}")

print(f"\n" + "="*50)
print("总结")
print("="*50)
print(f"分类变量 ({len(categorical_vars)} 个):")
for var in categorical_vars:
    unique_vals = df[var].nunique()
    print(f"  - {var} (唯一值: {unique_vals})")

print(f"\n连续变量 ({len(continuous_vars)} 个):")
for var in continuous_vars:
    unique_vals = df[var].nunique()
    print(f"  - {var} (唯一值: {unique_vals})")

print(f"\n" + "="*70)
print("分类变量详细信息")
print("="*70)
for var in categorical_vars:
    print(f"\n{var}:")
    print(f"  数据类型: {df[var].dtype}")
    print(f"  唯一值数量: {df[var].nunique()}")
    print(f"  唯一值: {sorted(df[var].dropna().unique())}")
    print(f"  缺失值: {df[var].isnull().sum()}")

print(f"\n" + "="*70)
print("连续变量详细信息")
print("="*70)
for var in continuous_vars:
    print(f"\n{var}:")
    print(f"  数据类型: {df[var].dtype}")
    print(f"  唯一值数量: {df[var].nunique()}")
    print(f"  统计摘要:")
    print(f"    最小值: {df[var].min()}")
    print(f"    最大值: {df[var].max()}")
    print(f"    平均值: {df[var].mean():.3f}")
    print(f"    标准差: {df[var].std():.3f}")
    print(f"    缺失值: {df[var].isnull().sum()}")