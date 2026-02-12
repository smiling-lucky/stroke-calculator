import pandas as pd
import os

data_path = r'd:\中风指标\验证集_encoded_20260211.xlsx'
df = pd.read_excel(data_path)
selected_features = ['Dyslipidemia', 'PHR', 'Uric_acid', 'MCV', 'Exercise', 'Age', 'AIP', 'Stroke']

print(f"检查数据集: {data_path}")
print("-" * 30)
missing_counts = df[selected_features].isnull().sum()
print("各列缺失值数量:")
print(missing_counts)

total_nan_rows = df[selected_features].isnull().any(axis=1).sum()
print("-" * 30)
print(f"包含缺失值的总行数: {total_nan_rows}")
print(f"数据集总行数: {len(df)}")
