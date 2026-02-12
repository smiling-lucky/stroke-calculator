import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

# 读取数据
df = pd.read_excel('d:\中风指标\elderly_hypertension_60plus.xlsx')

print('='*70)
print('多重插补处理（修复版）')
print('='*70)

print(f'\n原始数据形状: {df.shape[0]} 行, {df.shape[1]} 列')

# 查看缺失值情况
print('\n插补前缺失值统计:')
missing_before = df.isnull().sum()
missing_cols = missing_before[missing_before > 0]
if len(missing_cols) > 0:
    for col, count in missing_cols.items():
        print(f'  {col}: {count} ({count/len(df)*100:.2f}%)')
else:
    print('  无缺失值')

# 分离ID列（不需要插补）
id_col = df['ID'].copy()
df_features = df.drop(columns=['ID'])

# 识别数值型变量
numeric_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
print(f'\n需要插补的数值型变量数: {len(numeric_cols)}')

# 定义二分类变量（只能是0或1）
binary_vars = ['Gender', 'Marry', 'Hearte', 'Stroke', 'Dyslipidemia', 'Livere', 
               'Alcohol_consumption', 'Smoken', 'Hypertension', 'Diabetes', 'Exercise']

# 定义连续变量
continuous_vars = [col for col in numeric_cols if col not in binary_vars]

print(f'二分类变量数: {len(binary_vars)}')
print(f'连续变量数: {len(continuous_vars)}')

# 使用IterativeImputer进行多重插补
print('\n正在进行多重插补...')
print('  使用IterativeImputer (基于RandomForest)')

# 创建插补器
imputer = IterativeImputer(
    estimator=RandomForestRegressor(n_estimators=10, random_state=42, n_jobs=-1),
    max_iter=10,
    random_state=42,
    verbose=0
)

# 执行插补
df_imputed_array = imputer.fit_transform(df_features[numeric_cols])

# 转换回DataFrame
df_imputed = pd.DataFrame(df_imputed_array, columns=numeric_cols, index=df_features.index)

# 对二分类变量进行四舍五入并限制在0-1之间
print('\n处理二分类变量（确保只为0或1）:')
for var in binary_vars:
    if var in df_imputed.columns:
        # 四舍五入到最近的整数
        df_imputed[var] = df_imputed[var].round()
        # 限制在0-1范围内
        df_imputed[var] = df_imputed[var].clip(0, 1)
        # 转换为整数类型
        df_imputed[var] = df_imputed[var].astype(int)
        print(f'  {var}: 唯一值 = {sorted(df_imputed[var].unique())}')

# 将ID列加回来
df_final = pd.concat([id_col, df_imputed], axis=1)

# 确保列顺序与原始数据一致
df_final = df_final[df.columns]

print('\n插补完成！')

# 验证插补结果
print('\n插补后缺失值统计:')
missing_after = df_final.isnull().sum()
missing_cols_after = missing_after[missing_after > 0]
if len(missing_cols_after) > 0:
    for col, count in missing_cols_after.items():
        print(f'  {col}: {count}')
else:
    print('  无缺失值 ✓')

# 验证二分类变量
print('\n二分类变量验证（唯一值）:')
for var in binary_vars:
    if var in df_final.columns:
        unique_vals = sorted(df_final[var].unique())
        status = "✓" if unique_vals == [0, 1] else "⚠"
        print(f'  {status} {var}: {unique_vals}')

# 显示部分插补前后的对比
print('\n部分变量插补前后对比（均值）:')
print(f"{'变量名':<20}{'插补前均值':<15}{'插补后均值':<15}{'变化':<10}")
print('-'*60)
for col in missing_cols.index[:10]:
    before_mean = df[col].mean()
    after_mean = df_final[col].mean()
    change = after_mean - before_mean
    print(f"{col:<20}{before_mean:<15.3f}{after_mean:<15.3f}{change:<+10.3f}")

# 保存插补后的数据集
output_file = 'd:\中风指标\elderly_hypertension_60plus_imputed.xlsx'
df_final.to_excel(output_file, index=False)

print(f'\n插补后的数据集已保存到:')
print(f'  {output_file}')
print(f'\n数据集信息:')
print(f'  - 样本量: {len(df_final)} 人')
print(f'  - 变量数: {len(df_final.columns)} 个')
print(f'  - 缺失值: {df_final.isnull().sum().sum()} 个')

print('\n' + '='*70)
