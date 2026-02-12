import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report
import matplotlib.pyplot as plt

# 读取标准化后的数据
df = pd.read_excel('d:\中风指标\elderly_hypertension_60plus_standardized.xlsx')

print('='*70)
print('LASSO特征筛选（10折交叉验证）')
print('='*70)

print(f'\n数据形状: {df.shape[0]} 行, {df.shape[1]} 列')

# 准备数据
X = df.drop(columns=['ID', 'Stroke'])  # 自变量（排除ID和Stroke）
y = df['Stroke']  # 因变量

print(f'\n自变量数量: {X.shape[1]} 个')
print(f'因变量: Stroke')
print(f'样本量: {len(y)}')
print(f'中风患病率: {y.mean()*100:.2f}%')

# 特征名称
feature_names = X.columns.tolist()
print(f'\n自变量列表:')
for i, feat in enumerate(feature_names, 1):
    print(f'  {i:2d}. {feat}')

# LASSO with 10-fold cross-validation
print('\n' + '-'*70)
print('正在进行LASSO回归（10折交叉验证）...')
print('-'*70)

# 使用StratifiedKFold保持类别比例
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# LASSO CV
lasso_cv = LassoCV(
    cv=skf,
    random_state=42,
    max_iter=2000,
    n_alphas=100,
    verbose=0
)

# 拟合模型
lasso_cv.fit(X, y)

print(f'\n最优alpha (正则化强度): {lasso_cv.alpha_:.6f}')
print(f'交叉验证最佳得分 (R²): {lasso_cv.score(X, y):.4f}')

# 获取特征系数
coefficients = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': lasso_cv.coef_,
    'Abs_Coefficient': np.abs(lasso_cv.coef_)
})

# 按绝对系数排序
coefficients = coefficients.sort_values('Abs_Coefficient', ascending=False)

print('\n' + '='*70)
print('特征筛选结果（按系数绝对值降序）')
print('='*70)

# 分离选中和未选中的特征
selected_features = coefficients[coefficients['Coefficient'] != 0]
unselected_features = coefficients[coefficients['Coefficient'] == 0]

print(f'\n选中的关键变量 ({len(selected_features)} 个):')
print(f"{'排名':<6}{'变量名':<20}{'系数':<15}{'绝对系数':<15}")
print('-'*70)
for idx, (_, row) in enumerate(selected_features.iterrows(), 1):
    print(f"{idx:<6}{row['Feature']:<20}{row['Coefficient']:<15.6f}{row['Abs_Coefficient']:<15.6f}")

if len(unselected_features) > 0:
    print(f'\n未选中的变量 ({len(unselected_features)} 个):')
    for _, row in unselected_features.iterrows():
        print(f"  - {row['Feature']}")

# 计算AUC
y_pred_proba = lasso_cv.predict(X)
auc = roc_auc_score(y, y_pred_proba)
print(f'\n模型性能评估:')
print(f'  AUC: {auc:.4f}')

# 保存结果
print('\n' + '='*70)
print('保存筛选结果')
print('='*70)

# 保存选中的特征
selected_features_file = 'd:\中风指标\lasso_selected_features.xlsx'
selected_features.to_excel(selected_features_file, index=False)
print(f'\n选中的特征已保存到: {selected_features_file}')

# 保存所有特征的系数
all_coefficients_file = 'd:\中风指标\lasso_all_coefficients.xlsx'
coefficients.to_excel(all_coefficients_file, index=False)
print(f'所有特征系数已保存到: {all_coefficients_file}')

# 创建包含选中特征的新数据集
selected_feature_names = selected_features['Feature'].tolist()
df_selected = df[['ID', 'Stroke'] + selected_feature_names]

selected_data_file = 'd:\中风指标\elderly_hypertension_selected_features.xlsx'
df_selected.to_excel(selected_data_file, index=False)
print(f'选中特征的数据集已保存到: {selected_data_file}')

print(f'\n选中特征的数据集信息:')
print(f'  - 样本量: {len(df_selected)} 人')
print(f'  - 特征数: {len(selected_feature_names)} 个')
print(f'  - 特征列表: {selected_feature_names}')

print('\n' + '='*70)
