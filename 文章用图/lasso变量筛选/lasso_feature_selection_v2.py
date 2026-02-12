import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score

# 读取标准化后的数据
df = pd.read_excel('d:\中风指标\elderly_hypertension_60plus_standardized.xlsx')

print('='*70)
print('LASSO特征筛选（alpha = 0.005）')
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

# 设置alpha = 0.005
alpha_value = 0.005
print(f'\n设置的alpha值: {alpha_value}')
print('\n' + '-'*70)
print('正在进行LASSO回归...')
print('-'*70)

# 使用Lasso（固定alpha）
lasso = Lasso(
    alpha=alpha_value,
    random_state=42,
    max_iter=5000
)

# 拟合模型
lasso.fit(X, y)

# 10折交叉验证评估
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_val_score(lasso, X, y, cv=skf, scoring='roc_auc')

print(f'\n模型评估:')
print(f'  10折交叉验证AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})')

# 获取特征系数
coefficients = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': lasso.coef_,
    'Abs_Coefficient': np.abs(lasso.coef_)
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
print(f"{'排名':<6}{'变量名':<20}{'系数':<15}{'绝对系数':<15}{'方向':<10}")
print('-'*70)

for idx, (_, row) in enumerate(selected_features.iterrows(), 1):
    direction = "↑ 正相关" if row['Coefficient'] > 0 else "↓ 负相关"
    print(f"{idx:<6}{row['Feature']:<20}{row['Coefficient']:<15.6f}{row['Abs_Coefficient']:<15.6f}{direction:<10}")

if len(unselected_features) > 0:
    print(f'\n未选中的变量 ({len(unselected_features)} 个):')
    for _, row in unselected_features.iterrows():
        print(f"  - {row['Feature']}")

# 计算整体AUC
y_pred_proba = lasso.predict(X)
auc = roc_auc_score(y, y_pred_proba)
print(f'\n整体AUC: {auc:.4f}')

# 保存结果
print('\n' + '='*70)
print('保存筛选结果')
print('='*70)

# 保存选中的特征
selected_features_file = 'd:\中风指标\lasso_selected_features_alpha005.xlsx'
selected_features.to_excel(selected_features_file, index=False)
print(f'\n选中的特征已保存到: {selected_features_file}')

# 保存所有特征的系数
all_coefficients_file = 'd:\中风指标\lasso_all_coefficients_alpha005.xlsx'
coefficients.to_excel(all_coefficients_file, index=False)
print(f'所有特征系数已保存到: {all_coefficients_file}')

# 创建包含选中特征的新数据集
selected_feature_names = selected_features['Feature'].tolist()
df_selected = df[['ID', 'Stroke'] + selected_feature_names]

selected_data_file = 'd:\中风指标\elderly_hypertension_selected_alpha005.xlsx'
df_selected.to_excel(selected_data_file, index=False)
print(f'选中特征的数据集已保存到: {selected_data_file}')

print(f'\n选中特征的数据集信息:')
print(f'  - 样本量: {len(df_selected)} 人')
print(f'  - 特征数: {len(selected_feature_names)} 个')
print(f'  - 特征列表: {selected_feature_names}')

# 与之前的结果对比
print('\n' + '='*70)
print('与之前结果对比（alpha=0.000587）')
print('='*70)
print(f'  之前alpha: 0.000587 → 选中20个变量')
print(f'  当前alpha: 0.005 → 选中{len(selected_features)}个变量')
print(f'  减少变量数: {20 - len(selected_features)}个')

print('\n' + '='*70)
