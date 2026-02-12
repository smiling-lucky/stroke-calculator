import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_excel('插补后01_已标准化.xlsx')

# 查看数据基本信息
print("数据集形状:", data.shape)
print("\n前5行数据:")
print(data.head())

print("\n列名:")
print(data.columns.tolist())

print("\n数据类型:")
print(data.dtypes)

print("\n缺失值统计:")
print(data.isnull().sum())

# 检查Stones_binary列是否存在
if 'Stones_binary' in data.columns:
    print("\nStones_binary分布:")
    print(data['Stones_binary'].value_counts())
else:
    print("\n错误: Stones_binary列不存在")
    print("可用列:", data.columns.tolist())

# 准备数据：Stones_binary作为因变量，其余变量作为自变量
y = data['Stones_binary']
X = data.drop('Stones_binary', axis=1)

print(f"\n自变量形状: {X.shape}")
print(f"因变量形状: {y.shape}")

# 定义LASSO回归模型，使用10折交叉验证
# 设置一系列alpha值进行测试
alphas = np.logspace(-4, 1, 100)
lasso_cv = LassoCV(alphas=alphas, cv=10, max_iter=10000, random_state=42)

# 拟合模型
print("\n开始拟合LASSO模型...")
lasso_cv.fit(X, y)

print(f"最优alpha值: {lasso_cv.alpha_}")
print(f"交叉验证平均得分: {lasso_cv.score(X, y)}")

# 获取系数
coefficients = lasso_cv.coef_

# 筛选非零系数的变量
selected_features = X.columns[coefficients != 0]
selected_coefficients = coefficients[coefficients != 0]

print(f"\n筛选出的变量数量: {len(selected_features)}")
print("\n筛选出的变量及其系数:")
for feature, coef in zip(selected_features, selected_coefficients):
    print(f"{feature}: {coef:.6f}")

# 绘制系数路径图
plt.figure(figsize=(12, 8))
plt.semilogx(lasso_cv.alphas_, lasso_cv.mse_path_, ':')  
plt.semilogx(lasso_cv.alphas_, lasso_cv.mse_path_.mean(axis=-1), 'k',
             label='平均MSE', linewidth=2)
plt.axvline(lasso_cv.alpha_, linestyle='--', color='k',
            label='最优alpha')
plt.xlabel('Alpha')
plt.ylabel('均方误差')
plt.title('LASSO系数路径图 (10折交叉验证)')
plt.legend()
plt.grid(True)
plt.savefig('基于lasso/lasso_coefficient_path.png', dpi=300, bbox_inches='tight')
plt.show()

# 保存筛选结果到文件
results_df = pd.DataFrame({
    'Variable': selected_features,
    'Coefficient': selected_coefficients,
    'Absolute_Coefficient': np.abs(selected_coefficients)
})
results_df = results_df.sort_values('Absolute_Coefficient', ascending=False)

results_df.to_csv('基于lasso/lasso_selected_features.csv', index=False, encoding='utf-8-sig')
print("\n筛选结果已保存到 '基于lasso/lasso_selected_features.csv'")

# 保存详细的LASSO分析结果
lasso_results = {
    'optimal_alpha': lasso_cv.alpha_,
    'cv_score': lasso_cv.score(X, y),
    'selected_features_count': len(selected_features),
    'selected_features': selected_features.tolist(),
    'coefficients': coefficients.tolist(),
    'non_zero_coefficients': selected_coefficients.tolist()
}

# 保存为文本报告
with open('基于lasso/lasso_analysis_report.txt', 'w', encoding='utf-8') as f:
    f.write("LASSO变量筛选分析报告\n")
    f.write("="*50 + "\n\n")
    f.write(f"数据集信息:\n")
    f.write(f"- 总样本数: {data.shape[0]}\n")
    f.write(f"- 总变量数: {data.shape[1]}\n")
    f.write(f"- 因变量: Stones_binary\n")
    f.write(f"- Stones_binary分布: 0={data['Stones_binary'].value_counts()[0]}, 1={data['Stones_binary'].value_counts()[1]}\n\n")
    
    f.write(f"LASSO模型参数:\n")
    f.write(f"- 最优alpha值: {lasso_cv.alpha_}\n")
    f.write(f"- 交叉验证得分: {lasso_cv.score(X, y):.4f}\n")
    f.write(f"- 筛选出的变量数量: {len(selected_features)}\n\n")
    
    f.write("筛选出的变量及其系数:\n")
    f.write("-"*40 + "\n")
    for feature, coef in zip(selected_features, selected_coefficients):
        f.write(f"{feature}: {coef:.6f}\n")
    
    f.write(f"\n所有变量系数:\n")
    f.write("-"*40 + "\n")
    for feature, coef in zip(X.columns, coefficients):
        f.write(f"{feature}: {coef:.6f}\n")

print(f"详细分析报告已保存到 '基于lasso/lasso_analysis_report.txt'")