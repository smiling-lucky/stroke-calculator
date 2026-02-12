import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os
import warnings
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# 确保输出目录存在
output_dir = 'SHAP结果分析'
os.makedirs(output_dir, exist_ok=True)

# 设置字体
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 加载数据和模型
print("加载数据与模型...")
data = pd.read_excel(r'd:\中风指标\elderly_hypertension_60plus_standardized.xlsx')
selected_features = ['Dyslipidemia', 'PHR', 'Uric_acid', 'MCV', 'Exercise', 'Age', 'AIP']
y = data['Stroke']
X = data[selected_features]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

model_path = r'd:\中风指标\gradient-boosting-no-isotonic-calibration.pkl'
model_pipeline = joblib.load(model_path)
base_model = model_pipeline.named_steps['model']

# 2. 计算 SHAP 值
print("计算 SHAP 值...")
explainer = shap.TreeExplainer(base_model)
shap_values = explainer.shap_values(X_test)

# 3. 准备绘图数据
# 计算平均绝对 SHAP 值用于排序和条形图
mean_abs_shap = np.abs(shap_values).mean(axis=0)
# 按重要性降序排列特征
idx = np.argsort(mean_abs_shap)
sorted_features = [selected_features[i] for i in idx]
sorted_mean_abs = mean_abs_shap[idx]
sorted_shap_values = shap_values[:, idx]
sorted_X = X_test.values[:, idx]

# 4. 绘制组合图
print("绘制组合 SHAP 图...")
fig, ax1 = plt.subplots(figsize=(12, 8))

# 恢复 SHAP 标准红蓝配色 (Blue to Red)
colors = ["#1E88E5", "#ff0052"] # 标准 SHAP 蓝 -> 红
n_bins = 100
cmap = LinearSegmentedColormap.from_list("shap_red_blue", colors, N=n_bins)

# 设置背景网格
ax1.grid(True, axis='x', linestyle='--', alpha=0.3, zorder=0)

# 绘制蜂群图 (Bee Swarm) - 使用散点模拟
y_pos = np.arange(len(sorted_features))
for i in range(len(sorted_features)):
    # 归一化特征值用于颜色映射
    feature_vals = sorted_X[:, i]
    if feature_vals.max() != feature_vals.min():
        norm_vals = (feature_vals - feature_vals.min()) / (feature_vals.max() - feature_vals.min())
    else:
        norm_vals = np.zeros_like(feature_vals)
    
    # 获取 SHAP 值
    shaps = sorted_shap_values[:, i]
    
    # 模拟蜂群布局 (在 Y 轴上添加随机扰动)
    jitter = np.random.normal(0, 0.1, size=len(shaps))
    
    # 绘制散点
    scatter = ax1.scatter(shaps, y_pos[i] + jitter, 
                         c=norm_vals, cmap=cmap, s=20, alpha=0.8, edgecolors='none', zorder=3)

# 绘制顶部的条形图 (Mean SHAP Value)
ax2 = ax1.twiny() # 创建共享 Y 轴的第二个 X 轴
bars = ax2.barh(y_pos, sorted_mean_abs, color='#e1f5fe', alpha=0.6, height=0.6, zorder=1)

# 设置轴标签和样式
ax1.set_yticks(y_pos)
ax1.set_yticklabels(sorted_features, fontsize=12)
ax1.set_xlabel('Shapley Value Contribution (Bee Swarm)', fontsize=13, labelpad=10)
ax2.set_xlabel('Mean Shapley Value (Feature Importance)', fontsize=13, labelpad=10)

# 设置轴范围
max_shap = np.max(np.abs(shap_values)) * 1.1
ax1.set_xlim(-max_shap, max_shap)
ax2.set_xlim(0, np.max(mean_abs_shap) * 1.5)

# 添加颜色条 (Feature Value)
cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
cbar = plt.colorbar(scatter, cax=cbar_ax)
cbar.set_label('Feature value', fontsize=12, labelpad=10)
cbar.set_ticks([0, 1])
cbar.set_ticklabels(['Low', 'High'])

# 调整布局
plt.subplots_adjust(left=0.15, right=0.9, top=0.85, bottom=0.1)
ax1.set_ylabel('Features', fontsize=13)

# 保存图片
save_path = f'{output_dir}/6_combined_shap_plot.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"组合 SHAP 分析图已保存至: {save_path}")
