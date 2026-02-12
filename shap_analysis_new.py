import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.model_selection import train_test_split
import os
import warnings
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable

warnings.filterwarnings('ignore')

# 确保输出目录存在
output_dir = 'SHAP结果分析'
os.makedirs(output_dir, exist_ok=True)

# 设置字体支持
# 使用 Arial 或 DejaVu Sans 可以更好地支持 SHAP 的数学符号，避免出现小方框
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'SimHei'] # 优先使用 Arial
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示为方块的问题

# 1. 加载数据
print("加载数据...")
data = pd.read_excel(r'd:\中风指标\elderly_hypertension_60plus_standardized.xlsx')
selected_features = ['Dyslipidemia', 'PHR', 'Uric_acid', 'MCV', 'Exercise', 'Age', 'AIP']
y = data['Stroke']
X = data[selected_features]

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 2. 加载模型
print("加载模型...")
model_path = r'd:\中风指标\gradient-boosting-no-isotonic-calibration.pkl'
model_pipeline = joblib.load(model_path)
# 提取基础模型（GradientBoostingClassifier）
# 根据之前的 Read 结果，模型是一个 Pipeline，其中包含 'smote' 和 'model'
base_model = model_pipeline.named_steps['model']

# 3. 计算 SHAP 值
print("计算 SHAP 值...")
explainer = shap.TreeExplainer(base_model)
# 使用测试集进行分析
shap_values = explainer.shap_values(X_test)

# 4. 绘制摘要图 (Summary Plot)
print("生成摘要图...")
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, show=False)
plt.title('SHAP Summary Plot', fontsize=15, pad=20)
plt.tight_layout()
plt.savefig(f'{output_dir}/1_summary_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. 绘制特征重要性图 (Feature Importance)
print("生成特征重要性图...")
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title('SHAP Feature Importance', fontsize=15, pad=20)
plt.tight_layout()
plt.savefig(f'{output_dir}/2_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. 绘制决策路径图 (Decision Plot - 20个样本)
print("生成决策路径图 (20个样本)...")
plt.figure(figsize=(10, 8))
# 随机选择20个样本
sample_indices = np.random.choice(len(X_test), 20, replace=False)
shap.decision_plot(explainer.expected_value, shap_values[sample_indices], 
                   X_test.iloc[sample_indices], show=False, link='logit')
plt.title('SHAP Decision Plot (20 Samples)', fontsize=15, pad=20)
plt.tight_layout()
plt.savefig(f'{output_dir}/3_decision_plot_20_samples.png', dpi=300, bbox_inches='tight')
plt.close()

# 7. 绘制瀑布图 (Waterfall Plot - 2张)
print("生成瀑布图 (2张)...")
for i in range(2):
    plt.figure(figsize=(10, 8))
    # 创建 Explanation 对象
    exp = shap.Explanation(
        values=shap_values[i], 
        base_values=explainer.expected_value, 
        data=X_test.iloc[i], 
        feature_names=selected_features
    )
    shap.waterfall_plot(exp, show=False)
    plt.title(f'SHAP Waterfall Plot - Sample {i+1}', fontsize=15, pad=20)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/4_waterfall_plot_sample_{i+1}.png', dpi=300, bbox_inches='tight')
    plt.close()

# 8. 绘制特征交互网络图 (Feature Interaction Network)
print("生成特征交互网络图...")
# 计算交互值 (可能较慢，限制样本数以加快速度，如100个)
interaction_samples = min(100, len(X_test))
shap_interaction_values = explainer.shap_interaction_values(X_test.iloc[:interaction_samples])

def plot_interaction_network(interaction_values, feature_names, save_path):
    # 计算平均绝对交互值
    mean_interaction = np.abs(interaction_values).mean(axis=0)
    
    # 主效应 (对角线)
    main_effects = np.diag(mean_interaction)
    # 交互效应 (非对角线)
    # 交互矩阵是对称的，取上三角部分
    
    features = feature_names
    n_features = len(features)
    
    G = nx.Graph()
    for i in range(n_features):
        G.add_node(features[i], size=main_effects[i])
        for j in range(i + 1, n_features):
            inter_val = mean_interaction[i, j] * 2 # 总交互强度
            G.add_edge(features[i], features[j], weight=inter_val)

    # 布局
    pos = nx.circular_layout(G)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 准备颜色映射
    # ROC 图中 Gradient Boosting 使用的是 #9467bd (紫色)
    # 我们可以使用紫色系和蓝色系的组合来使配色协调
    # 主效应使用蓝色系 (匹配 Random Forest 的 #1f77b4)
    # 交互效应使用紫色系 (匹配 Gradient Boosting 的 #9467bd)
    
    # 创建自定义渐变色
    cmap_main = LinearSegmentedColormap.from_list('main_blue', ['#e1f5fe', '#1f77b4', '#0d47a1'])
    cmap_inter = LinearSegmentedColormap.from_list('inter_purple', ['#f3e5f5', '#9467bd', '#4a148c'])
    
    # 归一化用于颜色
    main_norm = plt.Normalize(vmin=main_effects.min(), vmax=main_effects.max())
    
    # 提取边和权重
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    inter_norm = plt.Normalize(vmin=min(weights), vmax=max(weights))
    
    # 绘制边
    for (u, v), w in zip(edges, weights):
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)],
            width=w * 50, # 调整线宽系数
            edge_color=[cmap_inter(inter_norm(w))],
            ax=ax, alpha=0.7
        )
    
    # 绘制节点
    for i, node in enumerate(G.nodes()):
        val = main_effects[i]
        nx.draw_networkx_nodes(
            G, pos, nodelist=[node],
            node_size=val * 5000, # 调整节点大小系数
            node_color=[cmap_main(main_norm(val))],
            ax=ax, edgecolors='#1f77b4', linewidths=1.5
        )
    
    # 绘制标签
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif', ax=ax)
    
    # 添加颜色条
    # 主效应颜色条 (右侧)
    sm_main = ScalarMappable(norm=main_norm, cmap=cmap_main)
    cbar_main = fig.colorbar(sm_main, ax=ax, fraction=0.046, pad=0.04)
    cbar_main.set_label('Main Effect Strength', fontsize=12)
    
    # 交互效应颜色条 (左侧)
    sm_inter = ScalarMappable(norm=inter_norm, cmap=cmap_inter)
    cbar_inter = fig.colorbar(sm_inter, ax=ax, fraction=0.046, pad=0.04, location='left')
    cbar_inter.set_label('Interaction Strength', fontsize=12)
    
    plt.title('Feature Interaction Network', fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

plot_interaction_network(shap_interaction_values, selected_features, f'{output_dir}/5_feature_interaction_network.png')

print(f"\n所有 SHAP 分析图已保存至文件夹: {output_dir}")
