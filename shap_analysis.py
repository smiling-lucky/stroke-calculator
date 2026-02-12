import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import warnings
warnings.filterwarnings('ignore')

output_dir = 'SHAP分析'
os.makedirs(output_dir, exist_ok=True)

data = pd.read_excel(r'd:\中风指标\elderly_hypertension_60plus_standardized.xlsx')

selected_features = ['Dyslipidemia', 'PHR', 'Uric_acid', 'MCV', 'Exercise', 'Age', 'AIP']

y = data['Stroke']
X = data[selected_features]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print('='*70)
print('SHAP分析 - Gradient Boosting模型（无Isotonic校准）')
print('='*70)
print(f"\n特征数量: {len(selected_features)}")
print(f"特征列表: {selected_features}")
print(f"X形状: {X.shape}")
print(f"y形状: {y.shape}")
print(f"中风患病率: {y.sum()/len(y)*100:.2f}%")
print(f"\n训练集大小: {X_train.shape[0]}")
print(f"测试集大小: {X_test.shape[0]}")

print("\n" + "="*70)
print("加载模型...")
print("="*70)

model = joblib.load(r'd:\中风指标\gradient-boosting-no-isotonic-calibration.pkl')
print("模型加载成功")

print("\n" + "="*70)
print("创建SHAP解释器...")
print("="*70)

base_model = model.named_steps['model']
print(f"基础模型类型: {type(base_model).__name__}")

explainer = shap.TreeExplainer(base_model)
print("SHAP TreeExplainer创建成功")

print("\n" + "="*70)
print("计算SHAP值...")
print("="*70)

X_train_processed = model.named_steps['smote'].fit_resample(X_train, y_train)[0]
X_test_processed = X_test

shap_values = explainer.shap_values(X_test_processed)
print(f"SHAP值计算完成")
print(f"SHAP值形状: {shap_values.shape}")

print("\n" + "="*70)
print("绘制SHAP摘要图...")
print("="*70)

plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test_processed, feature_names=selected_features, 
                 plot_type="dot", show=False)
plt.title('SHAP Summary Plot - Gradient Boosting Model', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(f'{output_dir}/shap_summary_plot.png', dpi=300, bbox_inches='tight')
plt.show()
print(f"摘要图已保存到: {output_dir}/shap_summary_plot.png")

plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test_processed, feature_names=selected_features, 
                 plot_type="bar", show=False)
plt.title('SHAP Feature Importance (Bar Plot)', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(f'{output_dir}/shap_feature_importance_bar.png', dpi=300, bbox_inches='tight')
plt.show()
print(f"特征重要性图（条形）已保存到: {output_dir}/shap_feature_importance_bar.png")

print("\n" + "="*70)
print("绘制SHAP特征重要性图...")
print("="*70)

mean_shap = np.abs(shap_values).mean(axis=0)
feature_importance_df = pd.DataFrame({
    'Feature': selected_features,
    'Mean |SHAP|': mean_shap
}).sort_values('Mean |SHAP|', ascending=False)

print("\n特征重要性排名:")
print(feature_importance_df.to_string(index=False))

plt.figure(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0, 1, len(selected_features)))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Mean |SHAP|'], color=colors)
plt.xlabel('Mean |SHAP Value|', fontsize=12, fontweight='bold')
plt.ylabel('Feature', fontsize=12, fontweight='bold')
plt.title('SHAP Feature Importance - Gradient Boosting Model', fontsize=16, fontweight='bold', pad=20)
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(f'{output_dir}/shap_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()
print(f"特征重要性图已保存到: {output_dir}/shap_feature_importance.png")

feature_importance_df.to_csv(f'{output_dir}/shap_feature_importance.csv', index=False, encoding='utf-8-sig')
print(f"特征重要性数据已保存到: {output_dir}/shap_feature_importance.csv")

print("\n" + "="*70)
print("绘制SHAP决策路径图（20条路径）...")
print("="*70)

sample_indices = np.random.choice(len(X_test_processed), min(20, len(X_test_processed)), replace=False)

for i, idx in enumerate(sample_indices):
    plt.figure(figsize=(10, 6))
    shap.decision_plot(explainer.expected_value, shap_values[idx], 
                     X_test_processed.iloc[idx], feature_names=selected_features,
                     show=False, link='logit')
    plt.title(f'SHAP Decision Path - Sample {i+1} (Index: {idx})', 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/shap_decision_path_{i+1}.png', dpi=300, bbox_inches='tight')
    plt.close()

print(f"决策路径图已保存到: {output_dir}/shap_decision_path_1-20.png")

print("\n" + "="*70)
print("绘制SHAP特征交互图...")
print("="*70)

shap_interaction_values = explainer.shap_interaction_values(X_test_processed[:100])
print(f"交互值计算完成，形状: {shap_interaction_values.shape}")

plt.figure(figsize=(12, 10))
shap.summary_plot(shap_interaction_values, X_test_processed[:100], 
                 feature_names=selected_features, plot_type="compact_dot", show=False)
plt.title('SHAP Interaction Summary', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(f'{output_dir}/shap_interaction_summary.png', dpi=300, bbox_inches='tight')
plt.show()
print(f"特征交互摘要图已保存到: {output_dir}/shap_interaction_summary.png")

plt.figure(figsize=(12, 10))
shap.dependence_plot(
    (selected_features[0], selected_features[1]), 
    shap_interaction_values, 
    X_test_processed[:100], 
    feature_names=selected_features,
    show=False
)
plt.title(f'SHAP Dependence Plot: {selected_features[0]} vs {selected_features[1]}', 
          fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(f'{output_dir}/shap_dependence_plot_{selected_features[0]}_{selected_features[1]}.png', 
            dpi=300, bbox_inches='tight')
plt.show()
print(f"依赖图已保存到: {output_dir}/shap_dependence_plot_{selected_features[0]}_{selected_features[1]}.png")

plt.figure(figsize=(12, 10))
shap.dependence_plot(
    (selected_features[1], selected_features[2]), 
    shap_interaction_values, 
    X_test_processed[:100], 
    feature_names=selected_features,
    show=False
)
plt.title(f'SHAP Dependence Plot: {selected_features[1]} vs {selected_features[2]}', 
          fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(f'{output_dir}/shap_dependence_plot_{selected_features[1]}_{selected_features[2]}.png', 
            dpi=300, bbox_inches='tight')
plt.show()
print(f"依赖图已保存到: {output_dir}/shap_dependence_plot_{selected_features[1]}_{selected_features[2]}.png")

print("\n" + "="*70)
print("绘制SHAP瀑布图（单个样本）...")
print("="*70)

sample_idx = 0
plt.figure(figsize=(10, 8))
shap.waterfall_plot(shap.Explanation(values=shap_values[sample_idx],
                                 base_values=explainer.expected_value,
                                 data=X_test_processed.iloc[sample_idx],
                                 feature_names=selected_features),
                   show=False, max_display=10)
plt.title(f'SHAP Waterfall Plot - Sample {sample_idx+1}', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(f'{output_dir}/shap_waterfall_plot.png', dpi=300, bbox_inches='tight')
plt.show()
print(f"瀑布图已保存到: {output_dir}/shap_waterfall_plot.png")

print("\n" + "="*80)
print("SHAP分析完成！")
print("="*80)
print("\n生成的文件:")
print(f"1. {output_dir}/shap_summary_plot.png - SHAP摘要图（散点图）")
print(f"2. {output_dir}/shap_feature_importance_bar.png - SHAP特征重要性（条形图）")
print(f"3. {output_dir}/shap_feature_importance.png - SHAP特征重要性（自定义图）")
print(f"4. {output_dir}/shap_feature_importance.csv - 特征重要性数据")
print(f"5. {output_dir}/shap_decision_path_1-20.png - 决策路径图（20条）")
print(f"6. {output_dir}/shap_interaction_summary.png - 特征交互摘要图")
print(f"7. {output_dir}/shap_dependence_plot_*.png - 特征依赖图")
print(f"8. {output_dir}/shap_waterfall_plot.png - SHAP瀑布图")
print("\n关键发现:")
print(f"- 最重要特征: {feature_importance_df.iloc[0]['Feature']}")
print(f"- 次重要特征: {feature_importance_df.iloc[1]['Feature']}")
print(f"- 特征重要性范围: {feature_importance_df['Mean |SHAP|'].min():.4f} - {feature_importance_df['Mean |SHAP|'].max():.4f}")
print("="*80)
