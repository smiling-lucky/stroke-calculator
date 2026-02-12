import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import json
import os
import warnings
warnings.filterwarnings('ignore')

output_dir = '8种机器学习'
os.makedirs(output_dir, exist_ok=True)

data = pd.read_excel(r'd:\中风指标\elderly_hypertension_60plus_standardized.xlsx')

selected_features = ['Dyslipidemia', 'PHR', 'Uric_acid', 'MCV', 'Exercise', 'Age', 'AIP']

y = data['Stroke']
X = data[selected_features]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print('='*70)
print('7种机器学习模型校准曲线')
print('='*70)
print(f"\n特征数量: {len(selected_features)}")
print(f"特征列表: {selected_features}")
print(f"X形状: {X.shape}")
print(f"y形状: {y.shape}")
print(f"中风患病率: {y.sum()/len(y)*100:.2f}%")
print(f"\n训练集大小: {X_train.shape[0]}")
print(f"测试集大小: {X_test.shape[0]}")

with open(r'd:\中风指标\随机搜索优化结果\超参数优化结果.json', 'r', encoding='utf-8') as f:
    optimized_params = json.load(f)

models = {
    'Random Forest': {
        'model': RandomForestClassifier(
            n_estimators=optimized_params['Random Forest']['best_params']['n_estimators'],
            max_depth=optimized_params['Random Forest']['best_params']['max_depth'],
            min_samples_split=optimized_params['Random Forest']['best_params']['min_samples_split'],
            min_samples_leaf=optimized_params['Random Forest']['best_params']['min_samples_leaf'],
            max_features=optimized_params['Random Forest']['best_params']['max_features'],
            class_weight='balanced',
            random_state=42
        ),
        'color': '#1f77b4'
    },
    'XGBoost': {
        'model': GradientBoostingClassifier(
            n_estimators=optimized_params['XGBoost']['best_params']['n_estimators'],
            max_depth=optimized_params['XGBoost']['best_params']['max_depth'],
            learning_rate=optimized_params['XGBoost']['best_params']['learning_rate'],
            min_samples_split=optimized_params['XGBoost']['best_params']['min_samples_split'],
            min_samples_leaf=optimized_params['XGBoost']['best_params']['min_samples_leaf'],
            subsample=optimized_params['XGBoost']['best_params']['subsample'],
            random_state=42
        ),
        'color': '#ff7f0e'
    },
    'SVM': {
        'model': SVC(
            C=optimized_params['SVM']['best_params']['C'],
            gamma=optimized_params['SVM']['best_params']['gamma'],
            kernel=optimized_params['SVM']['best_params']['kernel'],
            probability=True,
            class_weight='balanced',
            random_state=42
        ),
        'color': '#2ca02c'
    },
    'kNN': {
        'model': KNeighborsClassifier(
            n_neighbors=optimized_params['kNN']['best_params']['n_neighbors'],
            weights=optimized_params['kNN']['best_params']['weights'],
            metric=optimized_params['kNN']['best_params']['metric']
        ),
        'color': '#d62728'
    },
    'Gradient Boosting': {
        'model': GradientBoostingClassifier(
            n_estimators=optimized_params['Gradient Boosting']['best_params']['n_estimators'],
            max_depth=optimized_params['Gradient Boosting']['best_params']['max_depth'],
            learning_rate=optimized_params['Gradient Boosting']['best_params']['learning_rate'],
            min_samples_split=optimized_params['Gradient Boosting']['best_params']['min_samples_split'],
            min_samples_leaf=optimized_params['Gradient Boosting']['best_params']['min_samples_leaf'],
            subsample=optimized_params['Gradient Boosting']['best_params']['subsample'],
            random_state=42
        ),
        'color': '#9467bd'
    },
    'AdaBoost': {
        'model': AdaBoostClassifier(
            n_estimators=optimized_params['AdaBoost']['best_params']['n_estimators'],
            learning_rate=optimized_params['AdaBoost']['best_params']['learning_rate'],
            random_state=42
        ),
        'color': '#8c564b'
    },
    'Neural Network': {
        'model': MLPClassifier(
            hidden_layer_sizes=tuple(optimized_params['Neural Network']['best_params']['hidden_layer_sizes']),
            activation=optimized_params['Neural Network']['best_params']['activation'],
            alpha=optimized_params['Neural Network']['best_params']['alpha'],
            learning_rate=optimized_params['Neural Network']['best_params']['learning_rate'],
            max_iter=1000,
            random_state=42
        ),
        'color': '#e377c2'
    }
}

print("\n" + "="*70)
print("开始训练模型并生成校准曲线...")
print("="*70)

calibration_data = {}
brier_scores = {}

for model_name, model_info in models.items():
    print(f"\n{'='*50}")
    print(f"处理模型: {model_name}")
    print(f"{'='*50}")
    
    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('model', model_info['model'])
    ])
    
    pipeline.fit(X_train, y_train)
    
    y_pred_prob = pipeline.predict_proba(X_test)[:, 1]
    
    prob_true, prob_pred = calibration_curve(y_test, y_pred_prob, n_bins=10, strategy='uniform')
    
    brier_score = brier_score_loss(y_test, y_pred_prob)
    
    calibration_data[model_name] = {
        'prob_true': prob_true,
        'prob_pred': prob_pred,
        'color': model_info['color']
    }
    
    brier_scores[model_name] = brier_score
    
    print(f"Brier Score: {brier_score:.4f}")

print("\n" + "="*70)
print("绘制校准曲线...")
print("="*70)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

for model_name, data in calibration_data.items():
    ax1.plot(data['prob_pred'], data['prob_true'], 
             marker='o', linewidth=2, label=f"{model_name} (Brier: {brier_scores[model_name]:.4f})",
             color=data['color'], markersize=6)

ax1.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')

ax1.set_xlabel('Mean Predicted Probability', fontsize=12, fontweight='bold')
ax1.set_ylabel('Fraction of Positives', fontsize=12, fontweight='bold')
ax1.set_title('Calibration Curves - 7 ML Models\n(Stroke Prediction)', fontsize=14, fontweight='bold')
ax1.legend(loc='upper left', fontsize=9, framealpha=0.9)
ax1.grid(True, alpha=0.3)
ax1.set_xlim([0, 1])
ax1.set_ylim([0, 1])

sorted_models = sorted(brier_scores.items(), key=lambda x: x[1])
model_names_sorted = [x[0] for x in sorted_models]
brier_values_sorted = [x[1] for x in sorted_models]
colors_sorted = [calibration_data[x[0]]['color'] for x in sorted_models]

bars = ax2.barh(model_names_sorted, brier_values_sorted, color=colors_sorted, alpha=0.7)
ax2.set_xlabel('Brier Score', fontsize=12, fontweight='bold')
ax2.set_title('Brier Score Comparison (Lower is Better)', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='x')
ax2.set_xlim(0, max(brier_values_sorted) * 1.2)

for i, (bar, value) in enumerate(zip(bars, brier_values_sorted)):
    width = bar.get_width()
    ax2.text(width + 0.005, bar.get_y() + bar.get_height()/2, 
             f'{value:.4f}', ha='left', va='center', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig(f'{output_dir}/calibration_curves_7_models.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n校准曲线已保存到: {output_dir}/calibration_curves_7_models.png")

fig2, ax = plt.subplots(figsize=(14, 10))

for model_name, data in calibration_data.items():
    ax.plot(data['prob_pred'], data['prob_true'], 
             marker='o', linewidth=2.5, label=f"{model_name}",
             color=data['color'], markersize=7)

ax.plot([0, 1], [0, 1], 'k--', linewidth=2.5, label='Perfect Calibration')

ax.set_xlabel('Mean Predicted Probability', fontsize=14, fontweight='bold')
ax.set_ylabel('Fraction of Positives', fontsize=14, fontweight='bold')
ax.set_title('Calibration Curves - 7 Machine Learning Models\n(Stroke Prediction with Optimized Hyperparameters)', 
             fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='upper left', fontsize=11, framealpha=0.9, ncol=2)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

plt.tight_layout()
plt.savefig(f'{output_dir}/calibration_curves_7_models_large.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"大尺寸校准曲线已保存到: {output_dir}/calibration_curves_7_models_large.png")

brier_df = pd.DataFrame({
    'Model': list(brier_scores.keys()),
    'Brier_Score': list(brier_scores.values())
}).sort_values('Brier_Score')

brier_df.to_csv(f'{output_dir}/brier_scores_comparison.csv', index=False, encoding='utf-8-sig')

print(f"\nBrier分数已保存到: {output_dir}/brier_scores_comparison.csv")

print("\n" + "="*80)
print("Brier分数比较 (越低越好)")
print("="*80)
print(brier_df.to_string(index=False))

print("\n" + "="*80)
print("校准曲线绘制完成！")
print("="*80)
print("\n生成的文件:")
print(f"1. {output_dir}/calibration_curves_7_models.png - 双图版（校准曲线+Brier分数）")
print(f"2. {output_dir}/calibration_curves_7_models_large.png - 大尺寸校准曲线")
print(f"3. {output_dir}/brier_scores_comparison.csv - Brier分数比较表")
print("="*80)
