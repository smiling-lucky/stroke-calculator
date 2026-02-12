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

X_train_cal, X_val, y_train_cal, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print('='*70)
print('7种机器学习模型Isotonic校准曲线')
print('='*70)
print(f"\n特征数量: {len(selected_features)}")
print(f"特征列表: {selected_features}")
print(f"X形状: {X.shape}")
print(f"y形状: {y.shape}")
print(f"中风患病率: {y.sum()/len(y)*100:.2f}%")
print(f"\n训练集大小: {X_train.shape[0]} (训练校准: {X_train_cal.shape[0]}, 验证: {X_val.shape[0]})")
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
print("开始训练模型并生成Isotonic校准曲线...")
print("="*70)

calibration_data_uncalibrated = {}
calibration_data_calibrated = {}
brier_scores_uncalibrated = {}
brier_scores_calibrated = {}

for model_name, model_info in models.items():
    print(f"\n{'='*50}")
    print(f"处理模型: {model_name}")
    print(f"{'='*50}")
    
    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('model', model_info['model'])
    ])
    
    pipeline.fit(X_train_cal, y_train_cal)
    
    y_pred_prob_uncalibrated = pipeline.predict_proba(X_test)[:, 1]
    brier_uncalibrated = brier_score_loss(y_test, y_pred_prob_uncalibrated)
    
    prob_true_uncal, prob_pred_uncal = calibration_curve(
        y_test, y_pred_prob_uncalibrated, n_bins=10, strategy='uniform'
    )
    
    calibration_data_uncalibrated[model_name] = {
        'prob_true': prob_true_uncal,
        'prob_pred': prob_pred_uncal,
        'color': model_info['color']
    }
    
    brier_scores_uncalibrated[model_name] = brier_uncalibrated
    
    print(f"未校准 Brier Score: {brier_uncalibrated:.4f}")
    
    calibrated_model = CalibratedClassifierCV(
        estimator=pipeline,
        method='isotonic',
        cv=5
    )
    
    calibrated_model.fit(X_train_cal, y_train_cal)
    
    y_pred_prob_calibrated = calibrated_model.predict_proba(X_test)[:, 1]
    brier_calibrated = brier_score_loss(y_test, y_pred_prob_calibrated)
    
    prob_true_cal, prob_pred_cal = calibration_curve(
        y_test, y_pred_prob_calibrated, n_bins=10, strategy='uniform'
    )
    
    calibration_data_calibrated[model_name] = {
        'prob_true': prob_true_cal,
        'prob_pred': prob_pred_cal,
        'color': model_info['color']
    }
    
    brier_scores_calibrated[model_name] = brier_calibrated
    
    improvement = ((brier_uncalibrated - brier_calibrated) / brier_uncalibrated) * 100
    print(f"Isotonic校准 Brier Score: {brier_calibrated:.4f}")
    print(f"改进: {improvement:+.2f}%")

print("\n" + "="*70)
print("绘制Isotonic校准曲线...")
print("="*70)

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for idx, (model_name, data_uncal) in enumerate(calibration_data_uncalibrated.items()):
    ax = axes[idx]
    data_cal = calibration_data_calibrated[model_name]
    
    ax.plot(data_uncal['prob_pred'], data_uncal['prob_true'], 
             marker='o', linewidth=2.5, label='Uncalibrated',
             color=data_uncal['color'], markersize=6, alpha=0.7)
    
    ax.plot(data_cal['prob_pred'], data_cal['prob_true'], 
             marker='s', linewidth=2.5, label='Isotonic Calibrated',
             color=data_cal['color'], markersize=6, linestyle='--')
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfect', alpha=0.5)
    
    ax.set_xlabel('Mean Predicted Probability', fontsize=10, fontweight='bold')
    ax.set_ylabel('Fraction of Positives', fontsize=10, fontweight='bold')
    ax.set_title(f'{model_name}\nBrier: {brier_scores_uncalibrated[model_name]:.4f} → {brier_scores_calibrated[model_name]:.4f}', 
                 fontsize=11, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

axes[-1].axis('off')

plt.suptitle('Isotonic Calibration Curves - 7 ML Models\n(Stroke Prediction)', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(f'{output_dir}/calibration_curves_isotonic.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nIsotonic校准曲线已保存到: {output_dir}/calibration_curves_isotonic.png")

fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

for model_name, data in calibration_data_uncalibrated.items():
    ax1.plot(data['prob_pred'], data['prob_true'], 
             marker='o', linewidth=2, label=f"{model_name}",
             color=data['color'], markersize=6, alpha=0.6)

ax1.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')

ax1.set_xlabel('Mean Predicted Probability', fontsize=12, fontweight='bold')
ax1.set_ylabel('Fraction of Positives', fontsize=12, fontweight='bold')
ax1.set_title('Uncalibrated Models', fontsize=14, fontweight='bold')
ax1.legend(loc='upper left', fontsize=9, framealpha=0.9)
ax1.grid(True, alpha=0.3)
ax1.set_xlim([0, 1])
ax1.set_ylim([0, 1])

for model_name, data in calibration_data_calibrated.items():
    ax2.plot(data['prob_pred'], data['prob_true'], 
             marker='s', linewidth=2, label=f"{model_name}",
             color=data['color'], markersize=6, alpha=0.6)

ax2.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')

ax2.set_xlabel('Mean Predicted Probability', fontsize=12, fontweight='bold')
ax2.set_ylabel('Fraction of Positives', fontsize=12, fontweight='bold')
ax2.set_title('Isotonic Calibrated Models', fontsize=14, fontweight='bold')
ax2.legend(loc='upper left', fontsize=9, framealpha=0.9)
ax2.grid(True, alpha=0.3)
ax2.set_xlim([0, 1])
ax2.set_ylim([0, 1])

plt.tight_layout()
plt.savefig(f'{output_dir}/calibration_comparison_isotonic.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"校准对比图已保存到: {output_dir}/calibration_comparison_isotonic.png")

brier_comparison_df = pd.DataFrame({
    'Model': list(brier_scores_uncalibrated.keys()),
    'Brier_Uncalibrated': list(brier_scores_uncalibrated.values()),
    'Brier_Isotonic': list(brier_scores_calibrated.values()),
    'Improvement_%': [((brier_scores_uncalibrated[k] - brier_scores_calibrated[k]) / brier_scores_uncalibrated[k] * 100) 
                     for k in brier_scores_uncalibrated.keys()]
}).sort_values('Brier_Isotonic')

brier_comparison_df.to_csv(f'{output_dir}/brier_scores_isotonic_comparison.csv', index=False, encoding='utf-8-sig')

print(f"\nBrier分数比较已保存到: {output_dir}/brier_scores_isotonic_comparison.csv")

print("\n" + "="*80)
print("Isotonic校准效果比较")
print("="*80)
print(brier_comparison_df.to_string(index=False))

print("\n" + "="*80)
print("Isotonic校准总结")
print("="*80)

best_improvement = brier_comparison_df['Improvement_%'].max()
best_model = brier_comparison_df.loc[brier_comparison_df['Improvement_%'].idxmax(), 'Model']

print(f"\n校准效果最好的模型: {best_model}")
print(f"最大改进: {best_improvement:.2f}%")

avg_improvement = brier_comparison_df['Improvement_%'].mean()
print(f"平均改进: {avg_improvement:.2f}%")

models_improved = (brier_comparison_df['Improvement_%'] > 0).sum()
print(f"校准后性能提升的模型数: {models_improved}/7")

print("\n" + "="*80)
print("Isotonic校准曲线绘制完成！")
print("="*80)
print("\n生成的文件:")
print(f"1. {output_dir}/calibration_curves_isotonic.png - 各模型Isotonic校准对比（7子图）")
print(f"2. {output_dir}/calibration_comparison_isotonic.png - 校准前后对比（双图）")
print(f"3. {output_dir}/brier_scores_isotonic_comparison.csv - Brier分数比较表")
print("="*80)
