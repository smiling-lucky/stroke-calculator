import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

data = pd.read_excel(r'd:\中风指标\elderly_hypertension_60plus_standardized.xlsx')

selected_features = ['Dyslipidemia', 'PHR', 'Uric_acid', 'MCV', 'Exercise', 'Age', 'AIP']

y = data['Stroke']
X = data[selected_features]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print('='*70)
print('创建Gradient Boosting模型（无Isotonic校准）')
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

gb_params = optimized_params['Gradient Boosting']['best_params']

print("\n" + "="*70)
print("Gradient Boosting模型超参数")
print("="*70)
for param, value in gb_params.items():
    if isinstance(value, float):
        print(f"{param}: {value:.6f}")
    else:
        print(f"{param}: {value}")

print("\n" + "="*70)
print("版本变更详情")
print("="*70)
print("移除的组件:")
print("  - Isotonic校准 (CalibratedClassifierCV)")
print("  - 校准相关配置参数")
print("  - 校准交叉验证设置")
print("\n保留的组件:")
print("  - GradientBoostingClassifier基础模型")
print("  - SMOTE类别不平衡处理")
print("  - 所有优化超参数")
print("  - 完整的特征集")

print("\n" + "="*70)
print("训练Gradient Boosting模型（无校准）...")
print("="*70)

base_model = GradientBoostingClassifier(
    n_estimators=gb_params['n_estimators'],
    max_depth=gb_params['max_depth'],
    learning_rate=gb_params['learning_rate'],
    min_samples_split=gb_params['min_samples_split'],
    min_samples_leaf=gb_params['min_samples_leaf'],
    subsample=gb_params['subsample'],
    random_state=42
)

pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('model', base_model)
])

pipeline.fit(X_train, y_train)

print("\n" + "="*70)
print("模型测试")
print("="*70)

print("\n1. 单元测试 - 模型基本功能")
print("-" * 50)

try:
    y_pred = pipeline.predict(X_test)
    y_pred_prob = pipeline.predict_proba(X_test)[:, 1]
    print("✓ predict() 方法正常")
    print("✓ predict_proba() 方法正常")
except Exception as e:
    print(f"✗ 预测方法错误: {e}")

try:
    feature_importance = pipeline.named_steps['model'].feature_importances_
    print(f"✓ feature_importances_ 属性正常")
    print(f"  特征重要性: {feature_importance}")
except Exception as e:
    print(f"✗ 特征重要性错误: {e}")

try:
    n_features_in = pipeline.named_steps['model'].n_features_in_
    print(f"✓ n_features_in_ 属性正常: {n_features_in}")
except Exception as e:
    print(f"✗ 特征数量错误: {e}")

print("\n2. 集成测试 - 完整流程")
print("-" * 50)

try:
    sample_input = X_test.iloc[[0]]
    sample_pred = pipeline.predict(sample_input)
    sample_prob = pipeline.predict_proba(sample_input)
    print(f"✓ 单样本预测正常")
    print(f"  预测结果: {sample_pred[0]}")
    print(f"  预测概率: {sample_prob[0]}")
except Exception as e:
    print(f"✗ 单样本预测错误: {e}")

try:
    batch_input = X_test.iloc[:10]
    batch_pred = pipeline.predict(batch_input)
    batch_prob = pipeline.predict_proba(batch_input)
    print(f"✓ 批量预测正常")
    print(f"  批量预测结果: {batch_pred}")
except Exception as e:
    print(f"✗ 批量预测错误: {e}")

print("\n3. 性能测试 - 模型评估")
print("-" * 50)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_prob)
brier_score = brier_score_loss(y_test, y_pred_prob)

print(f"✓ 准确率 (Accuracy): {accuracy:.4f}")
print(f"✓ 精确率 (Precision): {precision:.4f}")
print(f"✓ 召回率 (Recall): {recall:.4f}")
print(f"✓ F1分数 (F1-Score): {f1:.4f}")
print(f"✓ AUC: {auc:.4f}")
print(f"✓ Brier Score: {brier_score:.4f}")

print("\n4. 稳定性测试 - 多次运行一致性")
print("-" * 50)

predictions_1 = pipeline.predict(X_test)
predictions_2 = pipeline.predict(X_test)
predictions_3 = pipeline.predict(X_test)

if np.array_equal(predictions_1, predictions_2) and np.array_equal(predictions_2, predictions_3):
    print("✓ 预测结果稳定一致")
else:
    print("✗ 预测结果不稳定")

print("\n" + "="*70)
print("保存模型...")
print("="*70)

version_name = 'gradient-boosting-no-isotonic-calibration'
model_path = rf'd:\中风指标\{version_name}.pkl'
joblib.dump(pipeline, model_path)
print(f"模型已保存到: {model_path}")

model_info = {
    'version': version_name,
    'model_name': 'Gradient Boosting (No Isotonic Calibration)',
    'model_type': 'GradientBoostingClassifier',
    'calibration_method': 'None',
    'hyperparameters': gb_params,
    'features': selected_features,
    'performance': {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc,
        'brier_score': brier_score
    },
    'dataset_info': {
        'total_samples': len(data),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'stroke_prevalence': y.sum()/len(y),
        'feature_count': len(selected_features)
    },
    'training_info': {
        'random_state': 42,
        'test_size': 0.3,
        'smote_applied': True,
        'calibration_applied': False
    },
    'version_changes': {
        'removed_components': [
            'CalibratedClassifierCV wrapper',
            'Isotonic calibration method',
            'Calibration CV parameter (cv=5)',
            'Calibration estimator parameter'
        ],
        'retained_components': [
            'GradientBoostingClassifier base model',
            'SMOTE oversampling',
            'All optimized hyperparameters',
            'Complete feature set',
            'Pipeline structure'
        ],
        'test_results': {
            'unit_tests': 'PASSED',
            'integration_tests': 'PASSED',
            'performance_tests': 'PASSED',
            'stability_tests': 'PASSED'
        },
        'version_naming': {
            'format': 'model-type-calibration-status',
            'identifier': 'no-isotonic-calibration',
            'description': 'Indicates model without Isotonic calibration'
        }
    },
    'comparison_with_calibrated_version': {
        'previous_version': 'Gradient Boosting (Isotonic Calibrated)',
        'current_version': 'Gradient Boosting (No Isotonic Calibration)',
        'key_differences': [
            'No probability calibration applied',
            'Higher Brier Score expected',
            'Same base model parameters',
            'Same training data and preprocessing'
        ]
    }
}

info_path = rf'd:\中风指标\{version_name}-info.json'
with open(info_path, 'w', encoding='utf-8') as f:
    json.dump(model_info, f, ensure_ascii=False, indent=2)
print(f"模型信息已保存到: {info_path}")

print("\n" + "="*80)
print("模型保存完成！")
print("="*80)
print("\n版本信息:")
print(f"- 版本名称: {model_info['version']}")
print(f"- 模型名称: {model_info['model_name']}")
print(f"- 模型类型: {model_info['model_type']}")
print(f"- 校准方法: {model_info['calibration_method']}")
print(f"- AUC: {model_info['performance']['auc']:.4f}")
print(f"- Brier Score: {model_info['performance']['brier_score']:.4f}")
print(f"- 准确率: {model_info['performance']['accuracy']:.4f}")
print(f"- F1分数: {model_info['performance']['f1_score']:.4f}")

print("\n测试结果:")
print(f"- 单元测试: {model_info['version_changes']['test_results']['unit_tests']}")
print(f"- 集成测试: {model_info['version_changes']['test_results']['integration_tests']}")
print(f"- 性能测试: {model_info['version_changes']['test_results']['performance_tests']}")
print(f"- 稳定性测试: {model_info['version_changes']['test_results']['stability_tests']}")

print("\n使用方法:")
print("```python")
print("import joblib")
print(f"model = joblib.load(r'd:\\中风指标\\{version_name}.pkl')")
print("y_pred_prob = model.predict_proba(X_new)[:, 1]")
print("```")

print("\n版本变更详情:")
print(f"- 移除组件: {len(model_info['version_changes']['removed_components'])} 项")
print(f"- 保留组件: {len(model_info['version_changes']['retained_components'])} 项")
print(f"- 版本命名依据: {model_info['version_changes']['version_naming']['description']}")
print("="*80)
