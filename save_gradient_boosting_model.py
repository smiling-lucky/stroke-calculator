import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
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
print('保存Gradient Boosting模型为最优模型（基于Isotonic校准分析）')
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
print("训练Gradient Boosting模型...")
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
print("应用Isotonic校准...")
print("="*70)

calibrated_model = CalibratedClassifierCV(
    estimator=pipeline,
    method='isotonic',
    cv=5
)

calibrated_model.fit(X_train, y_train)

y_pred_prob = calibrated_model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_prob >= 0.5).astype(int)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_prob)
brier_score = brier_score_loss(y_test, y_pred_prob)

print("\n" + "="*70)
print("模型性能评估")
print("="*70)
print(f"准确率 (Accuracy): {accuracy:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"召回率 (Recall): {recall:.4f}")
print(f"F1分数 (F1-Score): {f1:.4f}")
print(f"AUC: {auc:.4f}")
print(f"Brier Score: {brier_score:.4f}")

print("\n" + "="*70)
print("保存模型...")
print("="*70)

model_path = r'd:\中风指标\最优模型.pkl'
joblib.dump(calibrated_model, model_path)
print(f"模型已保存到: {model_path}")

model_info = {
    'model_name': 'Gradient Boosting (Isotonic Calibrated)',
    'model_type': 'GradientBoostingClassifier',
    'calibration_method': 'Isotonic',
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
        'calibration_applied': True,
        'calibration_cv': 5
    }
}

info_path = r'd:\中风指标\最优模型信息.json'
with open(info_path, 'w', encoding='utf-8') as f:
    json.dump(model_info, f, ensure_ascii=False, indent=2)
print(f"模型信息已保存到: {info_path}")

print("\n" + "="*80)
print("最优模型保存完成！")
print("="*80)
print("\n模型信息:")
print(f"- 模型名称: {model_info['model_name']}")
print(f"- 模型类型: {model_info['model_type']}")
print(f"- 校准方法: {model_info['calibration_method']}")
print(f"- AUC: {model_info['performance']['auc']:.4f}")
print(f"- Brier Score: {model_info['performance']['brier_score']:.4f}")
print(f"- 准确率: {model_info['performance']['accuracy']:.4f}")
print(f"- F1分数: {model_info['performance']['f1_score']:.4f}")
print("\n使用方法:")
print("```python")
print("import joblib")
print("model = joblib.load(r'd:\\中风指标\\最优模型.pkl')")
print("y_pred_prob = model.predict_proba(X_new)[:, 1]")
print("```")
print("="*80)
