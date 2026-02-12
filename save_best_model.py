import pandas as pd
import json
import joblib
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

data = pd.read_excel(r'd:\中风指标\elderly_hypertension_60plus_standardized.xlsx')

selected_features = ['Dyslipidemia', 'PHR', 'Uric_acid', 'MCV', 'Exercise', 'Age', 'AIP']

y = data['Stroke']
X = data[selected_features]

performance_df = pd.read_csv(r'd:\中风指标\8种机器学习\models_performance_comparison_optimized.csv')

best_model_name = performance_df.loc[performance_df['AUC'].idxmax(), 'Model']
best_auc = performance_df['AUC'].max()

print('='*70)
print('最优模型信息保存')
print('='*70)
print(f"\n最优模型: {best_model_name}")
print(f"最优AUC: {best_auc:.4f}")

with open(r'd:\中风指标\随机搜索优化结果\超参数优化结果.json', 'r', encoding='utf-8') as f:
    optimized_params = json.load(f)

best_params = optimized_params[best_model_name]['best_params']

print(f"\n最优超参数:")
for param, value in best_params.items():
    if isinstance(value, float):
        print(f"  {param}: {value:.6f}")
    else:
        print(f"  {param}: {value}")

if best_model_name == 'Random Forest':
    best_model = RandomForestClassifier(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],
        max_features=best_params['max_features'],
        class_weight='balanced',
        random_state=42
    )
elif best_model_name == 'XGBoost':
    from sklearn.ensemble import GradientBoostingClassifier
    best_model = GradientBoostingClassifier(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        learning_rate=best_params['learning_rate'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],
        subsample=best_params['subsample'],
        random_state=42
    )
elif best_model_name == 'SVM':
    from sklearn.svm import SVC
    best_model = SVC(
        C=best_params['C'],
        gamma=best_params['gamma'],
        kernel=best_params['kernel'],
        probability=True,
        class_weight='balanced',
        random_state=42
    )
elif best_model_name == 'kNN':
    from sklearn.neighbors import KNeighborsClassifier
    best_model = KNeighborsClassifier(
        n_neighbors=best_params['n_neighbors'],
        weights=best_params['weights'],
        metric=best_params['metric']
    )
elif best_model_name == 'Gradient Boosting':
    from sklearn.ensemble import GradientBoostingClassifier
    best_model = GradientBoostingClassifier(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        learning_rate=best_params['learning_rate'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],
        subsample=best_params['subsample'],
        random_state=42
    )
elif best_model_name == 'AdaBoost':
    from sklearn.ensemble import AdaBoostClassifier
    best_model = AdaBoostClassifier(
        n_estimators=best_params['n_estimators'],
        learning_rate=best_params['learning_rate'],
        random_state=42
    )
elif best_model_name == 'Neural Network':
    from sklearn.neural_network import MLPClassifier
    best_model = MLPClassifier(
        hidden_layer_sizes=tuple(best_params['hidden_layer_sizes']),
        activation=best_params['activation'],
        alpha=best_params['alpha'],
        learning_rate=best_params['learning_rate'],
        max_iter=1000,
        random_state=42
    )

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("\n正在训练最优模型...")

pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('model', best_model)
])

pipeline.fit(X_train, y_train)

print("模型训练完成！")

model_info = {
    'model_name': best_model_name,
    'auc': float(best_auc),
    'hyperparameters': best_params,
    'features': selected_features,
    'feature_count': len(selected_features),
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    'total_samples': len(X),
    'stroke_prevalence': float(y.sum()/len(y)*100),
    'random_state': 42
}

with open(r'd:\中风指标\最优模型信息.json', 'w', encoding='utf-8') as f:
    json.dump(model_info, f, ensure_ascii=False, indent=2)

print(f"\n最优模型信息已保存到: d:\中风指标\最优模型信息.json")

joblib.dump(pipeline, r'd:\中风指标\最优模型.pkl')
print(f"最优模型已保存到: d:\中风指标\最优模型.pkl")

print("\n" + "="*70)
print("最优模型详细信息")
print("="*70)
print(f"模型名称: {model_info['model_name']}")
print(f"AUC: {model_info['auc']:.4f}")
print(f"特征数量: {model_info['feature_count']}")
print(f"特征列表: {', '.join(model_info['features'])}")
print(f"训练样本数: {model_info['training_samples']}")
print(f"测试样本数: {model_info['test_samples']}")
print(f"总样本数: {model_info['total_samples']}")
print(f"中风患病率: {model_info['stroke_prevalence']:.2f}%")
print(f"随机种子: {model_info['random_state']}")
print("\n超参数:")
for param, value in best_params.items():
    if isinstance(value, float):
        print(f"  {param}: {value:.6f}")
    else:
        print(f"  {param}: {value}")

print("\n" + "="*70)
print("保存完成！")
print("="*70)
print("\n后续使用方法:")
print("1. 加载模型: model = joblib.load('d:/中风指标/最优模型.pkl')")
print("2. 加载模型信息: info = json.load(open('d:/中风指标/最优模型信息.json', 'r', encoding='utf-8'))")
print("3. 预测: y_pred = model.predict(X_new)")
print("4. 预测概率: y_pred_prob = model.predict_proba(X_new)[:, 1]")
print("="*70)
