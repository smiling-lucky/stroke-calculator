import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, brier_score_loss
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import time
import json
import os
import warnings
warnings.filterwarnings('ignore')

output_dir = '8种机器学习'
os.makedirs(output_dir, exist_ok=True)
random_search_output_dir = '随机搜索优化结果'
os.makedirs(random_search_output_dir, exist_ok=True)

data = pd.read_excel(r'd:\中风指标\elderly_hypertension_60plus_standardized.xlsx')

selected_features = ['Dyslipidemia', 'PHR', 'Uric_acid', 'MCV', 'Exercise', 'Age', 'AIP']

y = data['Stroke']
X = data[selected_features]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print('='*70)
print('7种机器学习模型随机搜索超参数优化')
print('='*70)
print(f"\n特征数量: {len(selected_features)}")
print(f"特征列表: {selected_features}")
print(f"X形状: {X.shape}")
print(f"y形状: {y.shape}")
print(f"中风患病率: {y.sum()/len(y)*100:.2f}%")
print(f"\n训练集大小: {X_train.shape[0]}")
print(f"测试集大小: {X_test.shape[0]}")

cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
n_iterations = 50

param_distributions = {
    'Random Forest': {
        'model__n_estimators': [50, 100, 150, 200, 250, 300],
        'model__max_depth': [5, 8, 10, 12, 15, 18, 20],
        'model__min_samples_split': [2, 5, 10, 15, 20],
        'model__min_samples_leaf': [1, 2, 4, 6, 8, 10],
        'model__max_features': ['sqrt', 'log2', None]
    },
    'XGBoost': {
        'model__n_estimators': [50, 100, 150, 200, 250, 300],
        'model__max_depth': [3, 5, 7, 9, 11, 12],
        'model__learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
        'model__min_samples_split': [2, 5, 10, 15, 20],
        'model__min_samples_leaf': [1, 2, 4, 6, 8, 10],
        'model__subsample': [0.6, 0.7, 0.8, 0.9, 1.0]
    },
    'SVM': {
        'model__C': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        'model__gamma': [0.001, 0.01, 0.1, 1.0],
        'model__kernel': ['rbf']
    },
    'kNN': {
        'model__n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
        'model__weights': ['uniform', 'distance'],
        'model__metric': ['euclidean', 'manhattan']
    },
    'Gradient Boosting': {
        'model__n_estimators': [50, 100, 150, 200, 250, 300],
        'model__max_depth': [3, 5, 7, 9, 11, 12],
        'model__learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
        'model__min_samples_split': [2, 5, 10, 15, 20],
        'model__min_samples_leaf': [1, 2, 4, 6, 8, 10],
        'model__subsample': [0.6, 0.7, 0.8, 0.9, 1.0]
    },
    'AdaBoost': {
        'model__n_estimators': [50, 100, 150, 200, 250, 300],
        'model__learning_rate': [0.01, 0.05, 0.1, 0.5, 1.0, 1.5, 2.0]
    },
    'Neural Network': {
        'model__hidden_layer_sizes': [(64, 32), (100, 50), (128, 64), (150, 75), (200, 100)],
        'model__activation': ['relu', 'tanh'],
        'model__alpha': [0.0001, 0.001, 0.01, 0.1],
        'model__learning_rate': ['constant', 'adaptive']
    }
}

model_names = ['Random Forest', 'XGBoost', 'SVM', 'kNN', 'Gradient Boosting', 'AdaBoost', 'Neural Network']

all_results = {}
optimization_history = {}

print("\n" + "="*70)
print("开始随机搜索超参数优化...")
print("="*70)
print(f"优化方法: RandomizedSearchCV")
print(f"交叉验证: 5折 Stratified K-Fold")
print(f"随机采样次数: {n_iterations} 次/模型")
print("="*70)

for model_name in model_names:
    print(f"\n{'='*60}")
    print(f"优化模型: {model_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    if model_name == 'Random Forest':
        base_model = RandomForestClassifier(class_weight='balanced', random_state=42)
    elif model_name == 'XGBoost':
        base_model = GradientBoostingClassifier(random_state=42)
    elif model_name == 'SVM':
        base_model = SVC(probability=True, class_weight='balanced', random_state=42)
    elif model_name == 'kNN':
        base_model = KNeighborsClassifier()
    elif model_name == 'Gradient Boosting':
        base_model = GradientBoostingClassifier(random_state=42)
    elif model_name == 'AdaBoost':
        base_model = AdaBoostClassifier(random_state=42)
    elif model_name == 'Neural Network':
        base_model = MLPClassifier(max_iter=1000, random_state=42)
    
    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('model', base_model)
    ])
    
    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions[model_name],
        n_iter=n_iterations,
        cv=cv_inner,
        scoring='roc_auc',
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    random_search.fit(X_train, y_train)
    
    optimization_time = time.time() - start_time
    
    best_params = random_search.best_params_
    best_cv_auc = random_search.best_score_
    
    all_aucs = random_search.cv_results_['mean_test_score'].tolist()
    
    optimization_history[model_name] = {
        'iterations': n_iterations,
        'time': optimization_time,
        'best_auc': best_cv_auc,
        'all_aucs': all_aucs
    }
    
    all_results[model_name] = {
        'best_params': {k.replace('model__', ''): v for k, v in best_params.items()},
        'best_cv_auc': best_cv_auc,
        'optimization_time': optimization_time
    }
    
    print(f"\n最优超参数:")
    for param, value in best_params.items():
        if isinstance(value, float):
            print(f"  {param}: {value:.6f}")
        else:
            print(f"  {param}: {value}")
    print(f"\n最优交叉验证AUC: {best_cv_auc:.4f}")
    print(f"优化时间: {optimization_time:.2f}秒")

print("\n" + "="*70)
print("随机搜索优化完成！")
print("="*70)

with open(f'{random_search_output_dir}/超参数优化结果.json', 'w', encoding='utf-8') as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print(f"\n优化结果已保存到: {random_search_output_dir}/超参数优化结果.json")

summary_text = "="*80 + "\n"
summary_text += "7种机器学习模型随机搜索超参数优化结果\n"
summary_text += "="*80 + "\n\n"
summary_text += f"数据集: elderly_hypertension_60plus_standardized.xlsx\n"
summary_text += f"目标变量: Stroke\n"
summary_text += f"特征数量: {len(selected_features)}\n"
summary_text += f"特征列表: {', '.join(selected_features)}\n"
summary_text += f"训练集大小: {X_train.shape[0]}\n"
summary_text += f"测试集大小: {X_test.shape[0]}\n"
summary_text += f"中风患病率: {y.sum()/len(y)*100:.2f}%\n"
summary_text += f"优化方法: RandomizedSearchCV\n"
summary_text += f"交叉验证: 5折 Stratified K-Fold\n"
summary_text += f"随机采样次数: {n_iterations} 次/模型\n\n"
summary_text += "-"*80 + "\n"
summary_text += "各模型最优超参数\n"
summary_text += "-"*80 + "\n\n"

sorted_results = sorted(all_results.items(), key=lambda x: x[1]['best_cv_auc'], reverse=True)

for rank, (model_name, result) in enumerate(sorted_results, 1):
    summary_text += f"{rank}. {model_name}\n"
    summary_text += f"   最优CV AUC: {result['best_cv_auc']:.4f}\n"
    summary_text += f"   优化时间: {result['optimization_time']:.2f}秒\n"
    summary_text += "   超参数:\n"
    for param, value in result['best_params'].items():
        if isinstance(value, float):
            summary_text += f"      - {param}: {value:.6f}\n"
        else:
            summary_text += f"      - {param}: {value}\n"
    summary_text += "\n"

summary_text += "-"*80 + "\n"
summary_text += "生成的文件:\n"
summary_text += f"1. {random_search_output_dir}/超参数优化结果.json - JSON格式详细结果\n"
summary_text += f"2. {random_search_output_dir}/超参数优化总结.txt - 文本格式总结\n"
summary_text += f"3. {random_search_output_dir}/超参数优化过程.png - 优化过程可视化\n"
summary_text += "-"*80 + "\n"

with open(f'{random_search_output_dir}/超参数优化总结.txt', 'w', encoding='utf-8') as f:
    f.write(summary_text)

print(f"优化总结已保存到: {random_search_output_dir}/超参数优化总结.txt")

plt.figure(figsize=(14, 10))
colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))

for i, (model_name, history) in enumerate(optimization_history.items()):
    iterations = list(range(1, len(history['all_aucs']) + 1))
    plt.plot(iterations, history['all_aucs'], 
             color=colors[i], lw=2, alpha=0.8,
             label=f"{model_name} (Best: {history['best_auc']:.4f})")

plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Cross-Validation AUC', fontsize=12)
plt.title('Randomized Search Hyperparameter Optimization Progress\n(7 ML Models for Stroke Prediction)', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=9)
plt.grid(True, alpha=0.3)
plt.ylim(0.5, 1.0)

plt.tight_layout()
plt.savefig(f'{random_search_output_dir}/超参数优化过程.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"优化过程图已保存到: {random_search_output_dir}/超参数优化过程.png")

print("\n" + "="*80)
print("优化结果汇总表")
print("="*80)
print(f"\n{'模型':<20} {'最优CV AUC':<15} {'优化时间(秒)':<15}")
print("-"*50)
for model_name, result in sorted_results:
    print(f"{model_name:<20} {result['best_cv_auc']:<15.4f} {result['optimization_time']:<15.2f}")

print("\n" + "="*80)
print("所有优化任务完成！")
print("="*80)
