import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import time
import json
import os
import warnings
warnings.filterwarnings('ignore')

output_dir = '8种机器学习'
os.makedirs(output_dir, exist_ok=True)
hyperopt_output_dir = '贝叶斯优化结果'
os.makedirs(hyperopt_output_dir, exist_ok=True)

data = pd.read_excel(r'd:\中风指标\elderly_hypertension_60plus_standardized.xlsx')

selected_features = ['Dyslipidemia', 'PHR', 'Uric_acid', 'MCV', 'Exercise', 'Age', 'AIP']

y = data['Stroke']
X = data[selected_features]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print('='*70)
print('7种机器学习模型贝叶斯超参数优化')
print('='*70)
print(f"\n特征数量: {len(selected_features)}")
print(f"特征列表: {selected_features}")
print(f"X形状: {X.shape}")
print(f"y形状: {y.shape}")
print(f"中风患病率: {y.sum()/len(y)*100:.2f}%")
print(f"\n训练集大小: {X_train.shape[0]}")
print(f"测试集大小: {X_test.shape[0]}")

cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
n_iterations = 20

choice_mappings = {
    'Random Forest': {
        'max_features': ['sqrt', 'log2', None]
    },
    'SVM': {
        'kernel': ['rbf', 'poly', 'sigmoid']
    },
    'kNN': {
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    },
    'Neural Network': {
        'activation': ['relu', 'tanh'],
        'learning_rate': ['constant', 'adaptive']
    }
}

def create_pipeline(model_name, params):
    smote = SMOTE(random_state=42)
    
    if model_name == 'Random Forest':
        model = RandomForestClassifier(
            n_estimators=int(params['n_estimators']),
            max_depth=int(params['max_depth']),
            min_samples_split=int(params['min_samples_split']),
            min_samples_leaf=int(params['min_samples_leaf']),
            max_features=params['max_features'],
            class_weight='balanced',
            random_state=42
        )
    elif model_name == 'XGBoost':
        model = GradientBoostingClassifier(
            n_estimators=int(params['n_estimators']),
            max_depth=int(params['max_depth']),
            learning_rate=params['learning_rate'],
            min_samples_split=int(params['min_samples_split']),
            min_samples_leaf=int(params['min_samples_leaf']),
            subsample=params['subsample'],
            random_state=42
        )
    elif model_name == 'SVM':
        model = SVC(
            C=params['C'],
            gamma=params['gamma'],
            kernel=params['kernel'],
            probability=True,
            class_weight='balanced',
            random_state=42
        )
    elif model_name == 'kNN':
        model = KNeighborsClassifier(
            n_neighbors=int(params['n_neighbors']),
            weights=params['weights'],
            metric=params['metric']
        )
    elif model_name == 'Gradient Boosting':
        model = GradientBoostingClassifier(
            n_estimators=int(params['n_estimators']),
            max_depth=int(params['max_depth']),
            learning_rate=params['learning_rate'],
            min_samples_split=int(params['min_samples_split']),
            min_samples_leaf=int(params['min_samples_leaf']),
            subsample=params['subsample'],
            random_state=42
        )
    elif model_name == 'AdaBoost':
        model = AdaBoostClassifier(
            n_estimators=int(params['n_estimators']),
            learning_rate=params['learning_rate'],
            random_state=42
        )
    elif model_name == 'Neural Network':
        hidden_layer_size = int(params['hidden_layer_size'])
        model = MLPClassifier(
            hidden_layer_sizes=(hidden_layer_size, hidden_layer_size // 2),
            activation=params['activation'],
            alpha=params['alpha'],
            learning_rate=params['learning_rate'],
            max_iter=1000,
            random_state=42
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    pipeline = ImbPipeline([
        ('smote', smote),
        ('model', model)
    ])
    
    return pipeline

def objective(params, model_name):
    try:
        pipeline = create_pipeline(model_name, params)
        
        scores = []
        for train_idx, val_idx in cv_inner.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            pipeline.fit(X_tr, y_tr)
            y_pred_prob = pipeline.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, y_pred_prob)
            scores.append(score)
        
        mean_score = np.mean(scores)
        
        return {'loss': -mean_score, 'status': STATUS_OK, 'auc': mean_score}
    except Exception as e:
        return {'loss': 0, 'status': STATUS_OK, 'auc': 0}

param_spaces = {
    'Random Forest': {
        'n_estimators': hp.quniform('n_estimators', 50, 300, 10),
        'max_depth': hp.quniform('max_depth', 3, 20, 1),
        'min_samples_split': hp.quniform('min_samples_split', 2, 20, 1),
        'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 10, 1),
        'max_features': hp.choice('max_features', ['sqrt', 'log2', None])
    },
    'XGBoost': {
        'n_estimators': hp.quniform('n_estimators', 50, 300, 10),
        'max_depth': hp.quniform('max_depth', 3, 12, 1),
        'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
        'min_samples_split': hp.quniform('min_samples_split', 2, 20, 1),
        'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 10, 1),
        'subsample': hp.uniform('subsample', 0.6, 1.0)
    },
    'SVM': {
        'C': hp.loguniform('C', np.log(0.01), np.log(100)),
        'gamma': hp.loguniform('gamma', np.log(0.0001), np.log(10)),
        'kernel': hp.choice('kernel', ['rbf', 'poly', 'sigmoid'])
    },
    'kNN': {
        'n_neighbors': hp.quniform('n_neighbors', 3, 21, 2),
        'weights': hp.choice('weights', ['uniform', 'distance']),
        'metric': hp.choice('metric', ['euclidean', 'manhattan', 'minkowski'])
    },
    'Gradient Boosting': {
        'n_estimators': hp.quniform('n_estimators', 50, 300, 10),
        'max_depth': hp.quniform('max_depth', 3, 12, 1),
        'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
        'min_samples_split': hp.quniform('min_samples_split', 2, 20, 1),
        'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 10, 1),
        'subsample': hp.uniform('subsample', 0.6, 1.0)
    },
    'AdaBoost': {
        'n_estimators': hp.quniform('n_estimators', 50, 300, 10),
        'learning_rate': hp.uniform('learning_rate', 0.01, 2.0)
    },
    'Neural Network': {
        'hidden_layer_size': hp.quniform('hidden_layer_size', 32, 256, 16),
        'activation': hp.choice('activation', ['relu', 'tanh']),
        'alpha': hp.loguniform('alpha', np.log(0.0001), np.log(0.1)),
        'learning_rate': hp.choice('learning_rate', ['constant', 'adaptive'])
    }
}

model_names = ['Random Forest', 'XGBoost', 'SVM', 'kNN', 'Gradient Boosting', 'AdaBoost', 'Neural Network']

all_results = {}
optimization_history = {}

print("\n" + "="*70)
print("开始贝叶斯超参数优化...")
print("="*70)
print(f"优化方法: TPE (Tree-structured Parzen Estimator)")
print(f"交叉验证: 5折 Stratified K-Fold")
print(f"迭代次数: {n_iterations} 次/模型")
print("="*70)

for model_name in model_names:
    print(f"\n{'='*60}")
    print(f"优化模型: {model_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    trials = Trials()
    
    best = fmin(
        fn=lambda params: objective(params, model_name),
        space=param_spaces[model_name],
        algo=tpe.suggest,
        max_evals=n_iterations,
        trials=trials,
        rstate=np.random.default_rng(42)
    )
    
    optimization_time = time.time() - start_time
    
    best_params = {}
    for param, value in best.items():
        if model_name in choice_mappings and param in choice_mappings[model_name]:
            options = choice_mappings[model_name][param]
            if isinstance(value, (int, np.integer)) and 0 <= value < len(options):
                best_params[param] = options[int(value)]
            else:
                best_params[param] = value
        elif param in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 
                       'n_neighbors', 'hidden_layer_size']:
            best_params[param] = int(value)
        elif param in ['C', 'gamma', 'learning_rate', 'subsample', 'alpha']:
            best_params[param] = float(value)
        else:
            best_params[param] = value
    
    best_auc = -min(trials.losses())
    
    optimization_history[model_name] = {
        'iterations': n_iterations,
        'time': optimization_time,
        'best_auc': best_auc,
        'all_losses': [float(loss) for loss in trials.losses()],
        'all_aucs': [-float(loss) for loss in trials.losses()]
    }
    
    all_results[model_name] = {
        'best_params': best_params,
        'best_cv_auc': best_auc,
        'optimization_time': optimization_time
    }
    
    print(f"\n最优超参数:")
    for param, value in best_params.items():
        if isinstance(value, float):
            print(f"  {param}: {value:.6f}")
        else:
            print(f"  {param}: {value}")
    print(f"\n最优交叉验证AUC: {best_auc:.4f}")
    print(f"优化时间: {optimization_time:.2f}秒")

print("\n" + "="*70)
print("贝叶斯优化完成！")
print("="*70)

with open(f'{hyperopt_output_dir}/超参数优化结果.json', 'w', encoding='utf-8') as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print(f"\n优化结果已保存到: {hyperopt_output_dir}/超参数优化结果.json")

summary_text = "="*80 + "\n"
summary_text += "7种机器学习模型贝叶斯超参数优化结果\n"
summary_text += "="*80 + "\n\n"
summary_text += f"数据集: elderly_hypertension_60plus_standardized.xlsx\n"
summary_text += f"目标变量: Stroke\n"
summary_text += f"特征数量: {len(selected_features)}\n"
summary_text += f"特征列表: {', '.join(selected_features)}\n"
summary_text += f"训练集大小: {X_train.shape[0]}\n"
summary_text += f"测试集大小: {X_test.shape[0]}\n"
summary_text += f"中风患病率: {y.sum()/len(y)*100:.2f}%\n"
summary_text += f"优化方法: TPE (Tree-structured Parzen Estimator)\n"
summary_text += f"交叉验证: 5折 Stratified K-Fold\n"
summary_text += f"迭代次数: {n_iterations} 次/模型\n\n"
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
summary_text += f"1. {hyperopt_output_dir}/超参数优化结果.json - JSON格式详细结果\n"
summary_text += f"2. {hyperopt_output_dir}/超参数优化总结.txt - 文本格式总结\n"
summary_text += f"3. {hyperopt_output_dir}/超参数优化过程.png - 优化过程可视化\n"
summary_text += "-"*80 + "\n"

with open(f'{hyperopt_output_dir}/超参数优化总结.txt', 'w', encoding='utf-8') as f:
    f.write(summary_text)

print(f"优化总结已保存到: {hyperopt_output_dir}/超参数优化总结.txt")

plt.figure(figsize=(14, 10))
colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))

for i, (model_name, history) in enumerate(optimization_history.items()):
    iterations = list(range(1, len(history['all_aucs']) + 1))
    plt.plot(iterations, history['all_aucs'], 
             color=colors[i], lw=2, alpha=0.8,
             label=f"{model_name} (Best: {history['best_auc']:.4f})")

plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Cross-Validation AUC', fontsize=12)
plt.title('Bayesian Hyperparameter Optimization Progress\n(7 ML Models for Stroke Prediction)', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=9)
plt.grid(True, alpha=0.3)
plt.ylim(0.5, 1.0)

plt.tight_layout()
plt.savefig(f'{hyperopt_output_dir}/超参数优化过程.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"优化过程图已保存到: {hyperopt_output_dir}/超参数优化过程.png")

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
