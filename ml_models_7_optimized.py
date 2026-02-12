import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_curve, auc, classification_report, confusion_matrix,
                           accuracy_score, precision_score, recall_score, f1_score, brier_score_loss)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import seaborn as sns
import time
import os
import json
import warnings
warnings.filterwarnings('ignore')

output_dir = '8种机器学习'
os.makedirs(output_dir, exist_ok=True)

data = pd.read_excel(r'd:\中风指标\elderly_hypertension_60plus_standardized.xlsx')

selected_features = ['Dyslipidemia', 'PHR', 'Uric_acid', 'MCV', 'Exercise', 'Age', 'AIP']

y = data['Stroke']
X = data[selected_features]

print('='*70)
print('7种机器学习模型比较（中风预测）- 使用优化超参数')
print('='*70)
print(f"\n使用的特征数量: {len(selected_features)}")
print(f"特征列表: {selected_features}")
print(f"X形状: {X.shape}")
print(f"y形状: {y.shape}")
print(f"中风患病率: {y.sum()/len(y)*100:.2f}%")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\n训练集大小: {X_train.shape[0]}")
print(f"测试集大小: {X_test.shape[0]}")
print(f"训练集中中风样本数: {y_train.sum()} ({y_train.sum()/len(y_train)*100:.1f}%)")
print(f"测试集中中风样本数: {y_test.sum()} ({y_test.sum()/len(y_test)*100:.1f}%)")

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
        'params': optimized_params['Random Forest']['best_params'],
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
        'params': optimized_params['XGBoost']['best_params'],
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
        'params': optimized_params['SVM']['best_params'],
        'color': '#2ca02c'
    },
    'kNN': {
        'model': KNeighborsClassifier(
            n_neighbors=optimized_params['kNN']['best_params']['n_neighbors'],
            weights=optimized_params['kNN']['best_params']['weights'],
            metric=optimized_params['kNN']['best_params']['metric']
        ),
        'params': optimized_params['kNN']['best_params'],
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
        'params': optimized_params['Gradient Boosting']['best_params'],
        'color': '#9467bd'
    },
    'AdaBoost': {
        'model': AdaBoostClassifier(
            n_estimators=optimized_params['AdaBoost']['best_params']['n_estimators'],
            learning_rate=optimized_params['AdaBoost']['best_params']['learning_rate'],
            random_state=42
        ),
        'params': optimized_params['AdaBoost']['best_params'],
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
        'params': optimized_params['Neural Network']['best_params'],
        'color': '#e377c2'
    }
}

results = {}
roc_data = {}

print("\n" + "="*70)
print("开始模型训练 (使用SMOTE处理类别不平衡)...")
print("="*70)

for model_name, model_info in models.items():
    print(f"\n{'='*50}")
    print(f"训练模型: {model_name}")
    print(f"{'='*50}")
    
    print(f"\n使用的优化超参数:")
    for param, value in model_info['params'].items():
        if isinstance(value, float):
            print(f"  {param}: {value:.6f}")
        else:
            print(f"  {param}: {value}")
    
    start_time = time.time()
    
    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('model', model_info['model'])
    ])
    
    pipeline.fit(X_train, y_train)
    
    y_pred_prob = pipeline.predict_proba(X_test)[:, 1]
    
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    y_pred = (y_pred_prob >= optimal_threshold).astype(int)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    brier_score = brier_score_loss(y_test, y_pred_prob)
    
    training_time = time.time() - start_time
    
    results[model_name] = {
        'auc': roc_auc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'brier_score': brier_score,
        'training_time': training_time,
        'optimal_threshold': optimal_threshold,
        'y_pred_prob': y_pred_prob,
        'y_pred': y_pred,
        'params': model_info['params']
    }
    
    roc_data[model_name] = {'fpr': fpr, 'tpr': tpr}
    
    print(f"\n  - AUC: {roc_auc:.4f}")
    print(f"  - 准确率: {accuracy:.4f}")
    print(f"  - 精确率: {precision:.4f}")
    print(f"  - 召回率: {recall:.4f}")
    print(f"  - F1分数: {f1:.4f}")
    print(f"  - Brier Score: {brier_score:.4f}")
    print(f"  - 训练时间: {training_time:.2f}秒")
    print(f"  - 最优阈值: {optimal_threshold:.3f}")

performance_df = pd.DataFrame({
    'Model': list(results.keys()),
    'AUC': [results[model]['auc'] for model in results],
    'Accuracy': [results[model]['accuracy'] for model in results],
    'Precision': [results[model]['precision'] for model in results],
    'Recall': [results[model]['recall'] for model in results],
    'F1_Score': [results[model]['f1_score'] for model in results],
    'Brier_Score': [results[model]['brier_score'] for model in results],
    'Training_Time': [results[model]['training_time'] for model in results],
    'Optimal_Threshold': [results[model]['optimal_threshold'] for model in results]
})

performance_df = performance_df.sort_values('AUC', ascending=False)

print("\n" + "="*80)
print("模型性能比较 (按AUC降序排列)")
print("="*80)
print(performance_df.to_string(index=False))

plt.figure(figsize=(14, 10))

for model_name, data in roc_data.items():
    plt.plot(data['fpr'], data['tpr'], 
             color=models[model_name]['color'], lw=2, 
             label=f'{model_name} (AUC = {results[model_name]["auc"]:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves (7 Machine Learning Models)', fontsize=16, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.savefig(f'{output_dir}/all_models_roc_curves_optimized.png', dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(14, 8))

colors_bar = [models[model]['color'] for model in performance_df['Model']]
plt.barh(performance_df['Model'], performance_df['AUC'], color=colors_bar)
plt.xlabel('AUC', fontsize=12)
plt.title('AUC Performance Comparison - 7 ML Models', fontsize=16, fontweight='bold')
plt.xlim(0.5, 1.0)
plt.grid(True, alpha=0.3, axis='x')

for i, v in enumerate(performance_df['AUC']):
    plt.text(v + 0.005, i, f'{v:.3f}', va='center', fontweight='bold')

plt.savefig(f'{output_dir}/models_auc_comparison_optimized.png', dpi=300, bbox_inches='tight')
plt.show()

fig_combined, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

for model_name, data in roc_data.items():
    ax1.plot(data['fpr'], data['tpr'], 
             color=models[model_name]['color'], lw=2, 
             label=f'{model_name} (AUC = {results[model_name]["auc"]:.3f})')

ax1.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.05])
ax1.set_xlabel('False Positive Rate', fontsize=12)
ax1.set_ylabel('True Positive Rate', fontsize=12)
ax1.set_title('ROC Curves (7 Machine Learning Models)', fontsize=14, fontweight='bold')
ax1.legend(loc='lower right', fontsize=9)
ax1.grid(True, alpha=0.3)

colors_bar = [models[model]['color'] for model in performance_df['Model']]
ax2.barh(performance_df['Model'], performance_df['AUC'], color=colors_bar)
ax2.set_xlabel('AUC', fontsize=12)
ax2.set_title('AUC Performance Comparison - 7 ML Models', fontsize=14, fontweight='bold')
ax2.set_xlim(0.5, 1.0)
ax2.grid(True, alpha=0.3, axis='x')

for i, v in enumerate(performance_df['AUC']):
    ax2.text(v + 0.005, i, f'{v:.3f}', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig(f'{output_dir}/roc_auc_combined.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n组合图形已保存到: {output_dir}/roc_auc_combined.png")

performance_df.to_csv(f'{output_dir}/models_performance_comparison_optimized.csv', index=False, encoding='utf-8-sig')

with open(f'{output_dir}/detailed_report_optimized.txt', 'w', encoding='utf-8') as f:
    f.write("7种机器学习模型比较报告 - 中风预测（使用优化超参数）\n")
    f.write("="*60 + "\n")
    f.write(f"数据集: elderly_hypertension_60plus_standardized.xlsx\n")
    f.write(f"目标变量: Stroke\n")
    f.write(f"特征数量: {len(selected_features)}\n")
    f.write(f"特征列表: {', '.join(selected_features)}\n")
    f.write(f"训练集大小: {X_train.shape[0]}\n")
    f.write(f"测试集大小: {X_test.shape[0]}\n")
    f.write(f"中风患病率: {y.sum()/len(y)*100:.2f}%\n")
    f.write(f"使用技术: SMOTE处理类别不平衡\n")
    f.write(f"随机种子: 42\n")
    f.write(f"超参数优化方法: RandomizedSearchCV (50次迭代, 5折交叉验证)\n\n")
    f.write("模型性能排名 (按AUC):\n")
    f.write("-"*50 + "\n")
    for i, row in performance_df.iterrows():
        f.write(f"{i+1}. {row['Model']}: AUC={row['AUC']:.4f}, Accuracy={row['Accuracy']:.4f}, F1={row['F1_Score']:.4f}\n")
        f.write(f"   优化超参数: {results[row['Model']]['params']}\n\n")
    f.write("\n生成的文件:\n")
    f.write("-"*30 + "\n")
    f.write("- all_models_roc_curves_optimized.png: 综合ROC曲线\n")
    f.write("- models_auc_comparison_optimized.png: AUC比较条形图\n")
    f.write("- models_performance_comparison_optimized.csv: 性能比较表格\n")
    f.write("- detailed_report_optimized.txt: 详细实验报告\n")

print("\n" + "="*80)
print(f"完成！所有结果已保存到'{output_dir}'文件夹:")
print("="*80)
print("1. all_models_roc_curves_optimized.png - 综合ROC曲线图")
print("2. models_auc_comparison_optimized.png - AUC性能比较条形图")
print("3. roc_auc_combined.png - ROC曲线与AUC比较组合图")
print("4. models_performance_comparison_optimized.csv - 模型性能比较表格")
print("5. detailed_report_optimized.txt - 详细实验报告")

print(f"\n最佳模型: {performance_df.iloc[0]['Model']}")
print(f"最佳AUC: {performance_df.iloc[0]['AUC']:.4f}")
print(f"最佳准确率: {performance_df.iloc[0]['Accuracy']:.4f}")
print(f"最佳F1分数: {performance_df.iloc[0]['F1_Score']:.4f}")
print(f"最佳超参数: {results[performance_df.iloc[0]['Model']]['params']}")

print("\n" + "="*80)
