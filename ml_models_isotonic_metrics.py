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
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import time
import os
import json
import warnings

warnings.filterwarnings('ignore')

# 1. 配置与准备
output_dir = '8种机器学习'
os.makedirs(output_dir, exist_ok=True)

# 加载数据
data = pd.read_excel(r'd:\中风指标\elderly_hypertension_60plus_standardized.xlsx')
selected_features = ['Dyslipidemia', 'PHR', 'Uric_acid', 'MCV', 'Exercise', 'Age', 'AIP']
y = data['Stroke']
X = data[selected_features]

# 划分数据集 (与校准脚本保持一致)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 加载优化后的超参数
with open(r'd:\中风指标\随机搜索优化结果\超参数优化结果.json', 'r', encoding='utf-8') as f:
    optimized_params = json.load(f)

# 2. 定义模型配置
models_config = {
    'Random Forest': RandomForestClassifier(
        n_estimators=optimized_params['Random Forest']['best_params']['n_estimators'],
        max_depth=optimized_params['Random Forest']['best_params']['max_depth'],
        min_samples_split=optimized_params['Random Forest']['best_params']['min_samples_split'],
        min_samples_leaf=optimized_params['Random Forest']['best_params']['min_samples_leaf'],
        max_features=optimized_params['Random Forest']['best_params']['max_features'],
        class_weight='balanced',
        random_state=42
    ),
    'XGBoost': GradientBoostingClassifier(
        n_estimators=optimized_params['XGBoost']['best_params']['n_estimators'],
        max_depth=optimized_params['XGBoost']['best_params']['max_depth'],
        learning_rate=optimized_params['XGBoost']['best_params']['learning_rate'],
        min_samples_split=optimized_params['XGBoost']['best_params']['min_samples_split'],
        min_samples_leaf=optimized_params['XGBoost']['best_params']['min_samples_leaf'],
        subsample=optimized_params['XGBoost']['best_params']['subsample'],
        random_state=42
    ),
    'SVM': SVC(
        C=optimized_params['SVM']['best_params']['C'],
        gamma=optimized_params['SVM']['best_params']['gamma'],
        kernel=optimized_params['SVM']['best_params']['kernel'],
        probability=True,
        class_weight='balanced',
        random_state=42
    ),
    'kNN': KNeighborsClassifier(
        n_neighbors=optimized_params['kNN']['best_params']['n_neighbors'],
        weights=optimized_params['kNN']['best_params']['weights'],
        metric=optimized_params['kNN']['best_params']['metric']
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=optimized_params['Gradient Boosting']['best_params']['n_estimators'],
        max_depth=optimized_params['Gradient Boosting']['best_params']['max_depth'],
        learning_rate=optimized_params['Gradient Boosting']['best_params']['learning_rate'],
        min_samples_split=optimized_params['Gradient Boosting']['best_params']['min_samples_split'],
        min_samples_leaf=optimized_params['Gradient Boosting']['best_params']['min_samples_leaf'],
        subsample=optimized_params['Gradient Boosting']['best_params']['subsample'],
        random_state=42
    ),
    'AdaBoost': AdaBoostClassifier(
        n_estimators=optimized_params['AdaBoost']['best_params']['n_estimators'],
        learning_rate=optimized_params['AdaBoost']['best_params']['learning_rate'],
        random_state=42
    ),
    'Neural Network': MLPClassifier(
        hidden_layer_sizes=tuple(optimized_params['Neural Network']['best_params']['hidden_layer_sizes']),
        activation=optimized_params['Neural Network']['best_params']['activation'],
        alpha=optimized_params['Neural Network']['best_params']['alpha'],
        learning_rate=optimized_params['Neural Network']['best_params']['learning_rate'],
        max_iter=1000,
        random_state=42
    )
}

# 3. 训练、Isotonic 校准并评估
isotonic_results = []

print("="*70)
print("开始执行 Isotonic 校准及性能评估...")
print("="*70)

for model_name, model in models_config.items():
    print(f"\n正在处理模型: {model_name}...")
    
    # 基础 Pipeline (包含 SMOTE)
    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('model', model)
    ])
    
    # Isotonic 校准
    calibrated_model = CalibratedClassifierCV(
        estimator=pipeline,
        method='isotonic',
        cv=5
    )
    
    start_time = time.time()
    calibrated_model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # 获取校准后的概率
    y_prob = calibrated_model.predict_proba(X_test)[:, 1]
    
    # 计算 ROC 和 AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # 使用约登指数 (Youden's Index) 确定最优阈值
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    # 应用最优阈值进行分类
    y_pred = (y_prob >= optimal_threshold).astype(int)
    
    # 计算各项性能指标
    metrics = {
        'Model': model_name,
        'AUC': roc_auc,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1_Score': f1_score(y_test, y_pred),
        'Brier_Score': brier_score_loss(y_test, y_prob),
        'Optimal_Threshold': optimal_threshold,
        'Training_Time': training_time
    }
    
    isotonic_results.append(metrics)
    print(f"  - AUC: {metrics['AUC']:.4f}, Optimal Threshold: {metrics['Optimal_Threshold']:.4f}")
    print(f"  - Recall: {metrics['Recall']:.4f}, Accuracy: {metrics['Accuracy']:.4f}")

# 4. 保存结果
results_df = pd.DataFrame(isotonic_results).sort_values('AUC', ascending=False)
csv_path = f'{output_dir}/models_performance_comparison_isotonic.csv'
txt_path = f'{output_dir}/detailed_report_isotonic.txt'
results_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

# 生成 TXT 报告
with open(txt_path, 'w', encoding='utf-8') as f:
    f.write("7种机器学习模型 Isotonic 校准性能报告 - 中风预测\n")
    f.write("="*70 + "\n")
    f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"数据集: elderly_hypertension_60plus_standardized.xlsx\n")
    f.write(f"特征数量: {len(selected_features)}\n")
    f.write(f"特征列表: {', '.join(selected_features)}\n")
    f.write(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}\n")
    f.write(f"校准方法: Isotonic Regression\n")
    f.write(f"阈值选择方法: Youden's Index (约登指数)\n")
    f.write("-" * 70 + "\n\n")
    
    f.write("模型性能排名 (按AUC):\n")
    f.write("-" * 50 + "\n")
    for i, row in results_df.iterrows():
        f.write(f"{i+1}. {row['Model']}:\n")
        f.write(f"   - AUC: {row['AUC']:.4f}\n")
        f.write(f"   - Optimal Threshold: {row['Optimal_Threshold']:.4f}\n")
        f.write(f"   - Accuracy: {row['Accuracy']:.4f}\n")
        f.write(f"   - Recall: {row['Recall']:.4f}\n")
        f.write(f"   - Precision: {row['Precision']:.4f}\n")
        f.write(f"   - F1 Score: {row['F1_Score']:.4f}\n")
        f.write(f"   - Brier Score: {row['Brier_Score']:.4f}\n")
        f.write("\n")
    
    f.write("-" * 70 + "\n")
    f.write("报告生成完毕。\n")

print("\n" + "="*70)
print(f"评估完成！结果已保存至:\nCSV: {csv_path}\nTXT: {txt_path}")
print("="*70)
print(results_df.drop('Training_Time', axis=1).to_string(index=False))
