import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (roc_curve, auc, classification_report, confusion_matrix,
                           accuracy_score, precision_score, recall_score, f1_score, brier_score_loss)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import seaborn as sns
import time
import os
import warnings
warnings.filterwarnings('ignore')

# 确保输出目录存在
output_dir = '8种机器学习'
os.makedirs(output_dir, exist_ok=True)

# 读取数据
data = pd.read_excel(r'd:\中风指标\elderly_hypertension_60plus_standardized.xlsx')

# 使用LASSO筛选的7个变量
selected_features = ['Dyslipidemia', 'PHR', 'Uric_acid', 'MCV', 'Exercise', 'Age', 'AIP']

# 准备数据
y = data['Stroke']
X = data[selected_features]

print('='*70)
print('8种机器学习模型比较（中风预测）')
print('='*70)
print(f"\n使用的特征数量: {len(selected_features)}")
print(f"特征列表: {selected_features}")
print(f"X形状: {X.shape}")
print(f"y形状: {y.shape}")
print(f"中风患病率: {y.sum()/len(y)*100:.2f}%")

# 划分训练集和测试集 (70%训练, 30%测试)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 数据已经标准化，无需重复标准化

print(f"\n训练集大小: {X_train.shape[0]}")
print(f"测试集大小: {X_test.shape[0]}")
print(f"训练集中中风样本数: {y_train.sum()} ({y_train.sum()/len(y_train)*100:.1f}%)")
print(f"测试集中中风样本数: {y_test.sum()} ({y_test.sum()/len(y_test)*100:.1f}%)")

# 定义8种机器学习模型（带优化参数）
models = {
    'Random Forest': {
        'model': RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, class_weight='balanced'),
        'params': {'n_estimators': 150, 'max_depth': 10}
    },
    'XGBoost': {
        'model': GradientBoostingClassifier(n_estimators=150, max_depth=6, random_state=42),
        'params': {'n_estimators': 150, 'max_depth': 6}
    },
    'SVM': {
        'model': SVC(C=1.0, gamma='scale', probability=True, random_state=42, class_weight='balanced'),
        'params': {'C': 1.0, 'gamma': 'scale'}
    },
    'kNN': {
        'model': KNeighborsClassifier(n_neighbors=7, weights='distance'),
        'params': {'n_neighbors': 7, 'weights': 'distance'}
    },
    'Gradient Boosting': {
        'model': GradientBoostingClassifier(n_estimators=150, max_depth=6, random_state=42),
        'params': {'n_estimators': 150, 'max_depth': 6}
    },
    'AdaBoost': {
        'model': AdaBoostClassifier(n_estimators=100, random_state=42),
        'params': {'n_estimators': 100}
    },
    'Neural Network': {
        'model': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000),
        'params': {'hidden_layer_sizes': (100, 50)}
    },
    'Logistic Regression': {
        'model': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
        'params': {}
    }
}

# 存储结果
results = {}
roc_data = {}

print("\n" + "="*70)
print("开始模型训练 (使用SMOTE处理类别不平衡)...")
print("="*70)

# 对每个模型进行训练
for model_name, model_info in models.items():
    print(f"\n{'='*50}")
    print(f"训练模型: {model_name}")
    print(f"{'='*50}")
    
    start_time = time.time()
    
    # 创建带SMOTE的pipeline
    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('model', model_info['model'])
    ])
    
    # 训练模型
    pipeline.fit(X_train, y_train)
    
    # 预测概率
    y_pred_prob = pipeline.predict_proba(X_test)[:, 1]
    
    # 计算ROC曲线
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    # 使用最优阈值进行预测
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    y_pred = (y_pred_prob >= optimal_threshold).astype(int)
    
    # 计算所有评估指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    brier_score = brier_score_loss(y_test, y_pred_prob)
    
    # 计算训练时间
    training_time = time.time() - start_time
    
    # 存储结果
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
        'y_pred': y_pred
    }
    
    roc_data[model_name] = {'fpr': fpr, 'tpr': tpr}
    
    print(f"  - AUC: {roc_auc:.4f}")
    print(f"  - 准确率: {accuracy:.4f}")
    print(f"  - 精确率: {precision:.4f}")
    print(f"  - 召回率: {recall:.4f}")
    print(f"  - F1分数: {f1:.4f}")
    print(f"  - Brier Score: {brier_score:.4f}")
    print(f"  - 训练时间: {training_time:.2f}秒")
    print(f"  - 最优阈值: {optimal_threshold:.3f}")

# 创建性能比较表格
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

# 按AUC排序
performance_df = performance_df.sort_values('AUC', ascending=False)

print("\n" + "="*80)
print("模型性能比较 (按AUC降序排列)")
print("="*80)
print(performance_df.to_string(index=False))

# 绘制综合ROC曲线
plt.figure(figsize=(14, 10))
colors = plt.cm.Set3(np.linspace(0, 1, len(models)))

for i, (model_name, data) in enumerate(roc_data.items()):
    plt.plot(data['fpr'], data['tpr'], 
             color=colors[i], lw=2, 
             label=f'{model_name} (AUC = {results[model_name]["auc"]:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves - 8 Machine Learning Models\n(Stroke Prediction)', fontsize=16, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.savefig(f'{output_dir}/all_models_roc_curves.png', dpi=300, bbox_inches='tight')
plt.show()

# 绘制AUC比较条形图
plt.figure(figsize=(14, 8))
colors_bar = plt.cm.viridis(performance_df['AUC'])
plt.barh(performance_df['Model'], performance_df['AUC'], color=colors_bar)
plt.xlabel('AUC', fontsize=12)
plt.title('AUC Performance Comparison - 8 ML Models', fontsize=16, fontweight='bold')
plt.xlim(0.5, 1.0)
plt.grid(True, alpha=0.3, axis='x')

# 在条形上添加AUC值
for i, v in enumerate(performance_df['AUC']):
    plt.text(v + 0.005, i, f'{v:.3f}', va='center', fontweight='bold')

plt.savefig(f'{output_dir}/models_auc_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 保存结果到CSV文件
performance_df.to_csv(f'{output_dir}/models_performance_comparison.csv', index=False, encoding='utf-8-sig')

# 保存详细报告
with open(f'{output_dir}/detailed_report.txt', 'w', encoding='utf-8') as f:
    f.write("8种机器学习模型比较报告 - 中风预测\n")
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
    f.write("\n模型性能排名 (按AUC):\n")
    f.write("-"*50 + "\n")
    for i, row in performance_df.iterrows():
        f.write(f"{i+1}. {row['Model']}: AUC={row['AUC']:.4f}, Accuracy={row['Accuracy']:.4f}, F1={row['F1_Score']:.4f}\n")
    f.write("\n生成的文件:\n")
    f.write("-"*30 + "\n")
    f.write("- all_models_roc_curves.png: 综合ROC曲线\n")
    f.write("- models_auc_comparison.png: AUC比较条形图\n")
    f.write("- models_performance_comparison.csv: 性能比较表格\n")
    f.write("- detailed_report.txt: 详细实验报告\n")

print("\n" + "="*80)
print(f"完成！所有结果已保存到'{output_dir}'文件夹:")
print("="*80)
print("1. all_models_roc_curves.png - 综合ROC曲线图")
print("2. models_auc_comparison.png - AUC性能比较条形图")
print("3. models_performance_comparison.csv - 模型性能比较表格")
print("4. detailed_report.txt - 详细实验报告")

print(f"\n最佳模型: {performance_df.iloc[0]['Model']}")
print(f"最佳AUC: {performance_df.iloc[0]['AUC']:.4f}")
print(f"最佳准确率: {performance_df.iloc[0]['Accuracy']:.4f}")
print(f"最佳F1分数: {performance_df.iloc[0]['F1_Score']:.4f}")

print("\n" + "="*80)
