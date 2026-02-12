import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_curve, auc, classification_report, confusion_matrix,
                           accuracy_score, precision_score, recall_score, f1_score, brier_score_loss)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import seaborn as sns
import time
import os
import warnings
warnings.filterwarnings('ignore')

# 确保输出目录存在
os.makedirs('结果优化', exist_ok=True)

# 设置字体
plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
data = pd.read_excel('插补后01_已标准化.xlsx')

# 读取LASSO筛选的变量
lasso_features = pd.read_csv('基于lasso/lasso_selected_features.csv')
selected_features = lasso_features['Variable'].tolist()

# 准备数据
y = data['Stones_binary']
X = data[selected_features]

print(f"使用的特征数量: {len(selected_features)}")
print(f"特征列表: {selected_features}")
print(f"X形状: {X.shape}")
print(f"y形状: {y.shape}")
print(f"阳性样本比例: {y.sum()/len(y)*100:.2f}%")

# 划分训练集和测试集 (70%训练, 30%测试)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n训练集大小: {X_train.shape[0]}")
print(f"测试集大小: {X_test.shape[0]}")
print(f"训练集中阳性样本数: {y_train.sum()} ({y_train.sum()/len(y_train)*100:.1f}%)")
print(f"测试集中阳性样本数: {y_test.sum()} ({y_test.sum()/len(y_test)*100:.1f}%)")

# 定义8种机器学习模型（带优化参数）
models = {
    '随机森林 (Random Forest)': {
        'model': RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42),
        'params': {'n_estimators': 150, 'max_depth': 10}
    },
    'XGBoost': {
        'model': XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1, random_state=42),
        'params': {'n_estimators': 150, 'max_depth': 6, 'learning_rate': 0.1}
    },
    '支持向量机 (SVM)': {
        'model': SVC(C=1.0, gamma='scale', probability=True, random_state=42),
        'params': {'C': 1.0, 'gamma': 'scale'}
    },
    'K近邻 (kNN)': {
        'model': KNeighborsClassifier(n_neighbors=7, weights='distance'),
        'params': {'n_neighbors': 7, 'weights': 'distance'}
    },
    '梯度提升 (Gradient Boosting)': {
        'model': GradientBoostingClassifier(n_estimators=150, max_depth=6, random_state=42),
        'params': {'n_estimators': 150, 'max_depth': 6}
    },
    'AdaBoost': {
        'model': AdaBoostClassifier(n_estimators=100, random_state=42),
        'params': {'n_estimators': 100}
    },
    '神经网络 (Neural Network)': {
        'model': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000),
        'params': {'hidden_layer_sizes': (100, 50)}
    },
    '朴素贝叶斯 (Naive Bayes)': {
        'model': GaussianNB(),
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
    pipeline.fit(X_train_scaled, y_train)
    
    # 预测概率
    y_pred_prob = pipeline.predict_proba(X_test_scaled)[:, 1]
    
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
    '模型': list(results.keys()),
    'AUC': [results[model]['auc'] for model in results],
    '准确率': [results[model]['accuracy'] for model in results],
    '精确率': [results[model]['precision'] for model in results],
    '召回率': [results[model]['recall'] for model in results],
    'F1分数': [results[model]['f1_score'] for model in results],
    'Brier Score': [results[model]['brier_score'] for model in results],
    '训练时间(秒)': [results[model]['training_time'] for model in results],
    '最优阈值': [results[model]['optimal_threshold'] for model in results]
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

plt.plot([0, 1], [0, 1], 'k--', lw=2, label='随机猜测')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假阳性率 (False Positive Rate)', fontsize=12)
plt.ylabel('真阳性率 (True Positive Rate)', fontsize=12)
plt.title('8种机器学习模型ROC曲线比较 (SMOTE处理类别不平衡)', fontsize=16, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.savefig('结果优化/all_models_roc_curves.png', dpi=300, bbox_inches='tight')
plt.show()

# 绘制AUC比较条形图
plt.figure(figsize=(16, 10))
plt.barh(performance_df['模型'], performance_df['AUC'], color=plt.cm.viridis(performance_df['AUC']))
plt.xlabel('AUC值', fontsize=12)
plt.title('8种机器学习模型AUC性能比较 (SMOTE处理类别不平衡)', fontsize=16, fontweight='bold')
plt.xlim(0.5, 1.0)
plt.grid(True, alpha=0.3, axis='x')

# 在条形上添加AUC值
for i, v in enumerate(performance_df['AUC']):
    plt.text(v + 0.005, i, f'{v:.3f}', va='center', fontweight='bold')

plt.savefig('结果优化/models_auc_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 绘制校准曲线 (2x4布局)
print("\n生成校准曲线...")
fig, axes = plt.subplots(2, 4, figsize=(20, 12))
axes = axes.flatten()

for i, (model_name, data) in enumerate(roc_data.items()):
    if i < len(axes):
        # 绘制ROC曲线
        axes[i].plot(data['fpr'], data['tpr'], 
                    color=plt.cm.Set3(i), lw=2, 
                    label=f'AUC = {results[model_name]["auc"]:.3f}')
        axes[i].plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.7)
        axes[i].set_xlim([0.0, 1.0])
        axes[i].set_ylim([0.0, 1.05])
        axes[i].set_xlabel('假阳性率')
        axes[i].set_ylabel('真阳性率')
        axes[i].set_title(f'{model_name}', fontsize=12, fontweight='bold')
        axes[i].legend(loc='lower right')
        axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.suptitle('各模型ROC曲线 (2x4布局)', fontsize=16, fontweight='bold', y=1.02)
plt.savefig('结果优化/individual_roc_curves_2x4.png', dpi=300, bbox_inches='tight')
plt.show()

# 为每个模型输出详细报告
print("\n" + "="*80)
print("各模型详细分类报告")
print("="*80)

for model_name in results:
    print(f"\n{model_name}:")
    print("-" * len(model_name))
    print("分类报告:")
    print(classification_report(y_test, results[model_name]['y_pred']))
    print(f"Brier Score: {results[model_name]['brier_score']:.4f}")
    
    # 绘制混淆矩阵
    cm = confusion_matrix(y_test, results[model_name]['y_pred'])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['阴性', '阳性'], 
                yticklabels=['阴性', '阳性'])
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title(f'{model_name} - 混淆矩阵')
    plt.savefig(f'结果优化/confusion_matrix_{model_name.split()[0]}.png', dpi=300, bbox_inches='tight')
    plt.close()

# 保存所有结果到CSV文件
all_results = []
for model_name in results:
    for i in range(len(y_test)):
        all_results.append({
            'Model': model_name,
            'True_Label': y_test.iloc[i],
            'Predicted_Probability': results[model_name]['y_pred_prob'][i],
            'Predicted_Label': results[model_name]['y_pred'][i]
        })

results_df = pd.DataFrame(all_results)
performance_df.to_csv('结果优化/models_performance_comparison.csv', index=False, encoding='utf-8-sig')
results_df.to_csv('结果优化/all_models_prediction_results.csv', index=False, encoding='utf-8-sig')

# 保存模型参数
model_params_df = pd.DataFrame({
    'Model': list(models.keys()),
    'Parameters': [str(models[model]['params']) for model in models]
})
model_params_df.to_csv('结果优化/model_parameters.csv', index=False, encoding='utf-8-sig')

# 保存详细报告
with open('结果优化/detailed_report.txt', 'w', encoding='utf-8') as f:
    f.write("机器学习模型优化实验报告\n")
    f.write("="*60 + "\n")
    f.write(f"数据集: 插补后01_已标准化.xlsx\n")
    f.write(f"目标变量: Stones_binary\n")
    f.write(f"特征数量: {len(selected_features)}\n")
    f.write(f"特征列表: {', '.join(selected_features)}\n")
    f.write(f"训练集大小: {X_train.shape[0]}\n")
    f.write(f"测试集大小: {X_test.shape[0]}\n")
    f.write(f"阳性样本比例: {y.sum()/len(y)*100:.2f}%\n")
    f.write(f"使用技术: SMOTE处理类别不平衡\n")
    f.write(f"随机种子: 42\n")
    f.write("\n模型性能排名 (按AUC):\n")
    f.write("-"*50 + "\n")
    for i, row in performance_df.iterrows():
        f.write(f"{i+1}. {row['模型']}: AUC={row['AUC']:.4f}, 准确率={row['准确率']:.4f}, F1={row['F1分数']:.4f}\n")
    f.write("\n生成的文件:\n")
    f.write("-"*30 + "\n")
    f.write("- all_models_roc_curves.png: 综合ROC曲线\n")
    f.write("- models_auc_comparison.png: AUC比较条形图\n")
    f.write("- individual_roc_curves_2x4.png: 2x4布局ROC曲线\n")
    f.write("- confusion_matrix_*.png: 各模型混淆矩阵\n")
    f.write("- models_performance_comparison.csv: 性能比较表格\n")
    f.write("- all_models_prediction_results.csv: 详细预测结果\n")
    f.write("- model_parameters.csv: 模型参数\n")
    f.write("- detailed_report.txt: 详细实验报告\n")

print("\n" + "="*100)
print("优化完成！所有结果已保存到'结果优化'文件夹:")
print("="*100)
print("1. all_models_roc_curves.png - 综合ROC曲线图")
print("2. models_auc_comparison.png - AUC性能比较条形图")
print("3. individual_roc_curves_2x4.png - 2x4布局各模型ROC曲线")
print("4. confusion_matrix_*.png - 各模型混淆矩阵")
print("5. models_performance_comparison.csv - 模型性能比较表格")
print("6. all_models_prediction_results.csv - 详细预测结果")
print("7. model_parameters.csv - 模型参数配置")
print("8. detailed_report.txt - 详细实验报告")

print(f"\n最佳模型: {performance_df.iloc[0]['模型']}")
print(f"最佳AUC: {performance_df.iloc[0]['AUC']:.4f}")
print(f"最佳准确率: {performance_df.iloc[0]['准确率']:.4f}")
print(f"最佳F1分数: {performance_df.iloc[0]['F1分数']:.4f}")

print("\n评估指标包含:")
print("- 准确率 (Accuracy)")
print("- 精确率 (Precision)")
print("- 召回率 (Recall)")
print("- F1分数 (F1 Score)")
print("- Brier Score")
print("- AUC")
print("- 训练时间")
print("- 最优阈值")