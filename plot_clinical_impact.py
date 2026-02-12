import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import json
import os
import warnings
warnings.filterwarnings('ignore')

# 1. 确保输出目录存在
output_dir = '临床影响曲线'
os.makedirs(output_dir, exist_ok=True)

# 设置字体
plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 2. 加载数据
print("加载数据...")
data = pd.read_excel(r'd:\中风指标\elderly_hypertension_60plus_standardized.xlsx')
selected_features = ['Dyslipidemia', 'PHR', 'Uric_acid', 'MCV', 'Exercise', 'Age', 'AIP']
y = data['Stroke']
X = data[selected_features]

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 3. 加载优化后的超参数
print("加载超参数...")
with open(r'd:\中风指标\随机搜索优化结果\超参数优化结果.json', 'r', encoding='utf-8') as f:
    optimized_params = json.load(f)

# 4. 定义模型
models_config = {
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

def plot_clinical_impact_curve(y_true, y_prob, model_name, color, output_dir):
    thresholds = np.arange(0, 1.01, 0.01)
    n_samples = len(y_true)
    
    n_high_risk = []
    n_true_positives = []
    
    for pt in thresholds:
        # 预测为高风险的人数 (y_prob >= pt)
        is_high_risk = (y_prob >= pt)
        high_risk_count = np.sum(is_high_risk)
        
        # 真正的高风险人数 (y_prob >= pt 且 y_true == 1)
        true_positive_count = np.sum(is_high_risk & (y_true == 1))
        
        # 转换成每 1000 人的比例
        n_high_risk.append(high_risk_count / n_samples * 1000)
        n_true_positives.append(true_positive_count / n_samples * 1000)
    
    plt.figure(figsize=(10, 7))
    plt.plot(thresholds, n_high_risk, color='red', lw=2, label='Number high risk')
    plt.plot(thresholds, n_true_positives, color='blue', linestyle='--', lw=2, label='Number high risk with outcome')
    
    plt.xlim(0, 1)
    plt.ylim(0, 1000)
    plt.xlabel('Threshold Probability', fontsize=12, fontweight='bold')
    plt.ylabel('Number per 1000 people', fontsize=12, fontweight='bold')
    plt.title(f'Clinical Impact Curve - {model_name}', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    save_path = f'{output_dir}/CIC_{model_name.replace(" ", "_")}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  - 已保存: {save_path}")

# 5. 循环处理每个模型
print("\n开始生成临床影响曲线...")
for model_name, config in models_config.items():
    print(f"处理模型: {model_name}")
    
    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('model', config['model'])
    ])
    
    pipeline.fit(X_train, y_train)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    plot_clinical_impact_curve(y_test, y_prob, model_name, config['color'], output_dir)

print("\n" + "="*70)
print(f"所有临床影响曲线已成功生成并保存到 '{output_dir}' 文件夹中。")
print("="*70)
