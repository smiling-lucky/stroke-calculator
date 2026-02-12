import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import json
import os
import warnings
warnings.filterwarnings('ignore')

# 确保输出目录存在
output_dir = '8种机器学习'
os.makedirs(output_dir, exist_ok=True)

# 设置字体
plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 加载数据
print("加载数据...")
data = pd.read_excel(r'd:\中风指标\elderly_hypertension_60plus_standardized.xlsx')
selected_features = ['Dyslipidemia', 'PHR', 'Uric_acid', 'MCV', 'Exercise', 'Age', 'AIP']
y = data['Stroke']
X = data[selected_features]

# 划分数据集 - 使用与校准脚本相同的划分比例和随机种子
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 2. 加载优化后的超参数
print("加载超参数...")
with open(r'd:\中风指标\随机搜索优化结果\超参数优化结果.json', 'r', encoding='utf-8') as f:
    optimized_params = json.load(f)

# 3. 定义模型
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

# 4. 定义 DCA 计算函数
def calculate_net_benefit(y_true, y_prob, thresholds):
    net_benefit = []
    n = len(y_true)
    for pt in thresholds:
        if pt == 0: pt = 0.001
        if pt == 1: pt = 0.999
            
        tp = np.sum((y_prob >= pt) & (y_true == 1))
        fp = np.sum((y_prob >= pt) & (y_true == 0))
        
        nb = (tp / n) - (fp / n) * (pt / (1 - pt))
        net_benefit.append(nb)
    return net_benefit

# 5. 训练模型、进行 Isotonic 校准并计算净获益
print("\n训练模型、Isotonic 校准并计算净获益 (DCA)...")
thresholds = np.arange(0, 1.0, 0.01)
dca_results = {}

# 计算 "Treat All" 和 "Treat None" 的净获益
prevalence = np.mean(y_test)
nb_all = []
for pt in thresholds:
    if pt == 0: pt = 0.001
    if pt == 1: pt = 0.999
    tp_all = np.sum(y_test == 1)
    fp_all = np.sum(y_test == 0)
    nb = (tp_all / len(y_test)) - (fp_all / len(y_test)) * (pt / (1 - pt))
    nb_all.append(nb)

nb_none = np.zeros(len(thresholds))

summary_data = []

for model_name, config in models_config.items():
    print(f"处理模型: {model_name}")
    
    # 基础模型 Pipeline
    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('model', config['model'])
    ])
    
    # Isotonic 校准
    calibrated_model = CalibratedClassifierCV(
        estimator=pipeline,
        method='isotonic',
        cv=5
    )
    
    calibrated_model.fit(X_train, y_train)
    y_prob = calibrated_model.predict_proba(X_test)[:, 1]
    
    # 计算 DCA
    nb = calculate_net_benefit(y_test, y_prob, thresholds)
    dca_results[model_name] = {
        'nb': nb,
        'color': config['color']
    }
    
    max_nb = max(nb)
    opt_threshold = thresholds[np.argmax(nb)]
    summary_data.append({
        'Model': model_name,
        'Max_Net_Benefit': max_nb,
        'Optimal_Threshold': opt_threshold
    })

# 6. 绘制 DCA 曲线 (Isotonic 校准后)
print("\n绘制 Isotonic 校准后的 DCA 曲线...")
plt.figure(figsize=(12, 8))

# 绘制模型的 DCA 曲线
for model_name, data in dca_results.items():
    plt.plot(thresholds, data['nb'], color=data['color'], lw=2, label=model_name)

# 绘制 "Treat All" 和 "Treat None"
plt.plot(thresholds, nb_all, color='black', linestyle='--', lw=1.5, label='Treat All')
plt.plot(thresholds, nb_none, color='gray', linestyle='-', lw=1.5, label='Treat None')

# 设置 y 轴范围
plt.ylim(-0.05, prevalence + 0.05)
plt.xlim(0, 1)

plt.xlabel('Threshold Probability', fontsize=12, fontweight='bold')
plt.ylabel('Net Benefit', fontsize=12, fontweight='bold')
plt.title('Decision Curve Analysis (DCA) - Isotonic Calibrated Models\n(Stroke Prediction)', fontsize=16, fontweight='bold', pad=20)

# 关键：将图例移动到右上方，避免遮挡曲线
plt.legend(loc='upper right', fontsize=10, framealpha=0.8)
plt.grid(True, alpha=0.3)

plt.tight_layout()
save_path = f'{output_dir}/dca_comparison_isotonic.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"\nDCA 曲线 (Isotonic) 已保存到: {save_path}")

# 保存总结数据
summary_df = pd.DataFrame(summary_data).sort_values('Max_Net_Benefit', ascending=False)
summary_df.to_csv(f'{output_dir}/dca_summary_isotonic.csv', index=False, encoding='utf-8-sig')
print(f"DCA 总结数据已保存到: {output_dir}/dca_summary_isotonic.csv")
