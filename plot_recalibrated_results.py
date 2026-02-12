import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import warnings
from sklearn.metrics import roc_curve, auc, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore')

# 1. 配置与准备
output_dir = '外部验证结果_校准后'
os.makedirs(output_dir, exist_ok=True)

# 设置字体
plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 2. 加载数据与模型
print("加载数据和模型...")
val_data_path = r'd:\中风指标\验证集.xlsx'
model_path = r'd:\中风指标\最优模型.pkl'

val_data = pd.read_excel(val_data_path)
selected_features = ['Dyslipidemia', 'PHR', 'Uric_acid', 'MCV', 'Exercise', 'Age', 'AIP']
target = 'Stroke'

val_data = val_data.dropna(subset=selected_features + [target])
X_val = val_data[selected_features]
y_val = val_data[target]

model = joblib.load(model_path)
y_probs_raw = model.predict_proba(X_val)[:, 1]

# 3. 执行 Platt Scaling 再校准
print("应用校准映射 (Platt Scaling)...")
eps = 1e-15
logits = np.log((y_probs_raw + eps) / (1 - y_probs_raw + eps)).reshape(-1, 1)

lr_calibrator = LogisticRegression()
lr_calibrator.fit(logits, y_val)
joblib.dump(lr_calibrator, os.path.join(output_dir, 'platt_calibrator.pkl'))
print(f"Platt 校准器已保存至: {os.path.join(output_dir, 'platt_calibrator.pkl')}")
y_probs = lr_calibrator.predict_proba(logits)[:, 1]

# 4. 绘图函数定义 (仅显示校准后结果)

def plot_roc(y_true, y_probs, save_path):
    print("生成 ROC 曲线...")
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 7))
    plt.plot(fpr, tpr, color='#9467bd', lw=2, label=f'Gradient Boosting (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - External Validation (Recalibrated)', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_calibration(y_true, y_probs, save_path):
    print("生成校准曲线...")
    prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=10)
    brier = brier_score_loss(y_true, y_probs)
    
    plt.figure(figsize=(8, 7))
    plt.plot(prob_pred, prob_true, marker='s', ls='-', label=f'Gradient Boosting (Brier: {brier:.4f})', color='#9467bd', lw=2)
    plt.plot([0, 1], [0, 1], ls='--', color='black', lw=2, label='Perfectly calibrated')
    plt.xlabel('Mean predicted probability', fontsize=12)
    plt.ylabel('Fraction of positives', fontsize=12)
    plt.title('Calibration Curve - External Validation (Recalibrated)', fontsize=14, fontweight='bold')
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_dca(y_true, y_probs, save_path):
    print("生成 DCA 曲线...")
    thresholds = np.arange(0, 1.01, 0.01)
    net_benefit_model = []
    net_benefit_all = []
    
    n = len(y_true)
    for pt in thresholds:
        y_pred = (y_probs >= pt).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        nb = (tp / n) - (fp / n) * (pt / (1 - pt)) if pt < 1 else 0
        net_benefit_model.append(nb)
        
        tp_all = np.sum(y_true == 1)
        fp_all = np.sum(y_true == 0)
        nb_all = (tp_all / n) - (fp_all / n) * (pt / (1 - pt)) if pt < 1 else 0
        net_benefit_all.append(nb_all)
        
    plt.figure(figsize=(8, 7))
    plt.plot(thresholds, net_benefit_model, color='#9467bd', lw=2, label='Gradient Boosting')
    plt.plot(thresholds, net_benefit_all, color='black', lw=1.5, linestyle='--', label='All')
    plt.plot(thresholds, [0]*len(thresholds), color='gray', lw=1.5, label='None')
    
    y_min = max(min(net_benefit_model) - 0.05, -0.1)
    y_max = max(net_benefit_model) + 0.05
    plt.ylim([y_min, y_max])
    plt.xlim([0, 1])
    plt.xlabel('Threshold Probability', fontsize=12)
    plt.ylabel('Net Benefit', fontsize=12)
    plt.title('Decision Curve Analysis (DCA) - Recalibrated', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_cic(y_true, y_probs, save_path):
    print("生成临床影响曲线 (CIC)...")
    thresholds = np.arange(0, 1.01, 0.01)
    n = len(y_true)
    n_high_risk = []
    n_true_pos = []
    
    for pt in thresholds:
        y_pred = (y_probs >= pt).astype(int)
        n_high = np.sum(y_pred == 1)
        n_tp = np.sum((y_pred == 1) & (y_true == 1))
        n_high_risk.append(n_high / n * 1000)
        n_true_pos.append(n_tp / n * 1000)
        
    plt.figure(figsize=(8, 7))
    plt.plot(thresholds, n_high_risk, color='red', lw=2, label='Number high risk')
    plt.plot(thresholds, n_true_pos, color='blue', lw=2, linestyle='--', label='Number high risk with outcome')
    plt.xlim([0, 1])
    plt.ylim([0, 1000])
    plt.xlabel('Threshold Probability', fontsize=12)
    plt.ylabel('Number per 1000 people', fontsize=12)
    plt.title('Clinical Impact Curve - Recalibrated', fontsize=14)
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# 5. 执行绘图
plot_roc(y_val, y_probs, f'{output_dir}/1_ROC_Recalibrated.png')
plot_calibration(y_val, y_probs, f'{output_dir}/2_Calibration_Recalibrated.png')
plot_dca(y_val, y_probs, f'{output_dir}/3_DCA_Recalibrated.png')
plot_cic(y_val, y_probs, f'{output_dir}/4_CIC_Recalibrated.png')

print(f"\n校准后的分析图表已生成并保存至: {output_dir}")
