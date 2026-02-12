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
output_dir = '校准优化结果'
os.makedirs(output_dir, exist_ok=True)

# 设置字体
plt.rcParams['font.sans-serif'] = ['Arial', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 2. 加载数据与模型
print("正在加载数据和模型...")
val_data_path = r'd:\中风指标\验证集.xlsx'
model_path = r'd:\中风指标\最优模型.pkl'

val_data = pd.read_excel(val_data_path)
selected_features = ['Dyslipidemia', 'PHR', 'Uric_acid', 'MCV', 'Exercise', 'Age', 'AIP']
target = 'Stroke'

# 清理缺失值
val_data = val_data.dropna(subset=selected_features + [target])
X_val = val_data[selected_features]
y_val = val_data[target]

model = joblib.load(model_path)

# 3. 获取原始预测概率
y_probs_raw = model.predict_proba(X_val)[:, 1]

# 4. 执行 Platt Scaling 再校准
print("正在执行 Platt Scaling 再校准...")
# 将概率转换为 Logit 空间
eps = 1e-15
logits = np.log((y_probs_raw + eps) / (1 - y_probs_raw + eps)).reshape(-1, 1)

# 在验证集上拟合一个逻辑回归模型
lr_calibrator = LogisticRegression()
lr_calibrator.fit(logits, y_val)
y_probs_calibrated = lr_calibrator.predict_proba(logits)[:, 1]

# 5. 性能评估
def get_metrics(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    brier = brier_score_loss(y_true, y_prob)
    return roc_auc, brier, fpr, tpr

auc_raw, brier_raw, fpr_raw, tpr_raw = get_metrics(y_val, y_probs_raw)
auc_cal, brier_cal, fpr_cal, tpr_cal = get_metrics(y_val, y_probs_calibrated)

print("-" * 50)
print(f"【原始模型】 AUC: {auc_raw:.4f}, Brier Score: {brier_raw:.4f}")
print(f"【校准后】   AUC: {auc_cal:.4f}, Brier Score: {brier_cal:.4f}")
print("-" * 50)

# 6. 绘制对比图
print("正在生成对比图表...")

# (1) ROC 曲线对比
plt.figure(figsize=(8, 7))
plt.plot(fpr_raw, tpr_raw, color='#ff7f0e', lw=2, label=f'Original (AUC = {auc_raw:.3f})')
plt.plot(fpr_cal, tpr_cal, color='#2ca02c', lw=2, linestyle='--', label=f'Recalibrated (AUC = {auc_cal:.3f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.savefig(f'{output_dir}/1_ROC_Comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# (2) 校准曲线对比
plt.figure(figsize=(8, 7))
prob_true_raw, prob_pred_raw = calibration_curve(y_val, y_probs_raw, n_bins=10)
prob_true_cal, prob_pred_cal = calibration_curve(y_val, y_probs_calibrated, n_bins=10)

plt.plot(prob_pred_raw, prob_true_raw, 's-', color='#ff7f0e', label=f'Original (Brier: {brier_raw:.4f})')
plt.plot(prob_pred_cal, prob_true_cal, 's-', color='#2ca02c', label=f'Recalibrated (Brier: {brier_cal:.4f})')
plt.plot([0, 1], [0, 1], '--', color='gray', label='Perfectly Calibrated')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.title('Calibration Curve Comparison')
plt.legend(loc='upper left')
plt.grid(alpha=0.3)
plt.savefig(f'{output_dir}/2_Calibration_Comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# (3) DCA 曲线对比 (可选，展示对临床决策的影响)
def calculate_net_benefit(y_true, y_probs, thresholds):
    n = len(y_true)
    net_benefits = []
    for pt in thresholds:
        y_pred = (y_probs >= pt).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        if pt == 1:
            nb = 0
        else:
            nb = (tp / n) - (fp / n) * (pt / (1 - pt))
        net_benefits.append(nb)
    return net_benefits

thresholds = np.arange(0, 1.01, 0.01)
nb_raw = calculate_net_benefit(y_val, y_probs_raw, thresholds)
nb_cal = calculate_net_benefit(y_val, y_probs_calibrated, thresholds)

plt.figure(figsize=(8, 7))
plt.plot(thresholds, nb_raw, color='#ff7f0e', lw=2, label='Original Model')
plt.plot(thresholds, nb_cal, color='#2ca02c', lw=2, linestyle='--', label='Recalibrated Model')
plt.plot(thresholds, [0]*len(thresholds), color='gray', lw=1, label='None')
plt.xlabel('Threshold Probability')
plt.ylabel('Net Benefit')
plt.title('DCA Comparison')
plt.ylim([-0.05, max(max(nb_raw), max(nb_cal)) + 0.05])
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(f'{output_dir}/3_DCA_Comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"\n分析完成！所有结果已保存至: {output_dir}")
