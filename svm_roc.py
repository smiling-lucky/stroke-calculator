import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_curve, auc, roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# 读取数据
df = pd.read_excel('d:\中风指标\elderly_hypertension_selected_alpha005.xlsx')

print('='*70)
print('支持向量机(SVM)预测模型（基于LASSO筛选的7个变量）')
print('='*70)

# 7个选中的特征
selected_features = ['Dyslipidemia', 'PHR', 'Uric_acid', 'MCV', 'Exercise', 'Age', 'AIP']

# 准备数据
X = df[selected_features]
y = df['Stroke']

print(f'\n特征变量: {selected_features}')
print(f'样本量: {len(y)}')
print(f'中风患病率: {y.mean()*100:.2f}%')

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f'\n训练集: {len(X_train)} 人')
print(f'测试集: {len(X_test)} 人')

# 数据标准化（SVM需要）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 构建SVM模型
print('\n' + '-'*70)
print('训练SVM模型...')
print('-'*70)

svm_model = SVC(
    kernel='rbf',           # 核函数
    C=1.0,                  # 正则化参数
    gamma='scale',          # 核系数
    probability=True,       # 启用概率估计
    class_weight='balanced', # 处理类别不平衡
    random_state=42
)

# 训练模型
svm_model.fit(X_train_scaled, y_train)

# 预测概率
y_train_proba = svm_model.predict_proba(X_train_scaled)[:, 1]
y_test_proba = svm_model.predict_proba(X_test_scaled)[:, 1]

# 计算AUC
train_auc = roc_auc_score(y_train, y_train_proba)
test_auc = roc_auc_score(y_test, y_test_proba)

print(f'\n模型性能:')
print(f'  训练集 AUC: {train_auc:.4f}')
print(f'  测试集 AUC: {test_auc:.4f}')

# 10折交叉验证
print('\n' + '-'*70)
print('10折交叉验证...')
print('-'*70)

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# 需要重新标准化数据进行交叉验证
X_scaled = scaler.fit_transform(X)
cv_scores = cross_val_score(svm_model, X_scaled, y, cv=skf, scoring='roc_auc')

print(f'  10折交叉验证 AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})')
print(f'  各折AUC: {cv_scores.round(4)}')

# 绘制ROC曲线
print('\n' + '-'*70)
print('绘制ROC曲线...')
print('-'*70)

# 计算ROC曲线
fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba)

# 创建图形
plt.figure(figsize=(10, 8))

# 绘制训练集ROC
plt.plot(fpr_train, tpr_train, color='blue', lw=2, 
         label=f'Training (AUC = {train_auc:.3f})')

# 绘制测试集ROC
plt.plot(fpr_test, tpr_test, color='red', lw=2, 
         label=f'Test (AUC = {test_auc:.3f})')

# 绘制对角线
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random')

# 设置图形
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - Support Vector Machine (SVM)\n(7 Features Selected by LASSO)', 
          fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, alpha=0.3)

# 保存ROC曲线
roc_file = r'd:\中风指标\roc_curve_svm.png'
plt.savefig(roc_file, dpi=300, bbox_inches='tight')
print(f'\nROC曲线已保存到: {roc_file}')

plt.show()

# 测试集详细评估
print('\n' + '='*70)
print('测试集详细评估')
print('='*70)

y_test_pred = svm_model.predict(X_test_scaled)
print('\n分类报告:')
print(classification_report(y_test, y_test_pred, target_names=['No Stroke', 'Stroke']))

print('\n混淆矩阵:')
cm = confusion_matrix(y_test, y_test_pred)
print(f'                 预测')
print(f'                 无中风  中风')
print(f'实际  无中风      {cm[0,0]:4d}   {cm[0,1]:4d}')
print(f'      中风        {cm[1,0]:4d}   {cm[1,1]:4d}')

# 计算指标
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn)  # 灵敏度
specificity = tn / (tn + fp)  # 特异度
ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # 阳性预测值
npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # 阴性预测值

print(f'\n评估指标:')
print(f'  灵敏度 (Sensitivity): {sensitivity:.4f}')
print(f'  特异度 (Specificity): {specificity:.4f}')
print(f'  阳性预测值 (PPV): {ppv:.4f}')
print(f'  阴性预测值 (NPV): {npv:.4f}')

# 与随机森林对比
print('\n' + '='*70)
print('模型对比')
print('='*70)
print(f"{'模型':<20}{'训练集AUC':<15}{'测试集AUC':<15}{'10折CV AUC':<15}")
print('-'*65)
print(f"{'Random Forest':<20}{0.9918:<15.4f}{0.6570:<15.4f}{0.6486:<15.4f}")
print(f"{'SVM':<20}{train_auc:<15.4f}{test_auc:<15.4f}{cv_scores.mean():<15.4f}")

print('\n' + '='*70)
