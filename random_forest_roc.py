import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_curve, auc, roc_auc_score, classification_report, confusion_matrix

# 读取数据
df = pd.read_excel('d:\中风指标\elderly_hypertension_selected_alpha005.xlsx')

print('='*70)
print('随机森林预测模型（基于LASSO筛选的7个变量）')
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

# 构建随机森林模型
print('\n' + '-'*70)
print('训练随机森林模型...')
print('-'*70)

rf_model = RandomForestClassifier(
    n_estimators=200,      # 树的数量
    max_depth=10,          # 最大深度
    min_samples_split=10,  # 最小分裂样本数
    min_samples_leaf=5,    # 最小叶子样本数
    random_state=42,
    class_weight='balanced'  # 处理类别不平衡
)

# 训练模型
rf_model.fit(X_train, y_train)

# 预测概率
y_train_proba = rf_model.predict_proba(X_train)[:, 1]
y_test_proba = rf_model.predict_proba(X_test)[:, 1]

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
cv_scores = cross_val_score(rf_model, X, y, cv=skf, scoring='roc_auc')

print(f'  10折交叉验证 AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})')
print(f'  各折AUC: {cv_scores.round(4)}')

# 特征重要性
print('\n' + '='*70)
print('特征重要性排名')
print('='*70)

feature_importance = pd.DataFrame({
    'Feature': selected_features,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\n{'排名':<6}{'变量名':<15}{'重要性':<15}")
print('-'*40)
for idx, (_, row) in enumerate(feature_importance.iterrows(), 1):
    print(f"{idx:<6}{row['Feature']:<15}{row['Importance']:<15.4f}")

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
plt.title('ROC Curve - Random Forest\n(7 Features Selected by LASSO)', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, alpha=0.3)

# 保存ROC曲线
roc_file = r'd:\中风指标\roc_curve_random_forest.png'
plt.savefig(roc_file, dpi=300, bbox_inches='tight')
print(f'\nROC曲线已保存到: {roc_file}')

plt.show()

# 测试集详细评估
print('\n' + '='*70)
print('测试集详细评估')
print('='*70)

y_test_pred = rf_model.predict(X_test)
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

print('\n' + '='*70)
