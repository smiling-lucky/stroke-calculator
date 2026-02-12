import pandas as pd
import numpy as np

df = pd.read_excel('d:\中风指标\wave2026.02.01.xlsx')

print('='*80)
print('复合变量分析报告')
print('='*80)

# 基础变量（直接测量或收集的）
basic_vars = ['ID', 'Wave', 'Age', 'Education', 'Exercise', 'Gender', 'Marry', 'Rural',
              'Hypertension', 'Diabetes_mellitus', 'Cancre', 'Lunge', 'Hearte', 'Stroke',
              'Dyslipidemia', 'Livere', 'Kidneye', 'Alcohol_consumption', 'Smoken',
              'Height', 'Weight', 'Waist', 'Fasting', 'WBC', 'MCV', 'PLT', 
              'BUN', 'Blood_glucose', 'Scr', 'TC', 'TG', 'HDL-C', 'LDL-C', 
              'CRP', 'HbA1c', 'Uric_acid', 'Hemoglobin', 'Sleep',
              'Hypertension.1', 'Diabetes_mellitus.1', 'Follow_time', 'Event']

# 复合变量（由其他变量计算得到）
composite_vars = ['BMI', 'PHR', 'eGFRabdiff', 'eGFRrediff', 'Tyg', 'Tyg_bmi', 
                  'AIP', 'CTI', 'CMI', 'PP', 'SBP', 'DBP']

print('\n【一、基础变量】（共 {} 个）'.format(len(basic_vars)))
print('-'*60)
for i, var in enumerate(basic_vars, 1):
    print(f'{i:2d}. {var}')

print('\n【二、复合变量】（共 {} 个）'.format(len(composite_vars)))
print('-'*60)

# 1. BMI
print('\n1. BMI (Body Mass Index) - 体重指数')
print('   计算公式: BMI = Weight(kg) / Height(m)²')
print('   依赖变量: Weight, Height')
sample = df[['Weight', 'Height', 'BMI']].dropna().head(3)
for idx, row in sample.iterrows():
    calc_bmi = row['Weight'] / (row['Height']/100)**2
    print(f'   示例: Weight={row["Weight"]}, Height={row["Height"]} => BMI={calc_bmi:.2f} (实际值: {row["BMI"]:.2f})')

# 2. SBP
print('\n2. SBP (Systolic Blood Pressure) - 收缩压')
print('   注: SBP是直接测量值，但常与DBP一起作为血压指标')

# 3. DBP
print('\n3. DBP (Diastolic Blood Pressure) - 舒张压')
print('   注: DBP是直接测量值')

# 4. PP
print('\n4. PP (Pulse Pressure) - 脉压')
print('   计算公式: PP = SBP - DBP')
print('   依赖变量: SBP, DBP')
sample = df[['SBP', 'DBP', 'PP']].dropna().head(3)
for idx, row in sample.iterrows():
    calc_pp = row['SBP'] - row['DBP']
    print(f'   示例: SBP={row["SBP"]}, DBP={row["DBP"]} => PP={calc_pp:.1f} (实际值: {row["PP"]:.1f})')

# 5. PHR
print('\n5. PHR (Pulse Heart Rate) - 脉压心率比')
print('   计算公式: PHR = PP / Heart Rate')
print('   依赖变量: PP, Heart Rate（心率变量未在数据集中明确标识）')

# 6. Tyg
print('\n6. Tyg (Triglyceride-Glucose Index) - 甘油三酯-葡萄糖指数')
print('   计算公式: Tyg = ln[TG(mg/dL) × Fasting_glucose(mg/dL) / 2]')
print('   或: Tyg = ln[TG(mmol/L) × Fasting_glucose(mmol/L)]')
print('   依赖变量: TG, Fasting (或 Blood_glucose)')
sample = df[['TG', 'Fasting', 'Tyg']].dropna().head(3)
for idx, row in sample.iterrows():
    calc_tyg = np.log(row['TG'] * row['Fasting'])
    print(f'   示例: TG={row["TG"]:.2f}, Fasting={row["Fasting"]:.2f} => Tyg≈{calc_tyg:.2f} (实际值: {row["Tyg"]:.2f})')

# 7. Tyg_bmi
print('\n7. Tyg_bmi (TyG-BMI Index) - TyG-BMI指数')
print('   计算公式: Tyg_bmi = Tyg × BMI')
print('   依赖变量: Tyg, BMI')
sample = df[['Tyg', 'BMI', 'Tyg_bmi']].dropna().head(3)
for idx, row in sample.iterrows():
    calc_tyg_bmi = row['Tyg'] * row['BMI']
    print(f'   示例: Tyg={row["Tyg"]:.2f}, BMI={row["BMI"]:.2f} => Tyg_bmi={calc_tyg_bmi:.2f} (实际值: {row["Tyg_bmi"]:.2f})')

# 8. AIP
print('\n8. AIP (Atherogenic Index of Plasma) - 血浆动脉粥样硬化指数')
print('   计算公式: AIP = log(TG / HDL-C)')
print('   依赖变量: TG, HDL-C')
sample = df[['TG', 'HDL-C', 'AIP']].dropna().head(3)
for idx, row in sample.iterrows():
    calc_aip = np.log10(row['TG'] / row['HDL-C'])
    print(f'   示例: TG={row["TG"]:.2f}, HDL-C={row["HDL-C"]:.2f} => AIP={calc_aip:.2f} (实际值: {row["AIP"]:.2f})')

# 9. CTI
print('\n9. CTI (Cardiometabolic Index) - 心脏代谢指数')
print('   计算公式: CTI = (Waist / Height) × (TG / HDL-C)')
print('   依赖变量: Waist, Height, TG, HDL-C')
sample = df[['Waist', 'Height', 'TG', 'HDL-C', 'CTI']].dropna().head(3)
for idx, row in sample.iterrows():
    calc_cti = (row['Waist'] / row['Height']) * (row['TG'] / row['HDL-C'])
    print(f'   示例: Waist={row["Waist"]}, Height={row["Height"]}, TG={row["TG"]:.2f}, HDL-C={row["HDL-C"]:.2f} => CTI={calc_cti:.2f} (实际值: {row["CTI"]:.2f})')

# 10. CMI
print('\n10. CMI (Cardiometabolic Index) - 心脏代谢指数（另一种计算方式）')
print('   计算公式: CMI = (Waist / Height) × (TG / HDL-C)')
print('   依赖变量: Waist, Height, TG, HDL-C')
sample = df[['Waist', 'Height', 'TG', 'HDL-C', 'CMI']].dropna().head(3)
for idx, row in sample.iterrows():
    calc_cmi = (row['Waist'] / row['Height']) * (row['TG'] / row['HDL-C'])
    print(f'   示例: Waist={row["Waist"]}, Height={row["Height"]}, TG={row["TG"]:.2f}, HDL-C={row["HDL-C"]:.2f} => CMI={calc_cmi:.2f} (实际值: {row["CMI"]:.2f})')

# 11. eGFRabdiff
print('\n11. eGFRabdiff (eGFR based on Scr) - 基于血肌酐的估算肾小球滤过率')
print('   计算公式: CKD-EPI公式（基于Scr、Age、Gender）')
print('   依赖变量: Scr, Age, Gender')

# 12. eGFRrediff
print('\n12. eGFRrediff (eGFR based on CysC) - 基于胱抑素C的估算肾小球滤过率')
print('   计算公式: CKD-EPI公式（基于CysC、Age、Gender）')
print('   依赖变量: CysC（胱抑素C，数据集中未明确标识）, Age, Gender')

print('\n' + '='*80)
print('总结：复合变量依赖关系图')
print('='*80)
print('''
基础测量变量:
├── 人体测量: Height, Weight, Waist
├── 血压: SBP, DBP
├── 血脂: TC, TG, HDL-C, LDL-C
├── 血糖: Fasting, Blood_glucose, HbA1c
├── 肾功能: Scr, BUN
└── 其他: Age, Gender

第一层复合变量:
├── BMI = Weight / (Height/100)²
├── PP = SBP - DBP
└── AIP = log(TG / HDL-C)

第二层复合变量:
├── Tyg = ln(TG × Fasting)
├── CTI/CMI = (Waist/Height) × (TG/HDL-C)
└── Tyg_bmi = Tyg × BMI

第三层复合变量:
└── eGFRabdiff/eGFRrediff = CKD-EPI公式 (Scr/CysC, Age, Gender)
''')
