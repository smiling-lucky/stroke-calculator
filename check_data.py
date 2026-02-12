import pandas as pd
df = pd.read_excel(r'd:\中风指标\elderly_hypertension_60plus_imputed.xlsx')
print(df[['PHR', 'Uric_acid', 'MCV', 'Age', 'AIP', 'Dyslipidemia', 'Exercise']].head())
print("\nDescriptive statistics:")
print(df[['PHR', 'Uric_acid', 'MCV', 'Age', 'AIP']].describe())
