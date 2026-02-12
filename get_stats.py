import pandas as pd
import json

df = pd.read_excel(r'd:\中风指标\elderly_hypertension_60plus_imputed.xlsx')
features = ['PHR', 'Uric_acid', 'MCV', 'Age', 'AIP']
stats = df[features].agg(['mean', 'std']).to_dict()

with open('feature_stats.json', 'w') as f:
    json.dump(stats, f, indent=2)

print("Stats saved to feature_stats.json")
