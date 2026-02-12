import pandas as pd
import numpy as np
import joblib
import json
import os

def find_high_risk_combination():
    # 1. Load resources
    model = joblib.load('最优模型.pkl')
    platt_path = os.path.join('外部验证结果_校准后', 'platt_calibrator.pkl')
    platt_calibrator = joblib.load(platt_path) if os.path.exists(platt_path) else None
    
    with open('feature_stats.json', 'r') as f:
        stats = json.load(f)
        
    results = []
    # Test uncalibrated high risk cases with calibration
    test_cases = [
        {'Age': 80, 'PHR': 5.0, 'Uric_acid': 2.0, 'MCV': 70, 'AIP': 0.5, 'Dyslipidemia': 0, 'Exercise': 1},
        {'Age': 80, 'PHR': 5.0, 'Uric_acid': 2.0, 'MCV': 70, 'AIP': 0.5, 'Dyslipidemia': 0, 'Exercise': 0},
        {'Age': 70, 'PHR': 5.0, 'Uric_acid': 2.0, 'MCV': 90, 'AIP': 0.5, 'Dyslipidemia': 0, 'Exercise': 0},
    ]
    
    for case in test_cases:
        X = pd.DataFrame([case])
        X_std = X.copy()
        for var in ['PHR', 'Uric_acid', 'MCV', 'Age', 'AIP']:
            X_std[var] = (X_std[var] - stats[var]['mean']) / stats[var]['std']
        X_std = X_std[['Dyslipidemia', 'PHR', 'Uric_acid', 'MCV', 'Exercise', 'Age', 'AIP']]
        prob_raw = model.predict_proba(X_std)[0, 1]
        
        if platt_calibrator:
            eps = 1e-15
            logit = np.log((prob_raw + eps) / (1 - prob_raw + eps)).reshape(-1, 1)
            prob_final = platt_calibrator.predict_proba(logit)[0, 1]
        else:
            prob_final = prob_raw
        results.append((case, prob_final, prob_raw))
    return results

if __name__ == "__main__":
    results = find_high_risk_combination()
    for i, (case, prob, raw) in enumerate(results):
        print(f"Case {i+1}: Calibrated Prob={prob:.4f}, Raw Prob={raw:.4f}")
        print(case)
        print("-" * 30)
