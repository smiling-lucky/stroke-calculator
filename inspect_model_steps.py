import joblib
import pandas as pd

def inspect_model():
    model_path = r'd:\中风指标\最优模型.pkl'
    model = joblib.load(model_path)
    print("--- 模型结构 ---")
    print(model)
    
    if hasattr(model, 'steps'):
        print("\n--- 管道步骤 (Pipeline Steps) ---")
        for i, step in enumerate(model.steps):
            print(f"步骤 {i+1}: {step[0]} ({type(step[1]).__name__})")
            if step[0] == 'classifier' or i == len(model.steps) - 1:
                print(f"   参数: {step[1].get_params()}")

inspect_model()
