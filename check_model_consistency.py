import joblib
import pandas as pd
import sys

def check_consistency():
    # 1. 加载模型
    model_path = r'd:\中风指标\最优模型.pkl'
    try:
        model = joblib.load(model_path)
        print(f"成功加载模型: {model_path}")
        
        # 尝试从模型中获取特征名称
        # 如果是 Pipeline，通常特征名称在第一步或最后一步
        model_features = None
        if hasattr(model, 'feature_names_in_'):
            model_features = model.feature_names_in_.tolist()
        elif hasattr(model, 'named_steps'):
            # 如果是 Pipeline，检查是否有特征名称
            for name, step in model.named_steps.items():
                if hasattr(step, 'feature_names_in_'):
                    model_features = step.feature_names_in_.tolist()
                    break
        
        # 如果还是没有，可能需要从之前的脚本中推断，或者手动指定
        # 根据之前的脚本，我们知道 selected_features = ['Dyslipidemia', 'PHR', 'Uric_acid', 'MCV', 'Exercise', 'Age', 'AIP']
        expected_features = ['Dyslipidemia', 'PHR', 'Uric_acid', 'MCV', 'Exercise', 'Age', 'AIP']
        
        if model_features:
            print(f"模型中记录的特征名称: {model_features}")
        else:
            print(f"模型未记录特征名称，使用预期特征列表: {expected_features}")
            model_features = expected_features

    except Exception as e:
        print(f"加载模型失败: {e}")
        return

    # 2. 加载验证集表头
    data_path = r'd:\中风指标\验证集_encoded_20260211.xlsx'
    try:
        # 只读取第一行获取列名
        df_head = pd.read_excel(data_path, nrows=0)
        data_columns = df_head.columns.tolist()
        print(f"验证集中的列名: {data_columns}")
    except Exception as e:
        print(f"读取验证集失败: {e}")
        return

    # 3. 比较
    print("\n--- 比较结果 ---")
    missing_in_data = [f for f in model_features if f not in data_columns]
    
    if not missing_in_data:
        print("✅ 一致性检查通过！验证集中包含模型所需的所有变量。")
    else:
        print("❌ 存在不一致！")
        print(f"模型需要但验证集中缺失的变量: {missing_in_data}")
        
    # 检查大小写或空格问题
    for mf in model_features:
        for dc in data_columns:
            if mf.lower().strip() == dc.lower().strip() and mf != dc:
                print(f"⚠️ 注意: 变量 '{mf}' 与 '{dc}' 仅在大小写或空格上有差异。")

if __name__ == "__main__":
    check_consistency()
