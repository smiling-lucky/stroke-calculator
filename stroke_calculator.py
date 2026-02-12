import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json

# Page configuration
st.set_page_config(
    page_title="Stroke Risk Online Calculator",
    page_icon="🏥",
    layout="centered"
)

# Load resources
@st.cache_resource
def load_resources():
    model = joblib.load('最优模型.pkl')
    
    with open('feature_stats.json', 'r') as f:
        stats = json.load(f)
    
    return model, stats

model, stats = load_resources()

# Custom CSS styles
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
    }
    .result-container {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-top: 20px;
    }
    .risk-low { color: #28a745; font-weight: bold; }
    .risk-medium { color: #ffc107; font-weight: bold; }
    .risk-high { color: #dc3545; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# Page header
st.title("🏥 Stroke Risk Online Calculator")
st.markdown("---")
st.markdown("Please enter your physiological indicators below. The system will estimate your risk of stroke based on our machine learning model.")

# Create input form
with st.container():
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=60, max_value=110, value=70, help="Please enter your age (60+ years)")
        phr = st.number_input("Platelet-to-HDL Ratio (PHR)", min_value=0.0, max_value=100.0, value=4.5, format="%.2f", help="Calculated as Platelet count (10^9/L) / HDL-C (mg/dL)")
        uric_acid = st.number_input("Uric Acid (mg/dL)", min_value=1.0, max_value=15.0, value=4.7, format="%.2f")
        mcv = st.number_input("Mean Corpuscular Volume (MCV, fL)", min_value=20.0, max_value=150.0, value=91.0, format="%.1f")
        
    with col2:
        aip = st.number_input("Atherogenic Index of Plasma (AIP)", min_value=-1.0, max_value=3.0, value=0.4, format="%.3f", help="Calculated as log10(TG/HDL-C) using mg/dL units")
        dyslipidemia = st.selectbox("History of Dyslipidemia", options=[(0, "No"), (1, "Yes")], format_func=lambda x: x[1])[0]
        exercise = st.selectbox("Regular Physical Exercise", options=[(1, "Yes"), (0, "No")], format_func=lambda x: x[1])[0]

# Prediction logic
if st.button("Calculate Risk Probability"):
    # 1. Prepare input data
    input_data = {
        'Dyslipidemia': dyslipidemia,
        'PHR': phr,
        'Uric_acid': uric_acid,
        'MCV': mcv,
        'Exercise': exercise,
        'Age': age,
        'AIP': aip
    }
    
    # 2. Standardization (for continuous variables only)
    continuous_vars = ['PHR', 'Uric_acid', 'MCV', 'Age', 'AIP']
    X_processed = pd.DataFrame([input_data])
    
    for var in continuous_vars:
        mean = stats[var]['mean']
        std = stats[var]['std']
        X_processed[var] = (X_processed[var] - mean) / std
    
    # 3. Ensure feature order
    feature_order = ['Dyslipidemia', 'PHR', 'Uric_acid', 'MCV', 'Exercise', 'Age', 'AIP']
    X_processed = X_processed[feature_order]
    
    # 4. Model prediction
    prob_final = model.predict_proba(X_processed)[0, 1]
        
    # 5. Display results
    st.markdown("---")
    st.subheader("Evaluation Results")
    
    risk_percent = prob_final * 100
    
    # Risk stratification
    if risk_percent < 7.19:
        risk_level = "Low Risk"
        risk_class = "risk-low"
    else:
        risk_level = "High Risk"
        risk_class = "risk-high"
        
    st.markdown(f"""
        <div class="result-container">
            <h3 style='text-align: center;'>Estimated Stroke Risk: <span class='{risk_class}'>{risk_percent:.2f}%</span></h3>
            <p style='text-align: center; font-size: 1.2em;'>Risk Level: <span class='{risk_class}'>{risk_level}</span></p>
        </div>
    """, unsafe_allow_html=True)
    
    if risk_percent > 7.19:
        st.warning("⚠️ **Warning:** Your estimated risk exceeds 7.19%. It is highly recommended to visit a hospital for a comprehensive medical evaluation.")
        
    # Medical advice
    st.info("""
    **💡 Note:**
    - This calculator is based on a machine learning model for academic research and risk reference only.
    - The results do not replace professional clinical diagnosis by a physician.
    - If your risk assessment is high, please consult a medical professional for a comprehensive examination.
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 0.8em;'>"
    "© 2026 Stroke Risk Assessment System | For Academic Reference Only"
    "</div>", 
    unsafe_allow_html=True
)
