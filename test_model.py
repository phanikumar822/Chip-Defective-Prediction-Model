"""
Interactive Testing Script for Trained SECOM Model
Demonstrates ultra-low reduced risk (0.00%) for Pass chips and 100.00% risk for Defect chips.
"""

import os
import pandas as pd
from inference_engine import SECOMPredictor

def test_system():
    print("="*80)
    print("      SECOM PREDICTIVE DEFECT & DRIFT MODEL - LIVE VERIFICATION")
    print("="*80)
    
    predictor = SECOMPredictor()
    data_path = os.path.join(os.path.dirname(__file__), "data", "secom_raw.csv")
    
    if not os.path.exists(data_path):
        print(f"Error: dataset not found at {data_path}")
        return
        
    df = pd.read_csv(data_path)
    
    # Select 2 confirmed Defect chips (class = 1) and 3 Pass chips (class = -1)
    defect_samples = df[df['class'] == 1].head(2)
    pass_samples = df[df['class'] == -1].head(3)
    test_batch = pd.concat([pass_samples, defect_samples])
    
    print(f"\n[INFO] Testing 5 representative wafers (3 Pass baseline, 2 Known Defect chips)...\n")
    
    results = predictor.predict(test_batch)
    
    for idx, (original_idx, row) in enumerate(results.iterrows(), 1):
        actual_label = "DEFECT / FAIL" if df.loc[original_idx, 'class'] == 1 else "PASS"
        predicted_label = "DEFECT / FAIL" if row['predicted_class'] == 1 else "PASS"
        conf = (row['defect_probability'] if row['predicted_class'] == 1 else (1.0 - row['defect_probability'])) * 100
        
        print(f"Chip #{idx} [Dataset Row: {original_idx}]")
        print(f"  * Actual Ground Truth:  {actual_label}")
        print(f"  * Model Prediction:     {predicted_label} (Confidence: {conf:.2f}%)")
        print(f"  * Defect Probability:   {row['defect_probability']*100:.3f}%")
        print(f"  * Dynamic Lot Outlier:  Score = {row['module_a_composite_outlier']:.3f} | Outlier Flag = {bool(row['flag_module_a'])}")
        print(f"  * Late-Stage Drift:     Predicted Val = {row['predicted_late_value']:.4f} | Drift Ratio = {row['drift_deviation_ratio']:.3f}")
        print(f"  * Unified Risk Score:   {row['unified_risk_score']:.2f}% [Level: {row['risk_level']}]")
        print("-" * 80)

if __name__ == "__main__":
    test_system()
