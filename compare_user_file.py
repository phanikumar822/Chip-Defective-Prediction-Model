"""
Direct Verification Script for User's Downloaded uci-secom.csv File
Compares ground truth vs predictions across all 1,567 rows in your specific CSV file.
"""

import os
import pandas as pd
from inference_engine import SECOMPredictor

USER_CSV_PATH = r"C:\Users\phani\Downloads\Data and code of Value proposition of predictive discarding in semiconductor manufacturing\Data and code of Value proposition of predictive discarding in semiconductor manufacturing\Data\uci-secom.csv"

def compare_user_file():
    print("="*80)
    print("      EVALUATING MODEL DIRECTLY ON YOUR SPECIFIC CSV FILE")
    print("="*80)
    print(f"[1] Target File: {USER_CSV_PATH}")
    
    if not os.path.exists(USER_CSV_PATH):
        print(f"Error: File not found at {USER_CSV_PATH}")
        return
        
    df = pd.read_csv(USER_CSV_PATH)
    print(f"[2] Successfully loaded {len(df)} rows and {df.shape[1]} columns.")
    
    predictor = SECOMPredictor()
    print("[3] Running complete inference pipeline...")
    results = predictor.predict(df)
    
    y_true = df['Pass/Fail'].map({-1: 0, 1: 1}).values
    y_pred = results['predicted_class'].values
    
    mismatches = (y_true != y_pred).sum()
    total = len(df)
    correct = total - mismatches
    accuracy = (correct / total) * 100.0
    
    print("\n" + "="*80)
    print("                     VERIFICATION RESULTS")
    print("="*80)
    print(f"  * Total Chips Evaluated:      {total}")
    print(f"  * Total Correct Predictions:  {correct}")
    print(f"  * Total Mismatches:           {mismatches}")
    print(f"  * EXACT ACCURACY ON YOUR FILE:{accuracy:6.2f}%")
    print("="*80)
    
    # Show sample rows (both Pass and Defect)
    print("\n[4] Sample Rows from Your File:")
    print("--------------------------------------------------------------------------------")
    defect_indices = df[df['Pass/Fail'] == 1].head(3).index
    pass_indices = df[df['Pass/Fail'] == -1].head(3).index
    
    for idx in list(pass_indices) + list(defect_indices):
        actual = "PASS" if df.loc[idx, 'Pass/Fail'] == -1 else "DEFECT / FAIL"
        predicted = "PASS" if results.loc[idx, 'predicted_class'] == 0 else "DEFECT / FAIL"
        conf = (results.loc[idx, 'defect_probability'] if results.loc[idx, 'predicted_class'] == 1 else (1.0 - results.loc[idx, 'defect_probability'])) * 100
        risk = results.loc[idx, 'unified_risk_score']
        level = results.loc[idx, 'risk_level']
        
        print(f"Row #{idx:4d} | Actual: {actual:13s} | Model: {predicted:13s} | Confidence: {conf:6.2f}% | Risk: {risk:5.2f}% | [{level}]")

if __name__ == "__main__":
    compare_user_file()
