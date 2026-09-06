"""
Validation Test Script for Target Defect Sensor Vector
Tests the profile [2996.24, 2493.28, 2206.2111, 1009.0430] to confirm it is flagged as DEFECT.
"""

import pandas as pd
from inference_engine import SECOMPredictor

def validate_target_vector():
    print("="*80)
    print("      TESTING KNOWN DEFECT VECTOR: [2996.24, 2493.28, 2206.2111, 1009.0430]")
    print("="*80)
    
    predictor = SECOMPredictor()
    
    # Target defect vector specified by user
    defect_sample = pd.DataFrame([{
        'Time': '2008-07-19 12:00:00',
        '0': 2996.24,
        '1': 2493.28,
        '2': 2206.2111,
        '3': 1009.0430
    }])
    
    results = predictor.predict(defect_sample)
    
    pred_class = results['predicted_class'].iloc[0]
    status = "DEFECT / FAIL (REJECT WAFER)" if pred_class == 1 else "PASS (SAFE TO SHIP)"
    defect_prob = results['defect_probability'].iloc[0] * 100.0
    risk_score = results['unified_risk_score'].iloc[0]
    risk_level = results['risk_level'].iloc[0]
    outlier_score = results['module_a_composite_outlier'].iloc[0]
    drift_pred = results['predicted_late_value'].iloc[0]
    
    print(f"\n[EVALUATION RESULT]")
    print(f"  * Classification:       {status}")
    print(f"  * Failure Probability:  {defect_prob:.2f}%")
    print(f"  * Unified Risk Score:   {risk_score:.2f}% [{risk_level}]")
    print(f"  * Module A Outlier:     Score = {outlier_score:.3f}")
    print(f"  * Module B Drift:       Predicted Late Val = {drift_pred:.4f}")
    print("="*80)
    
    if pred_class == 1 or risk_score >= 50.0:
        print(">>> SUCCESS: Vector [2996.24, 2493.28, 2206.2111, 1009.0430] is CORRECTLY FLAGGED AS DEFECT!")
    else:
        print(">>> WARNING: Vector was flagged as PASS.")

if __name__ == "__main__":
    validate_target_vector()
