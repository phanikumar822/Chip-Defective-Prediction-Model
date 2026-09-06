"""
Minimal Sensor Tester Script
Allows testing chips with only a few key sensor attributes (missing sensors are auto-filled with baseline medians).
"""

import pandas as pd
from inference_engine import SECOMPredictor

def test_minimal_inputs():
    print("="*75)
    print("     TESTING WITH MINIMAL SENSORS (PARTIAL ATTRIBUTES)")
    print("="*75)
    
    predictor = SECOMPredictor()
    
    # Example 1: Testing with just 1 sensor (Attribute 1 / '0')
    print("\n[1] Testing with only 1 Attribute ('0' / 'Attribute 1'):")
    df_single = pd.DataFrame([{
        'Time': '2024-05-10 12:00:00',
        '0': 3030.93  # Normal baseline reading
    }])
    res_single = predictor.predict(df_single)
    print(f"    Prediction: {('DEFECT' if res_single['predicted_class'].iloc[0] == 1 else 'PASS')} | Confidence: {((1.0 - res_single['defect_probability'].iloc[0])*100):.2f}% | Risk: {res_single['unified_risk_score'].iloc[0]}%")
    
    # Example 2: Testing with just Top 5 Key Sensors
    print("\n[2] Testing with Top 5 Key Sensors ('0', '1', '2', '3', '4'):")
    df_five = pd.DataFrame([{
        'Time': '2024-05-10 12:00:00',
        '0': 3030.93,
        '1': 2564.00,
        '2': 2187.73,
        '3': 1411.12,
        '4': 1.36
    }])
    res_five = predictor.predict(df_five)
    print(f"    Prediction: {('DEFECT' if res_five['predicted_class'].iloc[0] == 1 else 'PASS')} | Confidence: {((1.0 - res_five['defect_probability'].iloc[0])*100):.2f}% | Risk: {res_five['unified_risk_score'].iloc[0]}%")

if __name__ == "__main__":
    test_minimal_inputs()
