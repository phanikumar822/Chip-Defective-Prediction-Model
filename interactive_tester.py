"""
Interactive Console App for SECOM Defect & Risk Predictor
Allows users to manually enter sensor inputs directly via terminal without any CSV file.
"""

import os
import sys
import pandas as pd
from inference_engine import SECOMPredictor

def run_interactive_app():
    predictor = SECOMPredictor()
    
    print("="*80)
    print("      SECOM SEMICONDUCTOR DEFECT PREDICTOR - INTERACTIVE CONSOLE")
    print("="*80)
    print("You can test a chip by entering sensor values directly without a CSV file.\n")
    
    while True:
        print("\nChoose an option:")
        print("  [1] Enter custom sensor numbers manually")
        print("  [2] Quick Test Preset: Standard Normal Chip (Baseline)")
        print("  [3] Quick Test Preset: High Voltage / Defective Chip")
        print("  [4] Quick Test Preset: Machine Drift / Anomaly Chip")
        print("  [5] Exit")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == '5':
            print("Exiting. Thank you!")
            break
            
        input_data = {}
        
        if choice == '1':
            print("\nEnter values for Key Sensors (Press Enter to keep default baseline):")
            sensors_to_ask = [
                ('0', 'Main Process Sensor 0', 3030.93),
                ('1', 'Voltage / Etching Sensor 1', 2564.00),
                ('2', 'Chamber Pressure Sensor 2', 2187.73),
                ('3', 'Gas Flow Sensor 3', 1411.12),
                ('4', 'Wafer Temp Sensor 4', 1.36),
                ('14', 'Intermediate Process Sensor 14', 7.95),
                ('20', 'Late-Stage Sensor 20', 1.40)
            ]
            
            for key, desc, default_val in sensors_to_ask:
                val_str = input(f"  * {desc} [Default: {default_val}]: ").strip()
                if val_str:
                    try:
                        input_data[key] = float(val_str)
                    except ValueError:
                        print(f"    Invalid number. Using default {default_val}")
                        input_data[key] = default_val
                else:
                    input_data[key] = default_val
                    
        elif choice == '2':
            # Normal Baseline
            input_data = {'0': 3030.93, '1': 2564.00, '2': 2187.73, '3': 1411.12, '4': 1.36}
            print("\n[Loaded Normal Baseline Sensor Profile]")
            
        elif choice == '3':
            # Defective Pattern
            input_data = {'0': 2988.72, '1': 2470.38, '2': 2201.21, '3': 1544.43, '4': 1.49, '14': 12.8}
            print("\n[Loaded Known Defective Sensor Profile]")
            
        elif choice == '4':
            # Machine Drift
            input_data = {'0': 3350.00, '1': 2890.00, '2': 2450.00, '3': 1950.00, '4': 2.85}
            print("\n[Loaded Severe Machine Drift / Outlier Profile]")
            
        else:
            print("Invalid selection. Please choose 1-5.")
            continue
            
        # Build DataFrame and run prediction
        df_user = pd.DataFrame([input_data])
        results = predictor.predict(df_user)
        
        pred_class = results['predicted_class'].iloc[0]
        status = "DEFECT / FAIL" if pred_class == 1 else "PASS"
        defect_prob = results['defect_probability'].iloc[0] * 100
        conf = defect_prob if pred_class == 1 else (100.0 - defect_prob)
        outlier_score = results['module_a_composite_outlier'].iloc[0]
        drift_pred = results['predicted_late_value'].iloc[0]
        drift_ratio = results['drift_deviation_ratio'].iloc[0]
        risk_score = results['unified_risk_score'].iloc[0]
        risk_level = results['risk_level'].iloc[0]
        
        print("\n" + "="*80)
        print("                     MODEL PREDICTION RESULT")
        print("="*80)
        print(f"  >>> CHIP STATUS:        {status}")
        print(f"  >>> MODEL CONFIDENCE:   {conf:.2f}%")
        print(f"  >>> DEFECT PROBABILITY: {defect_prob:.3f}%")
        print(f"  >>> UNIFIED RISK SCORE: {risk_score:.2f}% [{risk_level}]")
        print(f"  ----------------------------------------------------------------------")
        print(f"  * Dynamic Lot Outlier Score: {outlier_score:.3f} (Flagged Outlier: {bool(results['flag_module_a'].iloc[0])})")
        print(f"  * Late-Stage Drift Forecast: {drift_pred:.4f} (Drift Ratio: {drift_ratio:.3f})")
        print("="*80)

if __name__ == "__main__":
    run_interactive_app()
