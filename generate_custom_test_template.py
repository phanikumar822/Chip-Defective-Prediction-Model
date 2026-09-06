"""
Custom Test Data Generator & Template Builder
Creates a sample template CSV with properly structured sensor columns for manual editing.
"""

import os
import pandas as pd

def create_template():
    data_path = os.path.join(os.path.dirname(__file__), "data", "secom_raw.csv")
    df = pd.read_csv(data_path)
    
    # Take 2 Pass chips and 2 Defect chips as sample rows to serve as an easily editable template
    sample_template = pd.concat([
        df[df['class'] == -1].head(2),
        df[df['class'] == 1].head(2)
    ])
    
    # Rename columns to standard 0..589 format for clean Excel editing
    rename_dict = {'timestamp': 'Time', 'class': 'Pass/Fail'}
    for i in range(1, 591):
        rename_dict[f'Attribute {i}'] = str(i - 1)
        
    sample_template = sample_template.rename(columns=rename_dict)
    
    output_path = os.path.join(os.path.dirname(__file__), "custom_test_template.csv")
    sample_template.to_csv(output_path, index=False)
    print(f"[SUCCESS] Template created at: {output_path}")
    print(f"Total Columns: {sample_template.shape[1]}")
    print(f"Header preview: {sample_template.columns[:6].tolist()} ... {sample_template.columns[-4:].tolist()}")

if __name__ == "__main__":
    create_template()
