"""
SECOM Data Acquisition and Local Caching Script
Fetches SECOM dataset directly from UCI Repository and stores locally as CSV.
"""

import os
import sys
import pandas as pd
from ucimlrepo import fetch_ucirepo

def download_and_cache():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    csv_path = os.path.join(data_dir, "secom_raw.csv")
    
    if os.path.exists(csv_path):
        print(f"[INFO] Dataset already cached at {csv_path}")
        df = pd.read_csv(csv_path)
        return df

    print("[INFO] Fetching SECOM dataset (ID 179) from UCI Machine Learning Repository...")
    secom = fetch_ucirepo(id=179)
    df = secom.data.original
    
    # Save cache
    df.to_csv(csv_path, index=False)
        
    print(f"[SUCCESS] Successfully downloaded and cached SECOM dataset: {df.shape[0]} rows, {df.shape[1]} columns.")
    return df

if __name__ == "__main__":
    df = download_and_cache()
    print("[INFO] Class breakdown:")
    print(df['class'].value_counts())
