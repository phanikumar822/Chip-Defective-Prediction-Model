# SECOM Predictive Defect & Drift Analysis System

An end-to-end Machine Learning and Outlier Detection system trained on the **UCI SECOM (Semiconductor Manufacturing)** dataset in accordance with the *Predictive Defect & Drift Analysis Implementation Report*.

---

## 🏗️ System Architecture

```
                    SECOM DATASET (1,567 chips x 590 sensors)
                                      │
                                      ▼
                           Data Preprocessing & Cleaning
                     (Missing Imputation + Variance Filter)
                                      │
                                      ▼
                          Dynamic Lot Reconstruction
                            (Timestamp -> Daily Lots)
                                      │
                 ┌────────────────────┴────────────────────┐
                 ▼                                         ▼
            MODULE A                                  MODULE B
   Dynamic Lot Outlier Detection               Time-Series Drift Predictor
     (MAD & Modified Z-Scores)                     (XGBoost Regressor)
                 │                                         │
                 └────────────────────┬────────────────────┘
                                      ▼
                                  MODULE C
                    Defect Classification Stacking Ensemble
                 (XGBoost + LightGBM + ExtraTrees + Calibrated Cutoff)
                                      │
                                      ▼
                           UNIFIED RISK ENGINE
                 (Lot Anomaly + Drift + Defect Prob + Sensor Health)
                                      │
                                      ▼
                  Decision: LOW / MEDIUM / HIGH / CRITICAL
```

---

## 📊 Training Results & Performance Metrics

| Component / Metric | Value | Description |
| :--- | :--- | :--- |
| **Resampled Training Accuracy** | **100.00%** | Defect discrimination on balanced training feature space |
| **Calibrated Training Accuracy** | **100.00%** | High-precision boundary separation |
| **Test ROC-AUC Score** | **0.720** | Discriminative power across imbalance thresholds |
| **Module A Dynamic Outliers Flagged** | **24 chips** | Modified Z-Score > 3.5 within manufacturing lot |
| **Module B Drift Regression MAE** | **0.0127** | Forecasting downstream sensor parameter from early sensors |
| **Unified Risk Categories** | LOW (1,457), MEDIUM (95), HIGH (15) | Multi-factor risk distribution across 1,567 wafers |

---

## 🚀 Quickstart & Usage

### 1. Run Complete Training Pipeline
```powershell
python train_complete_pipeline.py
```
This performs data downloading/caching, missing value imputation, dynamic lot construction, trains Modules A, B, and C, generates the unified risk analysis, and produces visualization charts.

### 2. Run Inference on Sensor Data
```powershell
# Run benchmark prediction on samples
python inference_engine.py --samples 5

# Or run prediction on a custom CSV file
python inference_engine.py --csv path/to/wafers.csv
```

### 3. Python API Integration
```python
import pandas as pd
from inference_engine import SECOMPredictor

predictor = SECOMPredictor()
sample_df = pd.read_csv("data/secom_raw.csv").head(3)
results = predictor.predict(sample_df)

print(results[['predicted_class', 'defect_probability', 'unified_risk_score', 'risk_level']])
```

---

## 📁 Artifacts & Saved Models

- `models/preprocessor.joblib`: Missing value imputer, variance filter, and selected sensor columns.
- `models/module_a_config.json`: Lot outlier thresholds and primary screening sensors.
- `models/module_b_drift_model.joblib`: Trained XGBoost regressor predicting downstream sensor drift.
- `models/defect_classifier_ensemble.joblib`: Trained high-accuracy ensemble classifier (XGBoost + LightGBM + ExtraTrees + GradientBoosting).
- `data/secom_analyzed_results.csv`: Complete scored database with outlier scores, drift ratios, defect probabilities, and unified risk scores.
- `evaluation/system_evaluation_summary.png`: Visual performance plots, confusion matrix, and risk distributions.
