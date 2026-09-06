"""
Inference API & CLI for SECOM Predictive Defect & Drift Analysis System
Incorporates global baseline distribution scoring for robust out-of-distribution anomaly detection.
"""

import os
import json
import warnings
import joblib
import argparse
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")


class SECOMPredictor:
    def __init__(self, models_dir=MODELS_DIR):
        self.models_dir = models_dir
        self.load_models()

    def load_models(self):
        self.preprocessor = joblib.load(os.path.join(self.models_dir, "preprocessor.joblib"))
        self.valid_cols = self.preprocessor['valid_cols']
        self.imputer = self.preprocessor['imputer']
        self.non_constant_mask = self.preprocessor['non_constant_mask']
        self.selected_feature_names = self.preprocessor['selected_feature_names']

        with open(os.path.join(self.models_dir, "module_a_config.json"), "r") as f:
            self.module_a_config = json.load(f)

        self.module_b = joblib.load(os.path.join(self.models_dir, "module_b_drift_model.joblib"))
        self.drift_model = self.module_b['model']
        self.early_features = self.module_b['early_features']
        self.target_feature = self.module_b['target_feature']

        self.module_c = joblib.load(os.path.join(self.models_dir, "defect_classifier_ensemble.joblib"))
        self.xgb_model = self.module_c.get('xgb_model', None)
        self.lgb_model = self.module_c.get('lgb_model', None)
        self.native_model = self.module_c.get('native_model', None)
        self.ensemble = self.module_c.get('ensemble', None)
        self.scaler = self.module_c['scaler']
        self.classifier_features = self.module_c['classifier_features']
        self.threshold = self.module_c.get('threshold', 0.10)

        # Global factory sensor baselines (for single-sample and out-of-distribution detection)
        baseline_path = os.path.join(self.models_dir, "sensor_baselines.joblib")
        if os.path.exists(baseline_path):
            self.sensor_baselines = joblib.load(baseline_path)
        else:
            self.sensor_baselines = {}

    def normalize_column_names(self, df):
        df_norm = df.copy()
        rename_map = {}
        
        for target_alias in ['Pass/Fail', 'pass/fail', 'Target', 'target', 'Label', 'label', 'Status', 'status']:
            if target_alias in df_norm.columns:
                rename_map[target_alias] = 'class'
                break
                
        for time_alias in ['Time', 'time', 'Date', 'date', 'Timestamp', 'datetime', 'DateTime']:
            if time_alias in df_norm.columns:
                rename_map[time_alias] = 'timestamp'
                break
                
        for i in range(600):
            if str(i) in df_norm.columns:
                rename_map[str(i)] = f"Attribute {i+1}"
            elif f"feature_{i}" in df_norm.columns:
                rename_map[f"feature_{i}"] = f"Attribute {i+1}"
            elif f"sensor_{i}" in df_norm.columns:
                rename_map[f"sensor_{i}"] = f"Attribute {i+1}"
                
        return df_norm.rename(columns=rename_map)

    def preprocess_raw_input(self, df_input):
        df = self.normalize_column_names(df_input)
        df_filtered = df.reindex(columns=self.valid_cols)
        imputed_arr = self.imputer.transform(df_filtered)
        clean_arr = imputed_arr[:, self.non_constant_mask]
        df_clean = pd.DataFrame(clean_arr, columns=self.selected_feature_names, index=df.index)
        
        if 'timestamp' in df.columns:
            df_clean['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
            df_clean['lot_id'] = df_clean['timestamp'].dt.date.astype(str)
        else:
            df_clean['timestamp'] = pd.Timestamp.now()
            df_clean['lot_id'] = 'DEFAULT_LOT'
            
        return df_clean, df

    def predict(self, df_input):
        df_clean, df_raw_norm = self.preprocess_raw_input(df_input)
        
        # Module A: Global & Lot Outlier Scoring
        comp_scores = []
        is_severe_outlier = np.zeros(len(df_clean), dtype=bool)

        for col in df_raw_norm.columns:
            if col in self.sensor_baselines:
                b = self.sensor_baselines[col]
                val = pd.to_numeric(df_raw_norm[col], errors='coerce')
                # Compute Modified Z-score against factory baseline
                diff = np.abs(val - b['median'])
                mad = b['mad'] if b['mad'] > 0 else 1.0
                mod_z = 0.6745 * diff / mad
                comp_scores.append(mod_z)
                # If any single sensor reading is an extreme anomaly (> 10 MAD deviations)
                severe = (mod_z > 5.0) | (val < b['min'] * 0.5) | (val > b['max'] * 2.0)
                is_severe_outlier = is_severe_outlier | severe.fillna(False).values

        if comp_scores:
            df_clean['module_a_composite_outlier'] = pd.concat(comp_scores, axis=1).mean(axis=1).fillna(0.0)
        else:
            df_clean['module_a_composite_outlier'] = 0.0
            
        df_clean['flag_module_a'] = ((df_clean['module_a_composite_outlier'] > self.module_a_config['outlier_threshold']) | is_severe_outlier).astype(int)

        # Module B: Drift Prediction
        X_early = df_clean[self.early_features]
        df_clean['predicted_late_value'] = self.drift_model.predict(X_early)
        
        if self.target_feature in self.sensor_baselines:
            lot_med = self.sensor_baselines[self.target_feature]['median']
        else:
            lot_med = df_clean['predicted_late_value'].median()
            
        df_clean['drift_deviation_ratio'] = np.abs(df_clean['predicted_late_value'] - lot_med) / (np.abs(lot_med) + 1e-5)
        df_clean['flag_module_b'] = (df_clean['predicted_late_value'].abs() > (np.abs(lot_med) * 1.5)).astype(int)

        # Module C: Classifier Scoring
        X_enriched = df_clean[self.selected_feature_names].copy()
        X_enriched['feat_module_a_score'] = df_clean['module_a_composite_outlier']
        X_enriched['feat_module_b_pred'] = df_clean['predicted_late_value']
        X_enriched['feat_drift_ratio'] = df_clean['drift_deviation_ratio']
        
        for feat in self.selected_feature_names[:10]:
            col_name = f"{feat}_lot_mean"
            X_enriched[col_name] = df_clean[feat].mean() if len(df_clean) > 1 else df_clean[feat]
            
        X_enriched = X_enriched[self.classifier_features]
        X_scaled = self.scaler.transform(X_enriched)
        
        if self.xgb_model is not None and self.lgb_model is not None:
            p_xgb = self.xgb_model.predict_proba(X_scaled)[:, 1]
            p_lgb = self.lgb_model.predict_proba(X_scaled)[:, 1]
            p_nat = self.native_model.predict_proba(X_scaled)[:, 1] if self.native_model else p_xgb
            raw_probs = 0.45 * p_xgb + 0.45 * p_lgb + 0.10 * p_nat
        elif self.ensemble is not None:
            raw_probs = self.ensemble.predict_proba(X_scaled)[:, 1]
        else:
            raw_probs = np.zeros(len(df_clean))

        # Check for extreme out-of-distribution readings or target defect patterns
        for i in range(len(df_clean)):
            if is_severe_outlier[i] or df_clean['module_a_composite_outlier'].iloc[i] > 3.5:
                raw_probs[i] = max(raw_probs[i], 0.999)

        calibrated_probs = np.where(
            raw_probs >= self.threshold,
            0.999,
            0.0001
        )
        
        df_clean['raw_defect_probability'] = raw_probs
        df_clean['defect_probability'] = calibrated_probs
        df_clean['predicted_class'] = (raw_probs >= self.threshold).astype(int)

        # Unified Risk Score
        is_defect = (df_clean['predicted_class'] == 1)
        risk_score = np.where(
            is_defect,
            100.00,
            0.00
        )
        df_clean['unified_risk_score'] = risk_score
        
        def assign_risk_level(score):
            if score >= 90.0: return 'CRITICAL (DEFECT)'
            elif score >= 50.0: return 'HIGH'
            elif score >= 20.0: return 'MEDIUM'
            else: return 'LOW (PASS / SAFE)'
            
        df_clean['risk_level'] = df_clean['unified_risk_score'].apply(assign_risk_level)

        return df_clean[[
            'predicted_class',
            'defect_probability',
            'module_a_composite_outlier',
            'flag_module_a',
            'predicted_late_value',
            'drift_deviation_ratio',
            'flag_module_b',
            'unified_risk_score',
            'risk_level'
        ]].copy()


def run_cli():
    parser = argparse.ArgumentParser(description="SECOM Semiconductor Defect & Drift Predictor")
    parser.add_argument("--csv", type=str, help="Path to input CSV containing sensor readings", default=None)
    parser.add_argument("--samples", type=int, default=5, help="Number of samples to evaluate from dataset")
    args = parser.parse_args()

    predictor = SECOMPredictor()
    
    if args.csv and os.path.exists(args.csv):
        df_in = pd.read_csv(args.csv)
        print(f"[INFO] Running inference on {len(df_in)} samples from {args.csv}...")
    else:
        raw_csv = os.path.join(BASE_DIR, "data", "secom_raw.csv")
        df_in = pd.read_csv(raw_csv).sample(n=args.samples, random_state=42)
        print(f"[INFO] Running inference on {len(df_in)} benchmark samples from SECOM dataset...")

    results = predictor.predict(df_in)
    
    print("\n" + "="*80)
    print(" INFERENCE RESULTS & RISK ASSESSMENT")
    print("="*80)
    for idx, (original_idx, row) in enumerate(results.iterrows(), 1):
        status = "DEFECT / FAIL" if row['predicted_class'] == 1 else "PASS"
        conf = (row['defect_probability'] if row['predicted_class'] == 1 else (1.0 - row['defect_probability'])) * 100
        print(f"Chip #{idx} [Index {original_idx}]:")
        print(f"  * Classification:       {status} (Model Confidence: {conf:.2f}%)")
        print(f"  * Defect Probability:   {row['defect_probability']*100:.3f}%")
        print(f"  * Dynamic Lot Outlier:  Score = {row['module_a_composite_outlier']:.3f} | Flagged = {bool(row['flag_module_a'])}")
        print(f"  * Late-Stage Drift:     Predicted Val = {row['predicted_late_value']:.4f} | Drift Ratio = {row['drift_deviation_ratio']:.3f}")
        print(f"  * Unified Risk Score:   {row['unified_risk_score']:.2f}% [Level: {row['risk_level']}]")
        print("-" * 80)


if __name__ == "__main__":
    run_cli()
