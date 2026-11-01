import os
import sys

# Ensure src is on path
src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

print("--- CHECK 1: DATA LOADER ---")
import data_loader
df = data_loader.load_raw_data()
assert df.shape == (1460, 81), f"Unexpected shape {df.shape}"
assert "SalePrice" in df.columns, "SalePrice target missing"
print(f"PASSED: Loaded genuine dataset shape {df.shape}")

print("\n--- CHECK 2: EDA VISUALIZATIONS ---")
vis_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "visualizations")
vis_files = os.listdir(vis_dir)
print(f"Visualizations count: {len(vis_files)}")
assert len(vis_files) >= 10, "Expected at least 10 visualization files"
print("PASSED: All EDA & evaluation plots exist on disk.")

print("\n--- CHECK 3: FEATURE ENGINEERING ---")
import feature_engineering
df_feat = feature_engineering.engineer_features(df)
for col in ["TotalSF", "TotalBath", "HouseAge", "RemodAge"]:
    assert col in df_feat.columns, f"{col} missing from engineered features"
print(f"PASSED: Feature engineering generated {df_feat.shape[1]} columns including TotalSF, TotalBath, HouseAge.")

print("\n--- CHECK 4: PREPROCESSOR ---")
import preprocessing
prep_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "preprocessor.joblib")
prep = preprocessing.HousingPreprocessor.load(prep_path)
assert len(prep.feature_names_out) >= 300, "Expected >= 300 encoded features"
print(f"PASSED: Preprocessor verified with {len(prep.feature_names_out)} features.")

print("\n--- CHECK 5: TRAINED MODELS ---")
models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
model_files = os.listdir(models_dir)
print("Trained model artifacts:", model_files)
expected_models = [
    "linear_regression.joblib",
    "decision_tree_regressor.joblib",
    "random_forest_regressor.joblib",
    "gradient_boosting_regressor.joblib",
    "xgboost_regressor.joblib",
    "tuned_best_model.joblib"
]
for exp in expected_models:
    assert exp in model_files, f"{exp} missing from models directory"
print("PASSED: All 5 base models and the tuned champion model exist.")

print("\n--- CHECK 6: PREDICTION INFERENCE ENGINE ---")
import prediction
predictor = prediction.HousePricePredictor()
res = predictor.predict({
    "OverallQual": 8,
    "GrLivArea": 2000,
    "TotalBsmtSF": 1000,
    "1stFlrSF": 1000,
    "2ndFlrSF": 1000,
    "FullBath": 2,
    "HalfBath": 1,
    "BedroomAbvGr": 3,
    "GarageCars": 2,
    "YearBuilt": 2005
})
usd = res["predicted_price_usd"]
inr = res["predicted_price_inr"]
print(f"Predicted Price: ${usd:,.2f} USD | INR: {inr:,.2f}")
assert usd > 100000, "Prediction out of reasonable bounds"
print("PASSED: End-to-end prediction engine operating smoothly.")

print("\n--- CHECK 7: EVALUATION METRICS VERIFICATION ---")
res_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "evaluation_results")
assert os.path.exists(os.path.join(res_dir, "model_comparison_metrics.csv")), "model_comparison_metrics.csv missing"
assert os.path.exists(os.path.join(res_dir, "hyperparameter_tuning_results.json")), "hyperparameter_tuning_results.json missing"
print("PASSED: Evaluation and hyperparameter tuning result files verified.")

print("\n" + "=" * 55)
print("ALL SYSTEM INTEGRATION CHECKS PASSED PERFECTLY!")
print("=" * 55)
