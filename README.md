# House Price Prediction Using Machine Learning
### An End-to-End Predictive Analytics System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-red.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-brightgreen.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An enterprise-grade, reproducible, end-to-end Machine Learning portfolio and university viva project implementing the complete data science lifecycle on the **Ames Housing Dataset** (Dean De Cock, 2011).

The system implements data ingestion, data understanding, exploratory data analysis across 10 distinct perspectives, leakage-free Scikit-Learn `ColumnTransformer` preprocessing, domain feature engineering (`TotalSF`, `TotalBath`, `HouseAge`), 5-fold cross-validation, benchmarking across 5 regression models, hyperparameter optimization (`RandomizedSearchCV`), a dual-currency real-time prediction engine ($ USD & ₹ INR), and an interactive 12-page Streamlit analytical dashboard.

---

## 🏗️ Project Architecture & Directory Structure

```text
house-price-prediction/
│
├── app.py                         # 12-Page Streamlit Analytics & Prediction Dashboard
├── requirements.txt               # Pinned Python package dependencies
├── README.md                      # Comprehensive project documentation & guide
│
├── data/
│   └── house_prices.csv           # Ames Housing Dataset (1,460 rows, 81 columns)
│
├── notebooks/
│   └── house_price_analysis.ipynb # Interactive Jupyter Notebook covering all 12 phases
│
├── src/
│   ├── data_loader.py             # Dataset loading, caching, and statistical profiling
│   ├── eda.py                     # 10 EDA visualizations & Plotly figure generators
│   ├── preprocessing.py          # Leakage-free ColumnTransformer pipeline
│   ├── feature_engineering.py     # TotalSF, TotalBath, HouseAge, RemodAge engineering
│   ├── model_training.py          # 80/20 split, 5-fold CV, and training of 5 regressors
│   ├── model_evaluation.py        # MAE, MSE, RMSE, R² metrics and diagnostic plots
│   ├── model_comparison.py        # Benchmark comparison table & champion justification
│   └── prediction.py              # Real-time inference engine (USD & INR conversion)
│
├── models/
│   ├── preprocessor.joblib        # Fitted Scikit-Learn ColumnTransformer
│   ├── linear_regression.joblib   # Trained Ridge Linear Regression model
│   ├── decision_tree_regressor.joblib
│   ├── random_forest_regressor.joblib
│   ├── gradient_boosting_regressor.joblib
│   ├── xgboost_regressor.joblib
│   ├── tuned_best_model.joblib    # Champion Tuned Model (Gradient Boosting)
│   └── train_test_data.joblib     # Preprocessed train/test partitions cache
│
├── outputs/
│   ├── visualizations/            # 10 EDA figures + 4 evaluation diagnostic charts
│   └── evaluation_results/        # JSON & CSV performance metrics & logs
│
└── reports/
    └── final_project_report.md    # 20-Section Academic Project Report & Viva Dissertation
```

---

## 📊 Empirical Model Benchmark Results

All metrics were evaluated on the **20% held-out test partition (292 real residential transactions)**, kept untouched until final evaluation:

| Rank | Model | Test MAE ($) | Test MSE | Test RMSE ($) | Test $R^2$ | Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| 🥇 | **Tuned Gradient Boosting** | **$14,482.10** | **$5.5286 \times 10^8$** | **$23,513.00** | **0.9279** | **Champion Model (+17.27% improvement)** |
| 🥈 | **XGBoost Regressor** | $14,885.54 | $5.8575 \times 10^8$ | $24,202.35 | 0.9236 | Strong Non-Linear Regularization |
| 🥉 | **Linear Regression (Ridge)** | $17,504.65 | $7.5910 \times 10^8$ | $27,551.72 | 0.9010 | Reliable Linear Baseline |
| 4 | **Random Forest (Tuned)** | $16,104.22 | $7.7402 \times 10^8$ | $27,821.30 | 0.8991 | Ensemble Bagging |
| 5 | **Decision Tree Regressor** | $23,391.33 | $1.2280 \times 10^9$ | $35,042.27 | 0.8399 | High Variance / Overfitting |

---

## ⚙️ Installation & Setup

### 1. Clone or Navigate to Project Directory
```bash
cd house-price-prediction
```

### 2. Create and Activate Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Execution Guide (Step-by-Step)

You can run each modular phase individually from the command line:

```bash
# 1. Inspect dataset and summary statistics
python src/data_loader.py

# 2. Generate all 10 EDA visualizations to outputs/visualizations/
python src/eda.py

# 3. Test preprocessing pipeline and verify zero leakage
python src/preprocessing.py

# 4. Generate engineered features (TotalSF, TotalBath, HouseAge)
python src/feature_engineering.py

# 5. Train all 5 models with 5-Fold Cross Validation
python src/model_training.py

# 6. Evaluate all models on held-out test partition & generate diagnostic plots
python src/model_evaluation.py

# 7. Generate benchmark comparison tables and champion justification
python src/model_comparison.py

# 8. Run 5-fold cross-validated hyperparameter tuning
python src/hyperparameter_tuning.py

# 9. Test dual-currency prediction engine on sample input
python src/prediction.py
```

---

## 🖥️ Launching the Streamlit Web Application

To launch the full interactive 12-page web dashboard:

```bash
streamlit run app.py
```

The application will launch automatically in your browser at `http://localhost:8501`.

### Dashboard Navigation Modules:
1. **Dashboard Overview:** Executive summary, KPIs, and architecture workflow.
2. **Dataset Explorer:** Dynamic table viewer, schema metadata, missing values bar chart, and manual CSV uploader.
3. **Exploratory Data Analysis:** 10 diagnostic figures with observations, patterns, and modeling implications.
4. **Data Preprocessing:** ColumnTransformer architecture, median imputation, one-hot encoding, and leakage prevention.
5. **Feature Engineering:** Formulation of `TotalSF`, `TotalBath`, `HouseAge`, and correlation bar charts.
6. **Model Selection:** Theory, mathematics, suitability, pros & cons for all 5 algorithms.
7. **Model Training:** 5-fold CV metrics and overfitting/underfitting diagnostic badges.
8. **Model Evaluation:** Plain-language metric guides, actual vs. predicted plots, and residual error distributions.
9. **Model Comparison:** Interactive comparison tables and multi-metric Plotly charts.
10. **Hyperparameter Tuning:** Parameter grids, search logs, and before/after improvement comparisons.
11. **House Price Prediction:** Interactive form with sliders, model dropdown, dual USD/INR outputs, and CSV download.
12. **Project Conclusion:** University viva voce Q&A preparation, limitations, and future roadmap.

---

## 📓 Running the Jupyter Notebook

To explore the sequential interactive notebook:

```bash
jupyter notebook notebooks/house_price_analysis.ipynb
```

---

## 🎙️ University Viva Voce Preparation (Quick Q&A)

* **Q: Why choose the Ames dataset over Boston Housing?**  
  *A:* Boston Housing is officially deprecated in Scikit-Learn due to ethical concerns and small sample size (506 rows, 13 variables). Ames Housing provides 1,460 modern records with 81 comprehensive structural, temporal, and spatial features.
* **Q: How was data leakage prevented?**  
  *A:* The 80% train / 20% test partition was executed first. Preprocessing statistics (imputation medians, standard scaling parameters, category lists) were fitted strictly on the training partition and transformed onto the test set without parameter re-estimation.
* **Q: Why did Gradient Boosting and XGBoost achieve superior results?**  
  *A:* Property pricing exhibits multiplicative, non-linear interactions (e.g., square footage is valued significantly higher when paired with high overall quality and an affluent neighborhood). Sequential boosting models capture these interactions through shallow decision stumps optimized via gradient descent.
* **Q: What is the primary difference between MAE and RMSE?**  
  *A:* MAE measures average absolute error linearly ($L_1$ norm), treating all errors proportionally. RMSE squares errors before taking the root ($L_2$ norm), penalizing large outliers much more heavily.

---

## 📜 License & Citation
Developed for academic research and portfolio demonstration. Distributed under the MIT License.
Ames Housing Dataset courtesy of Professor Dean De Cock (Truman State University, 2011).
