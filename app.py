"""
HOUSE PRICE PREDICTION: Machine Learning & Predictive Analytics Dashboard
Multi-page Streamlit Application demonstrating end-to-end Data Science lifecycle.
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Add src to system path
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data_loader import load_raw_data, get_dataset_metadata, get_data_summary
from feature_engineering import engineer_features, get_feature_correlations
from preprocessing import HousingPreprocessor
from model_evaluation import METRIC_EXPLANATIONS, build_plotly_actual_vs_predicted, build_plotly_residuals
from model_training import MODEL_THEORY
from prediction import HousePricePredictor, USD_TO_INR_RATE

# Page Configuration
st.set_page_config(
    page_title="House Price Prediction | ML Analytics Dashboard",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background: linear-gradient(135deg, #F8FAFC 0%, #EFF6FF 100%);
        border: 1px solid #DBEAFE;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }
    .kpi-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1D4ED8;
    }
    .kpi-lbl {
        font-size: 0.85rem;
        color: #6B7280;
        font-weight: 600;
        text-transform: uppercase;
        margin-top: 0.3rem;
    }
    .section-banner {
        background-color: #1E40AF;
        color: white;
        padding: 0.6rem 1rem;
        border-radius: 6px;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .badge-success {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# Cache data loading
@st.cache_data
def get_cached_raw_data():
    return load_raw_data()

@st.cache_resource
def get_cached_predictor(model_name="Tuned Champion (Gradient Boosting)"):
    return HousePricePredictor(model_name=model_name)

df_raw = get_cached_raw_data()
metadata = get_dataset_metadata()

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/isometric/100/real-estate.png", width=70)
st.sidebar.title("Navigation")
st.sidebar.markdown("**Project Workflow Stages**")

pages = [
    "1. Dashboard Overview",
    "2. Dataset Explorer",
    "3. Exploratory Data Analysis",
    "4. Data Preprocessing",
    "5. Feature Engineering",
    "6. Model Selection",
    "7. Model Training",
    "8. Model Evaluation",
    "9. Model Comparison",
    "10. Hyperparameter Tuning",
    "11. House Price Prediction",
    "12. Project Conclusion"
]

selected_page = st.sidebar.radio("Select Lifecycle Phase:", pages)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Dataset:** Ames Housing (Dean De Cock)")
st.sidebar.markdown(f"**Target:** `SalePrice` ($ USD)")
st.sidebar.markdown(f"**Current Benchmark Rate:** 1 USD = ₹{USD_TO_INR_RATE:.2f}")


# ==============================================================================
# PAGE 1: DASHBOARD OVERVIEW
# ==============================================================================
if selected_page == "1. Dashboard Overview":
    st.markdown('<div class="main-header">HOUSE PRICE PREDICTION SYSTEM</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Machine Learning & Predictive Analytics Dashboard | End-to-End Data Science Lifecycle</div>', unsafe_allow_html=True)

    # Top KPI cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-val">{df_raw.shape[0]:,}</div>
            <div class="kpi-lbl">Total Records (Homes)</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-val">{df_raw.shape[1]}</div>
            <div class="kpi-lbl">Raw Attributes</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-val">0.9279</div>
            <div class="kpi-lbl">Champion Test R² Score</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-val">$23,513</div>
            <div class="kpi-lbl">Champion Test RMSE</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Executive Overview
    st.markdown("### 📋 Executive Summary")
    st.info(
        "This enterprise-grade Machine Learning solution predicts residential property valuations using the genuine "
        "**Ames Housing Dataset** (Dean De Cock, Truman State University). Spanning 81 granular property features, the system "
        "executes rigorous exploratory data analysis, data leakage prevention, domain feature engineering, multi-algorithm "
        "training across 5 regression models, 5-fold cross-validation, and systematic hyperparameter optimization."
    )

    # Architectural Lifecycle
    st.markdown("### 🏗️ End-to-End System Architecture")
    steps = [
        ("Phase 1-2", "Data Ingestion & Understanding", "1,460 rows, 81 columns, target variable SalePrice identification."),
        ("Phase 3", "Exploratory Data Analysis", "10 comprehensive distributions, scatterplots, heatmaps, and outlier boundaries."),
        ("Phase 4", "Data Preprocessing Pipeline", "Leakage-free Scikit-Learn ColumnTransformer (Median imputer, OneHotEncoder, Scaler)."),
        ("Phase 5", "Feature Engineering", "High-correlation composite metrics: TotalSF (+0.78), TotalBath (+0.63), HouseAge (-0.52)."),
        ("Phase 6-8", "Model Selection & Training", "Linear Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost."),
        ("Phase 9-10", "Model Evaluation & Comparison", "MAE, MSE, RMSE, R² benchmarked across Train, 5-Fold CV, and untouched 20% Test."),
        ("Phase 11", "Hyperparameter Optimization", "RandomizedSearchCV tuning tree depth, learning rate, subsample, and regularization."),
        ("Phase 12-13", "Dual-Currency Inference Engine", "Real-time interactive prediction interface with USD ($) and INR (₹) valuation.")
    ]
    for p_id, p_title, p_desc in steps:
        with st.expander(f"🔹 **{p_id}: {p_title}**"):
            st.write(p_desc)


# ==============================================================================
# PAGE 2: DATASET EXPLORER
# ==============================================================================
elif selected_page == "2. Dataset Explorer":
    st.markdown("## 🔍 Dataset Selection & Exploration")
    st.markdown(f"**Dataset Name:** {metadata['name']} | **Source:** [OpenML #{543}]({metadata['source_url']})")
    st.write(metadata["description"])

    summary = get_data_summary(df_raw)

    colA, colB, colC, colD = st.columns(4)
    colA.metric("Total Rows", f"{summary['total_rows']:,}")
    colB.metric("Total Columns", summary["total_columns"])
    colC.metric("Numerical Columns", summary["numerical_features_count"])
    colD.metric("Categorical Columns", summary["categorical_features_count"])

    # Manual upload option
    st.markdown("---")
    st.markdown("#### 📂 Upload Custom CSV Dataset (Optional Override)")
    uploaded_file = st.file_uploader("Upload an alternative housing CSV file", type=["csv"])
    display_df = df_raw
    if uploaded_file is not None:
        try:
            display_df = pd.read_csv(uploaded_file)
            st.success(f"Custom dataset loaded! Shape: {display_df.shape[0]} rows x {display_df.shape[1]} columns")
        except Exception as e:
            st.error(f"Error reading file: {e}")

    # Interactive Table Viewer
    st.markdown("#### 📄 Dataset Sample Preview")
    rows_to_show = st.slider("Select number of rows to preview:", min_value=5, max_value=50, value=10)
    st.dataframe(display_df.head(rows_to_show), use_container_width=True)

    # Missing Value Inspector
    st.markdown("#### ⚠️ Missing Values Breakdown")
    missing_dict = summary["missing_values_breakdown"]
    if missing_dict:
        m_df = pd.DataFrame(list(missing_dict.items()), columns=["Column", "Missing Count"])
        m_df["Missing Percentage (%)"] = (m_df["Missing Count"] / summary["total_rows"] * 100).round(2)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.dataframe(m_df.head(15), use_container_width=True)
        with col2:
            fig_miss = px.bar(
                m_df.head(10), x="Column", y="Missing Count",
                title="Top 10 Columns with Missing Values",
                color="Missing Percentage (%)", color_continuous_scale="Reds"
            )
            st.plotly_chart(fig_miss, use_container_width=True)

    # Statistical Summary
    st.markdown("#### 📊 Statistical Summary of Numerical Attributes")
    st.dataframe(display_df.describe().T.round(2), use_container_width=True)


# ==============================================================================
# PAGE 3: EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================================
elif selected_page == "3. Exploratory Data Analysis":
    st.markdown("## 📈 Comprehensive Exploratory Data Analysis")
    st.write("Detailed visual analysis across 10 distinct perspectives to extract behavioral dynamics for predictive modeling.")

    from eda import EDA_INSIGHTS

    vis_options = {
        "1. House Price Distribution": ("1_price_distribution", "1_house_price_distribution.png"),
        "2. Boxplot of House Prices": ("2_price_boxplot", "2_house_price_boxplot.png"),
        "3. Correlation Heatmap": ("3_correlation_heatmap", "3_correlation_heatmap.png"),
        "4. Living Area vs. Price": ("4_area_vs_price", "4_area_vs_price_scatter.png"),
        "5. Overall Quality vs. Price": ("5_overall_qual_vs_price", "5_overall_qual_vs_price.png"),
        "6. Numerical Distributions": ("6_numerical_distributions", "6_numerical_distributions.png"),
        "7. Categorical Analysis (Neighborhood)": ("7_categorical_analysis", "7_categorical_analysis.png"),
        "8. Outlier Detection": ("8_outlier_detection", "8_outlier_detection.png"),
        "9. Feature Relationships (Multivariate)": ("9_feature_relationships", "9_feature_relationships.png"),
        "10. Target Variable Normalization": ("10_target_variable_analysis", "10_target_variable_analysis.png")
    }

    selected_vis = st.selectbox("Select Visual Analysis:", list(vis_options.keys()))
    key, img_name = vis_options[selected_vis]
    info = EDA_INSIGHTS[key]

    img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "visualizations", img_name)

    col1, col2 = st.columns([3, 2])
    with col1:
        if os.path.exists(img_path):
            st.image(img_path, caption=info["title"], use_container_width=True)
        else:
            st.warning("Visual artifact not found on disk. Run `python src/eda.py` to regenerate.")
            
    with col2:
        st.markdown(f"### {info['title']}")
        st.markdown("**🔍 Empirical Observation:**")
        st.write(info["observation"])
        st.markdown("**💡 Useful Pattern Identified:**")
        st.write(info["pattern"])
        st.markdown("**⚙️ Effect on Model Development:**")
        st.info(info["modeling_implication"])

    # Interactive bonus charts
    st.markdown("---")
    st.markdown("#### ⚡ Interactive Plotly Explorer")
    int_choice = st.radio("Interactive Visual:", ["Interactive Price Distribution", "Interactive Living Area vs Price Scatter", "Interactive Correlation Heatmap"], horizontal=True)
    if int_choice == "Interactive Price Distribution":
        fig = px.histogram(df_raw, x="SalePrice", nbins=50, marginal="box", title="Interactive House Price Distribution with Boxplot", color_discrete_sequence=["#2563EB"])
        st.plotly_chart(fig, use_container_width=True)
    elif int_choice == "Interactive Living Area vs Price Scatter":
        fig = px.scatter(df_raw, x="GrLivArea", y="SalePrice", color="OverallQual", hover_data=["YearBuilt", "Neighborhood"], title="GrLivArea vs SalePrice colored by Overall Quality", color_continuous_scale="Viridis")
        st.plotly_chart(fig, use_container_width=True)
    else:
        num_cols = df_raw.select_dtypes(include=["number"]).drop(columns=["Id"], errors="ignore")
        top_corr_cols = num_cols.corr()["SalePrice"].abs().sort_values(ascending=False).head(12).index
        fig = px.imshow(num_cols[top_corr_cols].corr(), text_auto=".2f", aspect="auto", title="Correlation Heatmap of Top 12 Features", color_continuous_scale="Blues")
        st.plotly_chart(fig, use_container_width=True)


# ==============================================================================
# PAGE 4: DATA PREPROCESSING
# ==============================================================================
elif selected_page == "4. Data Preprocessing":
    st.markdown("## ⚙️ Data Preprocessing & Leakage Prevention")
    st.write("Structured transformation pipeline built with Scikit-Learn `ColumnTransformer` and `Pipeline`.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🛡️ Preventing Data Leakage")
        st.markdown("""
        - **Strict Partitioning First**: The dataset is split into **80% Training** and **20% Testing** before any imputation or scaling.
        - **Training-Only Statistics**: The median values for numerical imputation and frequent categories are computed *only* on the training split.
        - **Untouched Holdout**: The testing partition is purely evaluated via `.transform()` with zero parameter re-estimation.
        """)
    with col2:
        st.markdown("### 🧹 Outlier Treatment Protocol")
        st.markdown("""
        - **Dean De Cock Criterion**: Houses with Ground Living Area (`GrLivArea`) > 4,000 sq ft sold for under $300,000 represent atypical partial estate sales.
        - **Execution**: 2 extreme leverage outliers removed strictly from the training partition.
        """)

    st.markdown("### 🔄 Scikit-Learn ColumnTransformer Architecture")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        **Numerical Sub-Pipeline (46 attributes):**
        1. `SimpleImputer(strategy='median')`: Resolves missing values using robust median statistic.
        2. `StandardScaler()`: Zero-mean, unit-variance standardization for gradient stability.
        """)
    with c2:
        st.markdown("""
        **Categorical Sub-Pipeline (43 attributes):**
        1. `SimpleImputer(strategy='constant', fill_value='Missing')`: Captures domain semantics where NA signifies absence of amenity.
        2. `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`: Generates 310 total encoded binary dimensions.
        """)

    preprocessor_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "preprocessor.joblib")
    if os.path.exists(preprocessor_path):
        prep = joblib.load(preprocessor_path)
        st.success(f"Fitted Preprocessor Pipeline verified! Encoded feature output dimensions: **{len(prep.feature_names_out)} columns**.")


# ==============================================================================
# PAGE 5: FEATURE ENGINEERING
# ==============================================================================
elif selected_page == "5. Feature Engineering":
    st.markdown("## 🛠️ Feature Engineering & Selection")
    st.write("Domain-driven composite variables designed to maximize predictive correlation with property valuation.")

    df_feat = engineer_features(df_raw)
    corrs = get_feature_correlations(df_feat, top_n=15)

    st.markdown("### 📐 Derived Features Formulations")
    f1, f2 = st.columns(2)
    with f1:
        st.markdown("""
        - **Total Square Footage (`TotalSF`)**:
          $$\\text{TotalSF} = \\text{TotalBsmtSF} + \\text{1stFlrSF} + \\text{2ndFlrSF}$$
          *Significance: Single strongest unified spatial indicator ($r = +0.782$).*
        - **Total Bathrooms (`TotalBath`)**:
          $$\\text{TotalBath} = \\text{FullBath} + 0.5\\times\\text{HalfBath} + \\text{BsmtFullBath} + 0.5\\times\\text{BsmtHalfBath}$$
          *Significance: Aggregates sanitary capacity across entire dwelling ($r = +0.632$).*
        """)
    with f2:
        st.markdown("""
        - **House Age at Sale (`HouseAge`)**:
          $$\\text{HouseAge} = \\text{YrSold} - \\text{YearBuilt}$$
          *Significance: Direct depreciation penalty indicator ($r = -0.523$).*
        - **Remodel Age (`RemodAge`)**:
          $$\\text{RemodAge} = \\text{YrSold} - \\text{YearRemodAdd}$$
          *Significance: Captures freshness of structural modernization ($r = -0.509$).*
        """)

    st.markdown("### 📊 Correlation Ranking of Top 15 Engineered & Raw Predictors")
    corr_df = pd.DataFrame({"Feature": corrs.index, "Pearson Correlation (r)": corrs.values})
    fig_corr = px.bar(
        corr_df, x="Pearson Correlation (r)", y="Feature", orientation="h",
        color="Pearson Correlation (r)", color_continuous_scale="Blues",
        title="Top 15 Features by Correlation with SalePrice"
    )
    fig_corr.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_corr, use_container_width=True)


# ==============================================================================
# PAGE 6: MODEL SELECTION
# ==============================================================================
elif selected_page == "6. Model Selection":
    st.markdown("## 🧠 Model Selection & Algorithmic Foundations")
    st.write("Five diverse regression algorithms evaluated across parametric, single-tree, bagging, and boosting paradigms.")

    for name, details in MODEL_THEORY.items():
        with st.expander(f"🔹 **{name}**", expanded=(name == "XGBoost Regressor")):
            st.markdown(f"**Theoretical Concept:** {details['concept']}")
            st.markdown(f"**Domain Suitability for Housing:** {details['suitability']}")
            c1, c2 = st.columns(2)
            with c1:
                st.success(f"**Advantages:** {details['pros']}")
            with c2:
                st.error(f"**Limitations:** {details['cons']}")


# ==============================================================================
# PAGE 7: MODEL TRAINING
# ==============================================================================
elif selected_page == "7. Model Training":
    st.markdown("## 🏋️ Model Training & Cross-Validation Analysis")
    st.write("Models trained on 80% train split (1,166 clean samples) with 5-fold cross-validation to assess generalization.")

    cv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "evaluation_results", "training_cv_results.joblib")
    if os.path.exists(cv_path):
        cv_data = joblib.load(cv_path)
        
        cv_records = []
        for m_name, vals in cv_data.items():
            cv_records.append({
                "Model": m_name,
                "Train RMSE ($)": f"${vals['train_rmse']:,.2f}",
                "Train R²": f"{vals['train_r2']:.4f}",
                "CV-5 RMSE ($)": f"${vals['cv_rmse']:,.2f}",
                "CV-5 R²": f"{vals['cv_r2']:.4f}",
                "Fit Diagnosis": vals["fit_diagnosis"]
            })
        st.dataframe(pd.DataFrame(cv_records), use_container_width=True)

        st.markdown("### ⚠️ Overfitting / Underfitting Diagnoses")
        for m_name, vals in cv_data.items():
            badge_class = "badge-success" if "Well-Balanced" in vals["fit_diagnosis"] else "badge-error"
            st.markdown(f"- **{m_name}**: `{vals['fit_diagnosis']}`")


# ==============================================================================
# PAGE 8: MODEL EVALUATION
# ==============================================================================
elif selected_page == "8. Model Evaluation":
    st.markdown("## 🎯 Model Evaluation on Held-Out Test Partition")
    st.write("Diagnostic evaluation of the 20% untouched holdout test partition (292 real transactions).")

    # Metric Explanations
    st.markdown("### 📖 Regression Evaluation Metrics Explained")
    m_cols = st.columns(4)
    for idx, (m_key, m_info) in enumerate(METRIC_EXPLANATIONS.items()):
        with m_cols[idx]:
            st.markdown(f"**{m_info['full_name']}**")
            st.code(m_info["formula"])
            st.caption(m_info["explanation"])

    st.markdown("---")
    st.markdown("### 📉 Diagnostic Visualizations")
    
    vis_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "visualizations")
    p1 = os.path.join(vis_dir, "eval_actual_vs_predicted.png")
    p2 = os.path.join(vis_dir, "eval_residual_distribution.png")
    p3 = os.path.join(vis_dir, "eval_residual_vs_predicted.png")
    p4 = os.path.join(vis_dir, "eval_model_comparison_bars.png")

    c1, c2 = st.columns(2)
    with c1:
        if os.path.exists(p1):
            st.image(p1, caption="Figure: Actual vs Predicted Price", use_container_width=True)
        if os.path.exists(p3):
            st.image(p3, caption="Figure: Residuals vs Predicted Price (Homoscedasticity)", use_container_width=True)
    with c2:
        if os.path.exists(p2):
            st.image(p2, caption="Figure: Residual Error Distribution (Normality Check)", use_container_width=True)
        if os.path.exists(p4):
            st.image(p4, caption="Figure: Test Metrics Comparison", use_container_width=True)


# ==============================================================================
# PAGE 9: MODEL COMPARISON
# ==============================================================================
elif selected_page == "9. Model Comparison":
    st.markdown("## 🏆 Comprehensive Model Comparison Table")
    st.write("Standardized benchmark comparison across all 5 regression algorithms on the held-out test dataset.")

    comp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "evaluation_results", "full_model_comparison_table.csv")
    if os.path.exists(comp_path):
        comp_df = pd.read_csv(comp_path)
        
        display_tbl = comp_df.copy()
        display_tbl["Train RMSE ($)"] = display_tbl["Train RMSE ($)"].apply(lambda x: f"${x:,.2f}")
        display_tbl["CV-5 RMSE ($)"] = display_tbl["CV-5 RMSE ($)"].apply(lambda x: f"${x:,.2f}")
        display_tbl["Test MAE ($)"] = display_tbl["Test MAE ($)"].apply(lambda x: f"${x:,.2f}")
        display_tbl["Test RMSE ($)"] = display_tbl["Test RMSE ($)"].apply(lambda x: f"${x:,.2f}")
        display_tbl["Test MSE"] = display_tbl["Test MSE"].apply(lambda x: f"{x:.2e}")
        display_tbl["Test R²"] = display_tbl["Test R²"].round(4)
        
        st.dataframe(display_tbl, use_container_width=True)

        st.markdown("### 📊 Interactive Metric Comparison Chart")
        fig_bar = px.bar(
            comp_df, x="Model", y="Test RMSE ($)", color="Test R²",
            title="Untouched Test Set: RMSE & R² Comparison",
            color_continuous_scale="Viridis", text_auto=".0f"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Champion rationale
        exp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "evaluation_results", "best_model_explanation.json")
        if os.path.exists(exp_path):
            with open(exp_path) as f:
                exp_data = json.load(f)
            st.markdown(f"### 🥇 Best Performing Algorithm: **{exp_data['best_model_name']}**")
            st.markdown(f"**Test RMSE:** ${exp_data['test_rmse']:,.2f} | **Test MAE:** ${exp_data['test_mae']:,.2f} | **Test R²:** {exp_data['test_r2']:.4f}")
            for r in exp_data["key_reasons_for_superiority"]:
                st.markdown(f"- {r}")


# ==============================================================================
# PAGE 10: HYPERPARAMETER TUNING
# ==============================================================================
elif selected_page == "10. Hyperparameter Tuning":
    st.markdown("## 🎛️ Systematic Hyperparameter Tuning")
    st.write("Cross-validated parameter optimization via `RandomizedSearchCV` on Random Forest, Gradient Boosting, and XGBoost.")

    tuning_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "evaluation_results", "hyperparameter_tuning_results.json")
    if os.path.exists(tuning_path):
        with open(tuning_path) as f:
            tuning_data = json.load(f)

        st.success(f"**Tuned Champion Algorithm:** {tuning_data['champion_model_name']} | Test RMSE: **${tuning_data['champion_test_rmse']:,.2f}** | Test R²: **{tuning_data['champion_test_r2']:.4f}**")

        tuning_records = []
        for m_name, m_info in tuning_data["tuning_details"].items():
            tuning_records.append({
                "Algorithm": m_name,
                "Initial Test RMSE": f"${m_info['baseline_test_rmse']:,.2f}" if m_info["baseline_test_rmse"] else "N/A",
                "Tuned Test RMSE": f"${m_info['tuned_test_rmse']:,.2f}",
                "Tuned Test R²": f"{m_info['tuned_test_r2']:.4f}",
                "Improvement (%)": f"{m_info['improvement_pct']:+.2f}%",
                "Optimal Hyperparameters": str(m_info["best_params"])
            })

        st.dataframe(pd.DataFrame(tuning_records), use_container_width=True)

        st.markdown("### 🔍 Optimal Hyperparameters Found")
        for m_name, m_info in tuning_data["tuning_details"].items():
            with st.expander(f"⚙️ **{m_name}** Optimal Parameter Configuration"):
                st.json(m_info["best_params"])


# ==============================================================================
# PAGE 11: HOUSE PRICE PREDICTION
# ==============================================================================
elif selected_page == "11. House Price Prediction":
    st.markdown("## 🔮 House Price Prediction Interface")
    st.write("Configure physical property attributes to generate real-time market valuations.")

    predictor = get_cached_predictor()

    with st.form("prediction_form"):
        st.markdown("#### 1. Core Property Attributes")
        c1, c2, c3 = st.columns(3)
        with c1:
            overall_qual = st.slider("Overall Quality (1: Very Poor to 10: Excellent)", min_value=1, max_value=10, value=7)
            gr_liv_area = st.number_input("Ground Living Area (Sq Ft)", min_value=300, max_value=6000, value=1800, step=50)
            bedroom_abv = st.slider("Bedrooms Above Ground", min_value=1, max_value=8, value=3)
        with c2:
            total_bsmt_sf = st.number_input("Total Basement Area (Sq Ft)", min_value=0, max_value=4000, value=950, step=50)
            flr1_sf = st.number_input("1st Floor Area (Sq Ft)", min_value=300, max_value=4000, value=1100, step=50)
            flr2_sf = st.number_input("2nd Floor Area (Sq Ft)", min_value=0, max_value=3000, value=700, step=50)
        with c3:
            full_bath = st.slider("Full Bathrooms", min_value=1, max_value=5, value=2)
            half_bath = st.slider("Half Bathrooms", min_value=0, max_value=3, value=1)
            garage_cars = st.slider("Garage Capacity (Cars)", min_value=0, max_value=5, value=2)

        st.markdown("#### 2. Temporal & Location Details")
        c4, c5, c6 = st.columns(3)
        with c4:
            year_built = st.number_input("Year Built", min_value=1870, max_value=2026, value=2003)
            year_remod = st.number_input("Year Remodeled / Renovated", min_value=1950, max_value=2026, value=2005)
        with c5:
            neighborhoods = [
                "CollgCr", "Veenker", "Crawfor", "NoRidge", "Mitchel", "Somerst", "NWAmes",
                "OldTown", "BrkSide", "Sawyer", "NridgHt", "NAmes", "SawyerW", "IDOTRR",
                "MeadowV", "Edwards", "Timber", "Gilbert", "StoneBr", "ClearCr", "NPkVill",
                "Blmngtn", "BrDale", "SWISU", "Blueste"
            ]
            neighborhood = st.selectbox("Neighborhood", neighborhoods, index=0)
            bldg_type = st.selectbox("Building Type", ["1Fam", "TwnhsE", "Twnhs", "Duplex", "2fmCon"], index=0)
        with c6:
            house_style = st.selectbox("House Style", ["1Story", "2Story", "1.5Fin", "SLvl", "SFoyer", "1.5Unf", "2.5Unf", "2.5Fin"], index=1)
            central_air = st.selectbox("Central Air Conditioning", ["Y", "N"], index=0)

        st.markdown("#### 3. Inference Model Selection")
        selected_model = st.selectbox(
            "Select Trained Machine Learning Model for Prediction:",
            [
                "Tuned Champion (Gradient Boosting)",
                "XGBoost Regressor",
                "Gradient Boosting Regressor",
                "Random Forest Regressor",
                "Linear Regression",
                "Decision Tree Regressor"
            ],
            index=0
        )

        submitted = st.form_submit_button("🚀 Generate Property Valuation", use_container_width=True)

    if submitted:
        predictor.set_model(selected_model)
        user_inputs = {
            "OverallQual": overall_qual,
            "GrLivArea": gr_liv_area,
            "TotalBsmtSF": total_bsmt_sf,
            "1stFlrSF": flr1_sf,
            "2ndFlrSF": flr2_sf,
            "FullBath": full_bath,
            "HalfBath": half_bath,
            "BedroomAbvGr": bedroom_abv,
            "GarageCars": garage_cars,
            "GarageArea": garage_cars * 250,
            "YearBuilt": year_built,
            "YearRemodAdd": year_remod,
            "Neighborhood": neighborhood,
            "BldgType": bldg_type,
            "HouseStyle": house_style,
            "CentralAir": central_air
        }

        with st.spinner("Calculating valuation..."):
            result = predictor.predict(user_inputs)

        st.markdown("---")
        st.markdown("### 🏷️ Estimated Property Valuation Result")

        r1, r2, r3 = st.columns(3)
        with r1:
            st.metric("Estimated Price (USD)", f"${result['predicted_price_usd']:,.2f}")
        with r2:
            st.metric("Estimated Price (INR)", f"₹{result['predicted_price_inr']:,.2f}")
        with r3:
            st.metric("Model Utilized", result["model_used"])

        st.info(f"💡 **Currency Explanation:** {result['currency_explanation']}")

        # Derived property insights
        dm = result["derived_metrics"]
        st.markdown("#### 📐 Inferred Property Dimensions")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Calculated Total SF", f"{dm['TotalSF']:,.0f} sq ft")
        k2.metric("Total Bathrooms", f"{dm['TotalBath']} baths")
        k3.metric("Property Age", f"{dm['HouseAge']} years")
        k4.metric("Rate / Sq Ft (USD)", f"${dm['PricePerSqFt_USD']:.2f}")

        # Downloadable Prediction Report
        pred_record = {**user_inputs, **result}
        pred_record.pop("currency_explanation", None)
        pred_df = pd.DataFrame([pred_record])
        csv_data = pred_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Prediction Summary (CSV)",
            data=csv_data,
            file_name=f"house_price_prediction_{selected_model.replace(' ', '_')}.csv",
            mime="text/csv",
            use_container_width=True
        )


# ==============================================================================
# PAGE 12: PROJECT CONCLUSION
# ==============================================================================
elif selected_page == "12. Project Conclusion":
    st.markdown("## 🎓 Project Conclusion & Technical Viva Preparation")
    st.write("Summary of academic and engineering deliverables, viva questions, and production deployment roadmap.")

    st.markdown("### 📌 Key Project Findings")
    st.markdown("""
    1. **Feature Engineering Dominance**: Deriving composite spatial variables (`TotalSF`) and sanitary aggregations (`TotalBath`) boosted predictive signal more significantly than simply adding raw columns.
    2. **Algorithmic Hierarchy**: Gradient boosted architectures (**Tuned Gradient Boosting** with Test $R^2 = 0.9279$ and **XGBoost** with $R^2 = 0.9236$) decisively outperformed single decision trees ($R^2 = 0.8399$) and linear baselines ($R^2 = 0.9010$).
    3. **Impact of Outliers**: Treating Dean De Cock's documented high-leverage partial sales (>4,000 sq ft) prevented severe gradient distortion.
    4. **Hyperparameter Optimization**: Systematic cross-validated tuning improved Gradient Boosting test error by **+17.27%**.
    """)

    st.markdown("### 🎙️ University Viva Voce Preparation (Q&A)")
    viva_qa = [
        (
            "Q1: Why did you select the Ames Housing dataset instead of Boston Housing?",
            "A: The Boston Housing dataset has been officially deprecated by scikit-learn due to ethical issues and limited dimensionality (only 506 rows and 13 variables). The Ames dataset offers 1,460+ records, 81 high-dimensional features, modern construction attributes, and represents genuine residential transactions in Ames, Iowa."
        ),
        (
            "Q2: How did you prevent data leakage during preprocessing?",
            "A: We split the dataset into 80% train and 20% test partitions before fitting any preprocessing components. Imputation statistics (e.g. median values) and scaling parameters were computed strictly on the training partition and then applied blindly via `.transform()` to the test set."
        ),
        (
            "Q3: Why does Decision Tree overfit while Random Forest and Gradient Boosting generalize well?",
            "A: A single decision tree will recursively partition nodes until leaf purity is attained, memorizing idiosyncrasies and noise in the training set (Train R² 0.981 vs CV R² 0.749). Random Forest combats this through bootstrap bagging and feature subspace sampling, while Gradient Boosting uses shallow trees constrained by a small learning rate (shrinkage) and regularization."
        ),
        (
            "Q4: What is the mathematical difference between MAE and RMSE?",
            "A: MAE is the arithmetic mean of absolute errors (L1 norm), weighting all discrepancies proportionally. RMSE is the square root of the mean squared errors (L2 norm). By squaring errors first, RMSE penalizes large prediction errors more severely."
        )
    ]
    for q, a in viva_qa:
        with st.expander(f"🗣️ **{q}**"):
            st.write(a)

    st.markdown("### 🚀 Limitations & Future Scope")
    st.markdown("""
    - **Temporal Scope**: The Ames data spans 2006 to 2010. Real-world commercial deployment requires dynamic macro-economic inflation adjustments and mortgage interest rate feeds.
    - **Geospatial Coordinates**: Future enhancements could integrate exact GPS coordinates to compute distance to schools, parks, and downtown amenities.
    - **Computer Vision**: Incorporating property inspection photos using convolutional neural networks (CNNs) or Vision Transformers (ViTs) to augment tabular quality ratings.
    """)
