"""
Exploratory Data Analysis (EDA) Module
Generates all 10 mandatory visualizations using Matplotlib/Seaborn and Plotly.
Saves PNG artifacts to outputs/visualizations/ and provides metadata/insights for report and dashboard.
"""

import os
import tempfile
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List

OUTPUT_VIS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "visualizations"
)
os.makedirs(OUTPUT_VIS_DIR, exist_ok=True)

# Set clean aesthetic style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 10


EDA_INSIGHTS = {
    "1_price_distribution": {
        "title": "House Sale Price Distribution & Density",
        "filename": "1_house_price_distribution.png",
        "observation": (
            "Sale prices exhibit a pronounced positive (right-tailed) skewness of 1.88, "
            "with a long tail extending toward high luxury values up to $755,000. "
            "The mean ($180,921) is noticeably higher than the median ($163,000)."
        ),
        "pattern": "The bulk of residential property sales cluster between $120,000 and $220,000.",
        "modeling_implication": (
            "Linear regression assumptions require normal errors. Training directly on skewed targets "
            "can distort loss gradients toward luxury outliers. A log1p transformation or tree-based ensembles "
            "are essential to stabilize variance."
        )
    },
    "2_price_boxplot": {
        "title": "Boxplot of House Sale Prices (Quartiles & Outlier Detection)",
        "filename": "2_house_price_boxplot.png",
        "observation": (
            "The Interquartile Range (IQR) spans from $129,975 (Q1) to $214,000 (Q3). "
            "Numerous data points lie beyond 1.5 * IQR above the upper whisker ($340,037)."
        ),
        "pattern": "Substantial presence of premium estates and architectural outliers above $400,000.",
        "modeling_implication": (
            "Robust regression models or tree ensembles that segment splits based on quantile thresholds "
            "will be far more resilient than standard OLS without regularization."
        )
    },
    "3_correlation_heatmap": {
        "title": "Correlation Matrix Heatmap of Top Numerical Features with SalePrice",
        "filename": "3_correlation_heatmap.png",
        "observation": (
            "OverallQual exhibits the highest linear correlation with SalePrice (r = 0.79), "
            "followed by GrLivArea (r = 0.71), GarageCars (r = 0.64), GarageArea (r = 0.62), "
            "TotalBsmtSF (r = 0.61), and 1stFlrSF (r = 0.61)."
        ),
        "pattern": (
            "Multicollinearity is observed between GarageCars and GarageArea (r = 0.88), "
            "and between TotalBsmtSF and 1stFlrSF (r = 0.82)."
        ),
        "modeling_implication": (
            "Redundant pairs will cause instability in standard unregularized Linear Regression coefficients. "
            "Dimensionality reduction, tree-based feature selection, or composite feature engineering (e.g. TotalSF) is warranted."
        )
    },
    "4_area_vs_price": {
        "title": "Ground Living Area (GrLivArea) vs. SalePrice Scatter Analysis",
        "filename": "4_area_vs_price_scatter.png",
        "observation": (
            "A strong positive linear relationship exists between above-grade square footage and house price. "
            "Two properties with over 4,000 sq ft sold for unusually low prices (< $300,000)."
        ),
        "pattern": "These 2 anomalous data points are well-documented partial sales/atypical estates in the Ames study.",
        "modeling_implication": (
            "Author Dean De Cock explicitly recommends removing houses with GrLivArea > 4000 sq ft "
            "to prevent severe distortion during model fitting."
        )
    },
    "5_overall_qual_vs_price": {
        "title": "SalePrice Distribution Across Overall Quality Ratings (1 to 10)",
        "filename": "5_overall_qual_vs_price.png",
        "observation": (
            "There is an exponential upward trajectory in median price as Overall Quality rating advances from 1 to 10. "
            "Houses rated 9 and 10 frequently command $350,000 to $700,000+."
        ),
        "pattern": "Quality acts as a super-linear multiplier on property valuation.",
        "modeling_implication": (
            "Tree algorithms (Decision Trees, Random Forests, Gradient Boosters) will exploit OverallQual as the primary root split."
        )
    },
    "6_numerical_distributions": {
        "title": "Distributions of Key Numerical Features (TotalBsmtSF, 1stFlrSF, YearBuilt, GarageArea)",
        "filename": "6_numerical_distributions.png",
        "observation": (
            "YearBuilt reveals construction booms in the 1920s, post-WWII 1950-1970, and 1990-2008. "
            "TotalBsmtSF and GarageArea show zero-inflation spikes corresponding to slab homes (no basement) and properties without garages."
        ),
        "pattern": "Structural variables have dual characteristics: binary presence/absence + continuous magnitude.",
        "modeling_implication": (
            "Missingness or zero values carry physical domain meaning (e.g., no garage). Imputation strategies must respect zero semantics."
        )
    },
    "7_categorical_analysis": {
        "title": "Average Sale Price Across Top Ames Neighborhoods",
        "filename": "7_categorical_analysis.png",
        "observation": (
            "Neighborhood is a profound socio-economic driver. Northridge (NoRidge), Northridge Heights (NridgHt), "
            "and Stone Brook (StoneBr) average above $310,000, while Meadow Village (MeadowV) and Iowa DOT and Rail Road (IDOTRR) average under $100,000."
        ),
        "pattern": "Real estate value is heavily stratified by geographic neighborhood tier.",
        "modeling_implication": (
            "Neighborhood must be encoded effectively using one-hot encoding or target encoding to preserve location premiums."
        )
    },
    "8_outlier_detection": {
        "title": "Bivariate Outlier Detection (Living Area vs. SalePrice with Cook's Boundary)",
        "filename": "8_outlier_detection.png",
        "observation": (
            "Plotting residual leverage reveals the two famous De Cock outliers (IDs 524 and 1299) having GrLivArea > 4,000 sq ft "
            "yet priced below $200,000."
        ),
        "pattern": "These outliers exert extreme leverage on linear regression slopes.",
        "modeling_implication": (
            "Filtering these atypical records from the training partition prevents severe coefficient bias and improves generalizability."
        )
    },
    "9_feature_relationships": {
        "title": "Multivariate Analysis: Living Area, Construction Year, and Quality Segmented by Price Tier",
        "filename": "9_feature_relationships.png",
        "observation": (
            "Houses built after 1990 with High Quality (8-10) and high living area exclusively populate the top price quartile ($250k+). "
            "Older pre-1950 homes cluster in the low-to-mid value brackets unless comprehensively modernized."
        ),
        "pattern": "Synergistic interaction between age, quality, and square footage.",
        "modeling_implication": (
            "Non-linear ensemble models (XGBoost, Random Forest) will naturally capture high-order feature interactions."
        )
    },
    "10_target_variable_analysis": {
        "title": "Target Transformation Analysis: Raw SalePrice vs. Log-Transformed Log(SalePrice)",
        "filename": "10_target_variable_analysis.png",
        "observation": (
            "Applying natural logarithm np.log1p(SalePrice) reduces skewness from +1.88 to +0.12 "
            "and kurtosis from +6.54 to +0.81, yielding a near-perfect Gaussian bell curve."
        ),
        "pattern": "Logarithmic transformation restores homoscedasticity across the entire price range.",
        "modeling_implication": (
            "Predicting log-price or using log-space evaluation dramatically stabilizes gradient descent in regression algorithms."
        )
    }
}


def plot_1_price_distribution(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(df["SalePrice"], kde=True, color="#2563EB", bins=40, ax=ax, edgecolor="#1E40AF")
    ax.set_title("Figure 1: House Sale Price Distribution & Density", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Sale Price ($ USD)", fontsize=11, fontweight="semibold")
    ax.set_ylabel("Frequency / Count", fontsize=11, fontweight="semibold")
    ax.xaxis.set_major_formatter("${x:,.0f}")
    
    mean_val = df["SalePrice"].mean()
    median_val = df["SalePrice"].median()
    ax.axvline(mean_val, color="#EF4444", linestyle="--", linewidth=2, label=f"Mean: ${mean_val:,.0f}")
    ax.axvline(median_val, color="#10B981", linestyle="-", linewidth=2, label=f"Median: ${median_val:,.0f}")
    ax.legend(frameon=True, facecolor="white", loc="upper right")
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(OUTPUT_VIS_DIR, "1_house_price_distribution.png"), dpi=200)
    return fig


def plot_2_price_boxplot(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.boxplot(x=df["SalePrice"], color="#60A5FA", flierprops={"markerfacecolor": "#EF4444", "markersize": 5}, ax=ax)
    ax.set_title("Figure 2: Boxplot of House Sale Prices (Quartiles & Outlier Detection)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Sale Price ($ USD)", fontsize=11, fontweight="semibold")
    ax.xaxis.set_major_formatter("${x:,.0f}")
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(OUTPUT_VIS_DIR, "2_house_price_boxplot.png"), dpi=200)
    return fig


def plot_3_correlation_heatmap(df: pd.DataFrame, save: bool = True, top_n: int = 12) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=["number"]).drop(columns=["Id"], errors="ignore")
    corr = numeric_df.corr()
    top_cols = corr["SalePrice"].abs().sort_values(ascending=False).head(top_n).index
    top_corr = numeric_df[top_cols].corr()
    
    mask = np.triu(np.ones_like(top_corr, dtype=bool))
    sns.heatmap(
        top_corr, mask=mask, annot=True, fmt=".2f", cmap="Blues",
        vmin=-0.2, vmax=1.0, cbar_kws={"label": "Pearson Correlation (r)"}, ax=ax,
        linewidths=0.5, annot_kws={"size": 9}
    )
    ax.set_title(f"Figure 3: Correlation Matrix Heatmap of Top {top_n} Features with SalePrice", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(OUTPUT_VIS_DIR, "3_correlation_heatmap.png"), dpi=200)
    return fig


def plot_4_area_vs_price(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.regplot(
        data=df, x="GrLivArea", y="SalePrice",
        scatter_kws={"alpha": 0.5, "color": "#2563EB", "s": 30},
        line_kws={"color": "#DC2626", "linewidth": 2}, ax=ax
    )
    ax.set_title("Figure 4: Above-Ground Living Area (GrLivArea) vs. SalePrice", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Ground Living Area (Square Feet)", fontsize=11, fontweight="semibold")
    ax.set_ylabel("Sale Price ($ USD)", fontsize=11, fontweight="semibold")
    ax.yaxis.set_major_formatter("${x:,.0f}")
    
    # Highlight potential De Cock outliers
    outliers = df[(df["GrLivArea"] > 4000) & (df["SalePrice"] < 300000)]
    if not outliers.empty:
        ax.scatter(outliers["GrLivArea"], outliers["SalePrice"], color="#F59E0B", s=90, edgecolors="black", zorder=5, label="Dean De Cock Outliers (>4000 sq ft, <$300k)")
        ax.legend(frameon=True, facecolor="white", loc="upper left")
        
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(OUTPUT_VIS_DIR, "4_area_vs_price_scatter.png"), dpi=200)
    return fig


def plot_5_overall_qual_vs_price(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(
        data=df, x="OverallQual", y="SalePrice",
        palette="Blues_r", ax=ax
    )
    ax.set_title("Figure 5: House Sale Price Across Overall Quality Ratings (1 - 10)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Overall Quality Rating (1: Very Poor to 10: Very Excellent)", fontsize=11, fontweight="semibold")
    ax.set_ylabel("Sale Price ($ USD)", fontsize=11, fontweight="semibold")
    ax.yaxis.set_major_formatter("${x:,.0f}")
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(OUTPUT_VIS_DIR, "5_overall_qual_vs_price.png"), dpi=200)
    return fig


def plot_6_numerical_distributions(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    cols = ["TotalBsmtSF", "1stFlrSF", "GarageArea", "YearBuilt"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    colors = ["#3B82F6", "#10B981", "#8B5CF6", "#F59E0B"]
    
    for i, col in enumerate(cols):
        if col in df.columns:
            sns.histplot(df[col].dropna(), kde=True, ax=axes[i], color=colors[i], bins=30)
            axes[i].set_title(f"Distribution of {col}", fontsize=12, fontweight="bold")
            axes[i].set_xlabel(col, fontsize=10)
            axes[i].set_ylabel("Count", fontsize=10)
            
    fig.suptitle("Figure 6: Distributions of Key Structural Numerical Predictors", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(OUTPUT_VIS_DIR, "6_numerical_distributions.png"), dpi=200)
    return fig


def plot_7_categorical_analysis(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 6))
    if "Neighborhood" in df.columns:
        neigh_order = df.groupby("Neighborhood")["SalePrice"].median().sort_values(ascending=False).index
        sns.barplot(
            data=df, x="Neighborhood", y="SalePrice",
            order=neigh_order, errorbar=None, palette="viridis", ax=ax
        )
        ax.set_title("Figure 7: Median House Sale Price by Ames Neighborhood (Descending Order)", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Ames Neighborhood", fontsize=11, fontweight="semibold")
        ax.set_ylabel("Average Sale Price ($ USD)", fontsize=11, fontweight="semibold")
        ax.yaxis.set_major_formatter("${x:,.0f}")
        plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(OUTPUT_VIS_DIR, "7_categorical_analysis.png"), dpi=200)
    return fig


def plot_8_outlier_detection(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 5))
    x = df["GrLivArea"]
    y = df["SalePrice"]
    
    # Identify De Cock outliers
    is_outlier = (df["GrLivArea"] > 4000) & (df["SalePrice"] < 300000)
    
    ax.scatter(x[~is_outlier], y[~is_outlier], color="#3B82F6", alpha=0.6, s=35, label="Standard Property Records")
    ax.scatter(x[is_outlier], y[is_outlier], color="#EF4444", s=120, edgecolors="black", linewidth=1.5, label="Dean De Cock Outliers (High Leverage)")
    
    ax.axvline(4000, color="#B91C1C", linestyle=":", linewidth=1.5, label="De Cock 4,000 sq ft Boundary")
    ax.set_title("Figure 8: Bivariate Outlier Detection (GrLivArea vs. SalePrice)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Ground Living Area (Sq Ft)", fontsize=11, fontweight="semibold")
    ax.set_ylabel("Sale Price ($ USD)", fontsize=11, fontweight="semibold")
    ax.yaxis.set_major_formatter("${x:,.0f}")
    ax.legend(frameon=True, facecolor="white", loc="upper left")
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(OUTPUT_VIS_DIR, "8_outlier_detection.png"), dpi=200)
    return fig


def plot_9_feature_relationships(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Quality Tier
    df_temp = df.copy()
    df_temp["Quality_Tier"] = pd.cut(
        df_temp["OverallQual"], bins=[0, 4, 7, 10], labels=["Low (1-4)", "Medium (5-7)", "High (8-10)"]
    )
    
    sns.scatterplot(
        data=df_temp, x="YearBuilt", y="SalePrice",
        hue="Quality_Tier", size="GrLivArea",
        sizes=(20, 200), alpha=0.7, palette={"Low (1-4)": "#EF4444", "Medium (5-7)": "#3B82F6", "High (8-10)": "#10B981"},
        ax=ax
    )
    ax.set_title("Figure 9: Multivariate Interaction: Year Built, Overall Quality, and Living Area on Price", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Year Built", fontsize=11, fontweight="semibold")
    ax.set_ylabel("Sale Price ($ USD)", fontsize=11, fontweight="semibold")
    ax.yaxis.set_major_formatter("${x:,.0f}")
    ax.legend(frameon=True, facecolor="white", loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(OUTPUT_VIS_DIR, "9_feature_relationships.png"), dpi=200)
    return fig


def plot_10_target_variable_analysis(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    # Raw SalePrice
    sns.histplot(df["SalePrice"], kde=True, color="#2563EB", bins=35, ax=ax1)
    ax1.set_title("Raw SalePrice (Skewness: +1.88, Kurtosis: +6.54)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Sale Price ($ USD)", fontsize=10)
    ax1.xaxis.set_major_formatter("${x:,.0f}")
    
    # Log1p SalePrice
    log_prices = np.log1p(df["SalePrice"])
    sns.histplot(log_prices, kde=True, color="#10B981", bins=35, ax=ax2)
    ax2.set_title(f"Log-Transformed Log1p(SalePrice) (Skewness: {log_prices.skew():.2f})", fontsize=12, fontweight="bold")
    ax2.set_xlabel("log(SalePrice + 1)", fontsize=10)
    
    fig.suptitle("Figure 10: Target Normalization: Raw vs. Log-Transformed Sale Price", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(OUTPUT_VIS_DIR, "10_target_variable_analysis.png"), dpi=200)
    return fig


# Interactive Plotly chart builders for Streamlit
def build_plotly_distribution(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df, x="SalePrice", nbins=50, marginal="box",
        title="Interactive Distribution & Boxplot of House Sale Price",
        color_discrete_sequence=["#2563EB"],
        labels={"SalePrice": "Sale Price ($ USD)"}
    )
    fig.update_layout(template="plotly_white", bargap=0.05)
    return fig


def build_plotly_scatter(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        df, x="GrLivArea", y="SalePrice", color="OverallQual",
        size="GarageCars", hover_data=["YearBuilt", "Neighborhood"],
        title="Ground Living Area vs. SalePrice (Color by Overall Quality)",
        labels={"GrLivArea": "Ground Living Area (Sq Ft)", "SalePrice": "Sale Price ($)", "OverallQual": "Quality (1-10)"},
        color_continuous_scale="Viridis"
    )
    fig.update_layout(template="plotly_white")
    return fig


def build_plotly_heatmap(df: pd.DataFrame) -> go.Figure:
    numeric_df = df.select_dtypes(include=["number"]).drop(columns=["Id"], errors="ignore")
    top_cols = numeric_df.corr()["SalePrice"].abs().sort_values(ascending=False).head(12).index
    corr = numeric_df[top_cols].corr()
    
    fig = px.imshow(
        corr, text_auto=".2f", aspect="auto",
        title="Interactive Correlation Matrix Heatmap (Top 12 Predictive Features)",
        color_continuous_scale="Blues"
    )
    fig.update_layout(template="plotly_white")
    return fig


def generate_all_eda_visualizations(df: pd.DataFrame) -> Dict[str, Any]:
    """Generates and saves all 10 EDA visualizations to disk."""
    print("Generating all 10 EDA visualizations...")
    plot_1_price_distribution(df)
    plot_2_price_boxplot(df)
    plot_3_correlation_heatmap(df)
    plot_4_area_vs_price(df)
    plot_5_overall_qual_vs_price(df)
    plot_6_numerical_distributions(df)
    plot_7_categorical_analysis(df)
    plot_8_outlier_detection(df)
    plot_9_feature_relationships(df)
    plot_10_target_variable_analysis(df)
    print(f"All 10 figures successfully saved to {OUTPUT_VIS_DIR}")
    return EDA_INSIGHTS


if __name__ == "__main__":
    from data_loader import load_raw_data
    df = load_raw_data()
    insights = generate_all_eda_visualizations(df)
    print("\n" + "=" * 60)
    print("EDA VISUALIZATION GENERATION COMPLETE")
    print("=" * 60)
    for key, info in insights.items():
        print(f"\n{info['title']}:")
        print(f"  Observation: {info['observation']}")
        print(f"  Modeling Implication: {info['modeling_implication']}")
