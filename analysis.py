# analysis.py
import pandas as pd
import numpy as np

def load_data(path="water_5states_300rows.csv"):
    """Load CSV dataset and ensure columns are correctly typed."""
    df = pd.read_csv(path)
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    return df

def compute_state_summary(df):
    """Aggregate by state to get water use efficiency and stress metrics."""
    grouped = (
        df.groupby("state", as_index=False)
        .agg({
            "data_centers": "mean",
            "total_water_use_mld": "mean",
            "population_million": "mean",
            "avg_temp_celsius": "mean",
            "precipitation_mm": "mean",
            "drought_index": "mean",
            "cooling_efficiency": "mean"
        })
    )
    # Derived indicators
    grouped["water_per_center"] = (
        grouped["total_water_use_mld"] / grouped["data_centers"]
    )
    grouped["water_per_capita"] = (
        grouped["total_water_use_mld"] / (grouped["population_million"] * 1e6)
    ) * 1e3  # litres per capita equivalent
    grouped["stress_score"] = (
        (grouped["drought_index"] * grouped["cooling_efficiency"]) /
        np.log1p(grouped["precipitation_mm"])
    ).round(3)
    return grouped

def compute_monthly_trends(df):
    """Aggregate by month for trend visualizations."""
    monthly = (
        df.groupby("month", as_index=False)
        .agg({
            "total_water_use_mld": "sum",
            "data_centers": "sum",
            "population_million": "mean",
            "drought_index": "mean"
        })
    )
    monthly["water_per_center"] = (
        monthly["total_water_use_mld"] / monthly["data_centers"]
    )
    return monthly

def get_sensitivity_data(df):
    """Estimate relationship between population growth and water stress."""
    df["stress_score"] = (
        (df["drought_index"] * df["cooling_efficiency"]) /
        np.log1p(df["precipitation_mm"])
    )
    pivot = (
        df.groupby("state")[["population_million", "stress_score"]]
        .mean()
        .reset_index()
    )
    return pivot

# Quick test mode for CLI
if __name__ == "__main__":
    df = load_data()
    summary = compute_state_summary(df)
    print("=== State Summary ===")
    print(summary.head(), "\n")

    monthly = compute_monthly_trends(df)
    print("=== Monthly Trends ===")
    print(monthly.head())
