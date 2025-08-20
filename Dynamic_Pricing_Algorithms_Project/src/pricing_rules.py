import pandas as pd
import numpy as np

from src.elasticity_model import load_data as load_elasticity_data, fit_elasticity_model
from src.demand_forecast import load_data as load_forecast_data, train_forecast_model


def define_pricing_rules(df: pd.DataFrame, elasticity: float, forecast: pd.DataFrame):
    """
    Define pricing rules based on:
    - Elasticity (how sensitive demand is to price changes)
    - Forecasted demand (expected future demand trends)
    - Utilization thresholds
    """

    # Merge actual demand with forecast
    df_rules = df.copy()
    df_rules["BookingEndDateTime"] = pd.to_datetime(df_rules["BookingEndDateTime"], errors="coerce")

    forecast_short = forecast[["ds", "yhat"]].rename(columns={"ds": "BookingEndDateTime", "yhat": "forecast_demand"})
    forecast_short["BookingEndDateTime"] = pd.to_datetime(forecast_short["BookingEndDateTime"], errors="coerce")

    # Merge actual demand with forecast
    df_rules = pd.merge(df_rules, forecast_short, on="BookingEndDateTime", how="left")

    # Utilization metric
    df_rules["utilization"] = df_rules["Number Booked"] / df_rules["MaxBookees"]

    # Base price
    df_rules["recommended_price"] = df_rules["Price (INR)"]

    # -------------------------
    # Rule 1: Demand elasticity
    # -------------------------
    # If elasticity < -1 (elastic demand), avoid high price hikes
    if elasticity < -1:
        df_rules.loc[df_rules["utilization"] > 0.85, "recommended_price"] *= 1.10
        df_rules.loc[df_rules["utilization"] < 0.50, "recommended_price"] *= 0.90
    else:  # inelastic demand → users less sensitive to price
        df_rules.loc[df_rules["utilization"] > 0.85, "recommended_price"] *= 1.20
        df_rules.loc[df_rules["utilization"] < 0.50, "recommended_price"] *= 0.95

    # -------------------------
    # Rule 2: Forecasted demand
    # -------------------------
    df_rules.loc[df_rules["forecast_demand"] > df_rules["Number Booked"].mean() * 1.2, "recommended_price"] *= 1.10
    df_rules.loc[df_rules["forecast_demand"] < df_rules["Number Booked"].mean() * 0.8, "recommended_price"] *= 0.90

    # -------------------------
    # Rule 3: Time-based pricing
    # -------------------------
    df_rules["hour"] = pd.to_datetime(df_rules["BookingEndDateTime"]).dt.hour
    peak_hours = (df_rules["hour"].between(8, 11)) | (df_rules["hour"].between(17, 20))
    off_peak = (df_rules["hour"].between(12, 16)) | (df_rules["hour"] >= 21)

    df_rules.loc[peak_hours, "recommended_price"] *= 1.10
    df_rules.loc[off_peak, "recommended_price"] *= 0.90

    return df_rules


def run_pricing_rules():
    print("\n⚡ Running Dynamic Pricing Rules...")

    # Step 1: Elasticity
    df_elasticity = load_elasticity_data("data/Cleaned_Fitness_Classes_Data.csv")
    model, df_elasticity, metrics = fit_elasticity_model(df_elasticity)
    elasticity = metrics["Elasticity"]

    # Step 2: Forecast
    df_forecast = load_forecast_data("data/Cleaned_Fitness_Classes_Data.csv")
    model_f, forecast, test, forecast_metrics = train_forecast_model(df_forecast)

    # Step 3: Pricing rules
    df_rules = define_pricing_rules(df_elasticity, elasticity, forecast)

    # Save output
    output_path = "output/pricing_recommendations.csv"

    # Ensure recommended_price is float
    df_rules["recommended_price"] = df_rules["recommended_price"].astype(float)

    # Match actual CSV column names
    output_cols = [
        "BookingEndDateTime",
        "ActivityDescription",
        "ActivitySiteID",
        "Price (INR)",
        "recommended_price"
    ]

    df_rules[output_cols].to_csv(output_path, index=False)
    print(f"✅ Pricing recommendations saved to {output_path}")
    return df_rules


if __name__ == "__main__":
    recommendations = run_pricing_rules()
    print(recommendations.head())
