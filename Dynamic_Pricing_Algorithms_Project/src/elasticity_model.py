import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error


def load_data(path: str):
    """Load and preprocess dataset."""
    df = pd.read_csv(path)
    df["log_price"] = np.log(df["Price (INR)"])
    df["log_demand"] = np.log(df["Number Booked"])
    return df


def fit_elasticity_model(df: pd.DataFrame):
    """Fit log-log regression model and calculate elasticity."""
    X = sm.add_constant(df["log_price"])
    y = df["log_demand"]

    model = sm.OLS(y, X).fit()

    # Elasticity (slope of log_price)
    elasticity = model.params["log_price"]

    # Predictions
    df["predicted_log_demand"] = model.predict(X)
    df["predicted_demand"] = np.exp(df["predicted_log_demand"])

    # Metrics
    mse = mean_squared_error(df["Number Booked"], df["predicted_demand"])
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(df["Number Booked"], df["predicted_demand"])
    mape = np.mean(
        np.abs((df["Number Booked"] - df["predicted_demand"]) / df["Number Booked"])
    ) * 100

    metrics = {
        "R-squared": model.rsquared,
        "Adj R-squared": model.rsquared_adj,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "Elasticity": elasticity,
    }

    return model, df, metrics


def save_results(df: pd.DataFrame, metrics: dict, out_path="output/elasticity_output.csv"):
    """Save predictions + metrics."""
    df[["Price (INR)", "Number Booked", "predicted_demand"]].to_csv(out_path, index=False)

    print(f"✅ Predictions saved to {out_path}")
    print("\n📊 Model Validation Metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.3f}")
        else:
            print(f"{k}: {v}")
