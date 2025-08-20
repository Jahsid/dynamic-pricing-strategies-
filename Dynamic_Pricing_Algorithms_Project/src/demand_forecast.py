from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd

def load_data(path: str):
    """Load cleaned dataset."""
    df = pd.read_csv(path)
    daily_demand = (
        df.groupby('BookingEndDateTime')['Number Booked']
        .sum()
        .reset_index()
    )
    daily_demand.rename(
        columns={'BookingEndDateTime': 'ds', 'Number Booked': 'y'},
        inplace=True
    )
    return daily_demand

def train_forecast_model(df, train_ratio=0.8):
    """Train Prophet model and return forecast + metrics."""
    train_size = int(len(df) * train_ratio)
    train, test = df[:train_size], df[train_size:]

    model = Prophet(daily_seasonality=True, weekly_seasonality=True)
    model.fit(train)

    future = model.make_future_dataframe(periods=len(test))
    forecast = model.predict(future)

    # Evaluate
    y_true = test['y'].values
    y_pred = forecast['yhat'].iloc[-len(test):].values
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    return model, forecast, test, {"MAE": mae, "RMSE": rmse}

def save_forecast(forecast, path="output/forecast_output.csv"):
    """Save forecasted values to CSV."""
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(path, index=False)
    print(f"✅ Forecast saved to {path}")
