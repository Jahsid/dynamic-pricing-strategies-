from src.elasticity_model import load_data as load_elasticity_data, fit_elasticity_model, save_results
from src.demand_forecast import load_data as load_forecast_data, train_forecast_model, save_forecast
from src.pricing_rules import run_pricing_rules

def run_elasticity_analysis():
    print("\n🚀 Running Price Elasticity Model...")
    df = load_elasticity_data("data/Cleaned_Fitness_Classes_Data.csv")
    model, df, metrics = fit_elasticity_model(df)
    save_results(df, metrics)


def run_demand_forecast():
    print("\n📈 Running Demand Forecasting Model...")
    df = load_forecast_data("data/Cleaned_Fitness_Classes_Data.csv")
    model, forecast, test, metrics = train_forecast_model(df)
    save_forecast(forecast)

    print("\n📊 Forecast Validation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.3f}")


def main():
    # Step 1: Elasticity analysis
    run_elasticity_analysis()

    # Step 2: Demand forecasting
    run_demand_forecast()

    # Step 3: Dynamic Pricing
    run_pricing_rules()

if __name__ == "__main__":
    main()
