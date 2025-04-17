from alpha_vantage.timeseries import TimeSeries
import pandas as pd

# Your API key
api_key = "hidden"

# Initialize TimeSeries object
ts = TimeSeries(key=api_key, output_format="pandas")

# Fetch daily stock data (example: Microsoft)
symbol = "NVDA"
data, meta_data = ts.get_daily(symbol=symbol, outputsize="full")

# Display the first few rows
print(data.head())

# Save data to CSV for later use
data.to_csv(f"{symbol}_stock_data.csv")
print(f"Data saved to {symbol}_stock_data.csv")
