import pandas as pd

# Load the CSV file
data = pd.read_csv("NVDA_stock_data.csv", index_col="date", parse_dates=True)

# Display the first few rows of the data
print(data.head())

# Summary statistics for a quick overview
print(data.describe())

# Plot closing prices
import matplotlib.pyplot as plt

# Plot the closing prices
data["4. close"].plot(figsize=(10, 5), title="Closing Price Over Time")
plt.xlabel("Date")
plt.ylabel("Closing Price (USD)")
plt.grid()
plt.show()