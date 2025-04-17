import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Step 1: Load the data
data = pd.read_csv("C:/Users/owenj/OneDrive - Saint Kentigern/Desktop/AI_trader/NVDA_stock_data.csv", index_col="date", parse_dates=True)

# Step 2: Normalize the "4. close" column
scaler = MinMaxScaler()
data["Normalized Close"] = scaler.fit_transform(data[["4. close"]])

# Step 3: Add a 20-day moving average as a feature
data["20-Day MA"] = data["Normalized Close"].rolling(window=20).mean()

# Step 4: Create lagged features for previous 5 days
for i in range(1, 6):
    data[f"lag_{i}"] = data["Normalized Close"].shift(i)

# Drop rows with NaN values
data = data.dropna()

# Features: Previous 5 days + Moving Average
features = data[[f"lag_{i}" for i in range(1, 6)] + ["20-Day MA"]].values
# Target: Current day's normalized closing price
targets = data["Normalized Close"].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

# Step 5: Train a Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Step 6: Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R² Score: {r2}")

# Display the first few predictions vs actual values
test_data = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
print(test_data.head())

# Add actual stock prices to the test_data DataFrame
test_data["Close"] = data["4. close"].iloc[-len(test_data):].values  # Add actual prices

# Step 7: Define trading signals
threshold = 0.005  # Threshold for buy/sell signals (0.5% price change)
test_data["Signal"] = 0  # Default: Hold
test_data.loc[test_data["Predicted"] > test_data["Actual"] * (1 + threshold), "Signal"] = 1  # Buy Signal
test_data.loc[test_data["Predicted"] < test_data["Actual"] * (1 - threshold), "Signal"] = -1  # Sell Signal

# Step 8: Backtesting the strategy
initial_balance = 10000  # Starting with $10,000
balance = initial_balance
shares = 0

# Debugging: Track trades
print("Starting backtesting...")

# Simulate trading based on signals
for index, row in test_data.iterrows():
    if row["Signal"] == 1 and balance > 0:  # Buy Signal
        shares = balance / row["Close"]
        balance = 0  # Invest all balance
        print(f"Buying at {row['Close']:.2f}. Shares: {shares:.4f}")
    elif row["Signal"] == -1 and shares > 0:  # Sell Signal
        balance = shares * row["Close"]
        shares = 0  # Sell all shares
        print(f"Selling at {row['Close']:.2f}. Balance: ${balance:.2f}")

# Final Portfolio Value: Remaining balance + value of unsold shares
final_portfolio_value = balance + (shares * test_data["Close"].iloc[-1])
print(f"Final Portfolio Value: ${final_portfolio_value:.2f}")

# Calculate cumulative return
cumulative_return = (final_portfolio_value - initial_balance) / initial_balance
print(f"Cumulative Return: {cumulative_return:.2%}")

# Step 9: Visualize Actual vs Predicted values
plt.figure(figsize=(10, 5))
plt.plot(test_data.index, test_data["Actual"], label="Actual", alpha=0.8)
plt.plot(test_data.index, test_data["Predicted"], label="Predicted", alpha=0.8)
plt.legend()
plt.title("Actual vs Predicted Stock Prices")
plt.xlabel("Sample Index")
plt.ylabel("Normalized Price")
plt.grid()
plt.show()
