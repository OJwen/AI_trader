import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import time

# Step 1: Fetch Real-Time Data with Yahoo Finance
def fetch_real_time_data(symbol):
    ticker = yf.Ticker(symbol)
    latest_price = ticker.history(period="1d")["Close"].iloc[-1]
    return latest_price

# Step 2: Preload Historical Data and Train AI Model
def train_ai_model():
    # Load historical data
    historical_data = pd.read_csv("C:/Users/owenj/OneDrive - Saint Kentigern/Desktop/AI_trader/NVDA_stock_data.csv", index_col="date", parse_dates=True)

    # Normalize the "4. close" column
    scaler = MinMaxScaler()
    historical_data["Normalized Close"] = scaler.fit_transform(historical_data[["4. close"]])

    # Add lagged features for previous 5 days
    for i in range(1, 6):
        historical_data[f"lag_{i}"] = historical_data["Normalized Close"].shift(i)

    # Drop rows with NaN values
    historical_data = historical_data.dropna()

    # Features: Previous 5 days
    features = historical_data[[f"lag_{i}" for i in range(1, 6)]].values
    # Target: Current day's normalized closing price
    targets = historical_data["Normalized Close"].values

    # Train a Random Forest model
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(features, targets)

    return model, scaler

# Step 3: Generate Buy/Sell Signals
def generate_signal(latest_price, predicted_price, threshold=0.005):
    if predicted_price > latest_price * (1 + threshold):
        return "Buy"
    elif predicted_price < latest_price * (1 - threshold):
        return "Sell"
    else:
        return "Hold"

# Step 4: Real-Time Trading Loop
def real_time_trading(symbol, model, scaler):
    print("Starting real-time trading...")
    while True:
        # Fetch real-time data
        latest_price = fetch_real_time_data(symbol)

        # Prepare input for prediction
        normalized_price = scaler.transform([[latest_price]])[0][0]
        predicted_price = model.predict([[normalized_price]] * 5)[0]  # Mock features for simplicity

        # Generate trading signal
        signal = generate_signal(latest_price, predicted_price)
        print(f"Latest Price: ${latest_price:.2f}, Predicted Price: ${predicted_price:.2f}, Signal: {signal}")

        # Wait for the next interval
        time.sleep(60)  # Wait for 1 minute before fetching new data

# Step 5: Run the AI Trader
if __name__ == "__main__":
    # Train the AI model using historical data
    model, scaler = train_ai_model()

    # Start real-time trading for NVIDIA (NVDA)
    real_time_trading("NVDA", model, scaler)