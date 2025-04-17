# **AI Stock Trading Project with Real-Time Integration**

## **Overview**
This project leverages machine learning and real-time data integration to build an AI-powered stock trading system. Using historical data and real-time stock prices, the model predicts future price movements and generates actionable buy/sell signals based on dynamic thresholds. The system is optimized for NVIDIA stock (symbol: NVDA) and integrates **Yahoo Finance** for live trading data.

## **Motivation**
The goal of this project is to harness the power of AI to predict stock prices accurately and use actionable trading strategies to maximize returns. This initiative has enabled:
- Learning advanced machine learning techniques.
- Understanding financial data preprocessing and trading strategies.
- Developing real-time trading systems that generate buy/sell signals dynamically.

## **Project Progress**
### **1. Historical Data Analysis**
- Used NVIDIA historical stock data to build and train a Random Forest Regressor.
- Added normalized features (MinMaxScaler) and lagged historical prices for better predictive accuracy.
- Successfully evaluated model performance with metrics:
  - **Mean Squared Error (MSE):** 3.4050e-05
  - **Mean Absolute Error (MAE):** 0.0026
  - **R² Score:** 0.9983 (highly accurate predictions)

### **2. Trading Signals and Backtesting**
- Incorporated trading logic to define clear buy, sell, and hold signals based on predicted vs actual prices.
- Backtested strategies with an initial balance of $10,000:
  - Final Portfolio Value: **$17,387.87**
  - **Cumulative Return:** 73.88%
- Used Python scripts to simulate trading performance over historical data.

### **3. Real-Time Data Integration**
- Integrated Yahoo Finance API (`yfinance`) to fetch live stock prices.
- Built a real-time trading loop to predict future prices and generate dynamic trading signals every minute.
- Automated "Buy," "Sell," and "Hold" recommendations based on a 0.5% price change threshold.

### **4. Challenges Faced**
- Encountered network restrictions while in China, requiring VPN support for API connectivity.
- Developed workarounds, including switching to Yahoo Finance for seamless data access.

## **Future Directions**
As this project evolves, the next steps include:
1. **Enhancing Trading Logic**:
   - Incorporate advanced indicators like RSI, Bollinger Bands, and MACD for improved decision-making.
   - Experiment with neural networks like LSTMs for time-series predictions.

2. **Expanding Data Scope**:
   - Apply the model to other stocks and sectors for diversification.
   - Explore multi-stock portfolio optimization.

3. **Live Trading Deployment**:
   - Integrate with trading platforms (e.g., Interactive Brokers) for fully automated execution.
   - Deploy a dashboard for interactive monitoring of signals.

4. **Exploration in Australia**:
   - Utilize unrestricted network access to experiment further with real-time integration and automation.

## **Technologies Used**
- **Python Libraries**: Pandas, scikit-learn, yfinance, matplotlib
- **Machine Learning Model**: Random Forest Regressor
- **APIs**: Yahoo Finance (for real-time data access)
- **Tools**: Jupyter Notebook, Visual Studio Code

## **Achievements**
- Designed an end-to-end AI trading system capable of real-time analysis.
- Achieved exceptional returns on backtesting simulations.
- Learned and implemented cutting-edge techniques in financial modeling.

## **Acknowledgment**
This journey exemplifies the fusion of machine learning and financial analysis to create tangible results. While in China, network limitations added unique challenges, which I overcame through adaptability and determination. As I move forward in Australia, I look forward to continuing this exploration and pushing the boundaries of AI in trading.

---

### **Closing Thoughts**
I’m thrilled to share this project on LinkedIn and connect with professionals interested in financial AI, machine learning, and trading automation. Feel free to reach out if you'd like to discuss or collaborate on similar initiatives! 🚀
