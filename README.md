# 🚀 Crypto Insights Dashboard

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

An AI-powered cryptocurrency analytics platform that automates data retrieval, performs technical analysis, and provides machine learning-driven price predictions and investment recommendations.

## 🌟 Features

* **Real-time Market Data:** ETL pipeline integrating CoinGecko API for live price, volume, and market cap tracking.
* **Technical Analysis Engine:** Automated calculation of RSI, MACD, Bollinger Bands, and Volatility metrics.
* **AI Price Forecasting:** Uses **Gradient Boosting Regressor** to predict short-term price movements.
* **Smart Recommendations:** Implements **K-Means Clustering** to categorize assets by risk profile (Low/Medium/High) and suggest portfolio additions.
* **Interactive Dashboard:** Built with Streamlit for dynamic charting and portfolio management.

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-learn (GradientBoostingRegressor, KMeans, StandardScaler)
* **Visualization:** Matplotlib, Plotly
* **API:** CoinGecko

## 🚀 Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/crypto-insights-dashboard.git](https://github.com/yourusername/crypto-insights-dashboard.git)
    cd crypto-insights-dashboard
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application:**
    ```bash
    streamlit run app.py
    ```

## 📊 How It Works

1.  **Market Overview:** View top 250 cryptocurrencies with real-time sentiment analysis.
2.  **Portfolio Manager:** Add assets to track value. The system runs a regression model on your holdings to forecast next-day trends.
3.  **AI Recommendations:** Select your risk tolerance. The system clusters coins based on volatility and price trends to recommend assets that fit your profile.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.