import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests
from datetime import datetime, timedelta
import time
import pickle
import os
import warnings

# Machine Learning & Stats
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import ta  # Technical Analysis library

# Suppress warnings for cleaner UI
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Crypto Insights Dashboard",
    page_icon="🚀",
    layout="wide"
)

# Custom CSS for a professional look
st.markdown("""
    <style>
    .main { background-color: #f5f7fa; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 20px;
        color: #1e3a8a;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4e8df5;
        color: white;
    }
    h1, h2, h3 { color: #1e3a8a !important; }
    .metric-card {
        background-color: white;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 10px;
        border-left: 5px solid #4e8df5;
    }
    .recommendation-card {
        background-color: white;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        margin-bottom: 15px;
        border: 1px solid #e5e7eb;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Data Fetching Layer (CoinGecko API)
# -----------------------------------------------------------------------------
@st.cache_data(ttl=300)
def get_coin_list():
    """Fetches the top 250 coins by market cap."""
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        'vs_currency': 'usd',
        'order': 'market_cap_desc',
        'per_page': 250,
        'page': 1,
        'sparkline': False
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            return pd.DataFrame(response.json())
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching coin list: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_coin_price_history(coin_id, interval="1d"):
    """Fetches historical market data and calculates technical indicators."""
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    
    interval_map = {
        "1d": {"days": 30, "interval": "daily"},
        "1w": {"days": 90, "interval": "daily"},
        "1m": {"days": 180, "interval": "daily"},
        "1y": {"days": 365, "interval": "daily"}
    }
    
    params = {
        'vs_currency': 'usd',
        'days': interval_map[interval]['days'],
        'interval': interval_map[interval]['interval']
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Add volume if available
            volumes = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'volume'])
            df['volume'] = volumes['volume']
            
            # Technical Indicators
            if len(df) > 14:
                # RSI
                df['rsi'] = ta.momentum.RSIIndicator(df['price'], window=14).rsi()
                # MACD
                macd = ta.trend.MACD(df['price'])
                df['macd'] = macd.macd()
                df['macd_signal'] = macd.macd_signal()
                df['macd_diff'] = macd.macd_diff()
                # Bollinger Bands
                bb = ta.volatility.BollingerBands(df['price'])
                df['bollinger_high'] = bb.bollinger_hband()
                df['bollinger_low'] = bb.bollinger_lband()
                # Volatility
                df['volatility'] = df['price'].rolling(14).std() / df['price'].rolling(14).mean() * 100
                
            return df.dropna()
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching history: {e}")
        return pd.DataFrame()

# -----------------------------------------------------------------------------
# Analytics & ML Layer
# -----------------------------------------------------------------------------
def train_price_model(coin_id, price_data):
    """Trains a Gradient Boosting model to predict future prices."""
    try:
        if len(price_data) < 30: return None
        
        # Feature Engineering
        df = price_data.copy()
        for i in range(1, 4):
            df[f'lag_{i}'] = df['price'].shift(i)
        
        df = df.dropna()
        features = ['lag_1', 'lag_2', 'lag_3', 'rsi', 'volatility']
        # Filter only existing columns
        features = [f for f in features if f in df.columns]
        
        if not features: return None

        X = df[features]
        y = df['price']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
        model.fit(X_train_scaled, y_train)
        
        # Prediction for next available point (simulated)
        last_data = X.iloc[-1:].values
        last_scaled = scaler.transform(last_data)
        prediction = model.predict(last_scaled)[0]
        
        # Confidence score based on test set accuracy (R2)
        score = model.score(X_test_scaled, y_test)
        
        return {
            'prediction': prediction,
            'confidence': max(0, min(1, score)),  # Clip between 0 and 1
            'current_price': df['price'].iloc[-1]
        }
    except Exception as e:
        st.error(f"Modeling error: {e}")
        return None

def get_recommendations(portfolio_coins, all_coins, risk_level):
    """Generates recommendations using K-Means clustering based on risk profile."""
    if all_coins.empty: return []
    
    # Filter candidates
    candidates = all_coins[~all_coins['id'].isin(portfolio_coins)].head(50)
    
    # Simple feature extraction for clustering
    features = candidates[['current_price', 'market_cap', 'price_change_percentage_24h']].fillna(0)
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(features_scaled)
    candidates['cluster'] = clusters
    
    # Heuristic mapping of clusters to risk (simplified for demo)
    # We sort clusters by avg volatility/price change to assign risk levels
    cluster_risk = candidates.groupby('cluster')['price_change_percentage_24h'].std().sort_values()
    
    risk_map = {
        'low': cluster_risk.index[0],     # Lowest volatility cluster
        'medium': cluster_risk.index[1],
        'high': cluster_risk.index[2]     # Highest volatility cluster
    }
    
    target_cluster = risk_map.get(risk_level, 0)
    recs = candidates[candidates['cluster'] == target_cluster].head(5)
    return recs.to_dict('records')

# -----------------------------------------------------------------------------
# User Interface (Streamlit)
# -----------------------------------------------------------------------------
def main():
    st.title("🚀 Crypto Insights Dashboard")
    st.markdown("### AI-Powered Portfolio Management & Analytics")

    # Session State for Portfolio
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = {}

    all_coins = get_coin_list()

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Market Overview", "💼 My Portfolio", "🤖 AI Recommendations"])

    # --- TAB 1: MARKET OVERVIEW ---
    with tab1:
        st.subheader("Market Pulse")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if not all_coins.empty:
                display_cols = ['name', 'symbol', 'current_price', 'market_cap', 'price_change_percentage_24h']
                st.dataframe(
                    all_coins[display_cols].style.format({
                        'current_price': '${:.2f}',
                        'market_cap': '${:,.0f}',
                        'price_change_percentage_24h': '{:+.2f}%'
                    }),
                    use_container_width=True
                )
        
        with col2:
            st.subheader("Market Sentiment")
            avg_change = all_coins['price_change_percentage_24h'].mean() if not all_coins.empty else 0
            
            sentiment_color = "#28a745" if avg_change > 0 else "#dc3545"
            sentiment_text = "Bullish 🐂" if avg_change > 0 else "Bearish 🧸"
            
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: {sentiment_color};">
                <h3>{sentiment_text}</h3>
                <p style="font-size: 24px; font-weight: bold; color: {sentiment_color}">
                    {avg_change:+.2f}%
                </p>
                <p>Average 24h Change (Top 250)</p>
            </div>
            """, unsafe_allow_html=True)

    # --- TAB 2: PORTFOLIO ---
    with tab2:
        col_main, col_sidebar = st.columns([3, 1])
        
        with col_sidebar:
            st.markdown("### Add Asset")
            if not all_coins.empty:
                selected = st.selectbox("Select Coin", all_coins['id'].tolist(), format_func=lambda x: x.upper())
                amount = st.number_input("Amount", min_value=0.0, step=0.01)
                
                if st.button("Add to Portfolio"):
                    current_price = all_coins[all_coins['id'] == selected]['current_price'].iloc[0]
                    st.session_state.portfolio[selected] = {
                        'amount': amount,
                        'buy_price': current_price
                    }
                    st.success(f"Added {selected.upper()}")
                    time.sleep(1)
                    st.rerun()

        with col_main:
            st.subheader("Your Holdings")
            if st.session_state.portfolio:
                for coin_id, data in st.session_state.portfolio.items():
                    coin_meta = all_coins[all_coins['id'] == coin_id].iloc[0]
                    current_val = data['amount'] * coin_meta['current_price']
                    
                    # Fetch detailed history for sparkline/analysis
                    hist = get_coin_price_history(coin_id)
                    
                    with st.expander(f"{coin_meta['name']} ({coin_meta['symbol'].upper()}) - ${current_val:,.2f}", expanded=False):
                        c1, c2 = st.columns(2)
                        with c1:
                            st.metric("Current Price", f"${coin_meta['current_price']:,.2f}", f"{coin_meta['price_change_percentage_24h']:.2f}%")
                            st.metric("Holdings", f"{data['amount']} units")
                        
                        with c2:
                            # AI Prediction
                            if not hist.empty:
                                model_res = train_price_model(coin_id, hist)
                                if model_res:
                                    pred_delta = ((model_res['prediction'] - model_res['current_price']) / model_res['current_price']) * 100
                                    st.metric("AI Forecast (Next Day)", f"${model_res['prediction']:,.2f}", f"{pred_delta:+.2f}%", delta_color="normal")
                                    st.progress(model_res['confidence'], text=f"Model Confidence: {model_res['confidence']*100:.0f}%")
                        
                        # Charts
                        if not hist.empty:
                            fig, ax = plt.subplots(figsize=(10, 3))
                            ax.plot(hist['date'], hist['price'], label='Price')
                            if 'bollinger_high' in hist.columns:
                                ax.plot(hist['date'], hist['bollinger_high'], 'g--', alpha=0.3)
                                ax.plot(hist['date'], hist['bollinger_low'], 'r--', alpha=0.3)
                            ax.set_title("Price Trend & Bollinger Bands")
                            st.pyplot(fig)

            else:
                st.info("Your portfolio is empty. Add assets using the sidebar.")

    # --- TAB 3: RECOMMENDATIONS ---
    with tab3:
        st.subheader("Smart Recommendations")
        risk = st.select_slider("Select your Risk Profile", options=["low", "medium", "high"])
        
        if st.button("Generate Suggestions"):
            with st.spinner("Analyzing market clusters..."):
                recs = get_recommendations(list(st.session_state.portfolio.keys()), all_coins, risk)
                
                if recs:
                    for rec in recs:
                        st.markdown(f"""
                        <div class="recommendation-card">
                            <h4>{rec['name']} ({rec['symbol'].upper()})</h4>
                            <p>Price: <b>${rec['current_price']:,.2f}</b> | 24h Change: <b>{rec['price_change_percentage_24h']:.2f}%</b></p>
                            <p style="color: grey; font-size: 0.9em;">AI Cluster Analysis suggests this asset matches your <b>{risk}</b> risk profile based on recent volatility patterns.</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("No new recommendations available at this moment.")

if __name__ == "__main__":
    # Create models directory if needed
    os.makedirs('models', exist_ok=True)
    main()