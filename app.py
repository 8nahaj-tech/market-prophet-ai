import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from datetime import datetime, timedelta

# KONFIGURACJA
st.set_page_config(page_title="Market Prophet AI", page_icon="📈", layout="wide")

# STYLIZACJA
st.markdown("""
<style>
    .stApp { background-color: #000000; color: #e0e0e0; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    
    .kpi-card { background: #111; border: 1px solid #333; padding: 20px; border-radius: 10px; text-align: center; }
    .price-big { font-size: 38px; font-weight: 800; color: #fff; }
    .price-pos { color: #00ff00; font-size: 18px; font-weight: bold; }
    .price-neg { color: #ff0000; font-size: 18px; font-weight: bold; }
    
    .ai-box {
        background: linear-gradient(90deg, #1e1e2f 0%, #2a2a40 100%);
        border-left: 5px solid #bd00ff; padding: 15px; margin-top: 20px; border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# FUNKCJE
def get_stock_data(symbol, period="2y"):
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period=period)
        return df
    except: return None

def predict_trend(df, days_ahead):
    df = df.reset_index()
    df['Date_Num'] = df.index
    recent_df = df.tail(60) # Uczymy się na ostatnich 60 dniach
    
    X = recent_df['Date_Num'].values.reshape(-1, 1)
    y = recent_df['Close'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    last_idx = df['Date_Num'].max()
    future_idx = np.array(range(last_idx + 1, last_idx + days_ahead + 1)).reshape(-1, 1)
    future_prices = model.predict(future_idx)
    
    last_date = df['Date'].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, days_ahead + 1)]
    
    return future_dates, future_prices, model.coef_[0]

# INTERFEJS
st.sidebar.header("🎛️ Panel Kontrolny")
symbol = st.sidebar.text_input("Symbol (np. BTC-USD, TSLA, CDPROJEKT.WA)", value="BTC-USD").upper()
forecast_days = st.sidebar.slider("Prognoza (dni)", 7, 90, 30)

st.title("📈 Market Prophet AI")

if st.sidebar.button("🚀 URUCHOM ANALIZĘ"):
    with st.spinner('AI analizuje rynek...'):
        df = get_stock_data(symbol)
        
    if df is None or df.empty:
        st.error(f"❌ Nie znaleziono symbolu '{symbol}'.")
    else:
        # KPI
        curr = df['Close'].iloc[-1]
        prev = df['Close'].iloc[-2]
        chg = curr - prev
        pct = (chg / prev) * 100
        
        c1, c2, c3 = st.columns(3)
        with c1:
            cls = "price-pos" if chg >= 0 else "price-neg"
            sig = "+" if chg >= 0 else ""
            st.markdown(f"""<div class="kpi-card"><div style="color:#888; font-size:12px">AKTUALNA CENA</div><div class="price-big">{curr:,.2f}</div><div class="{cls}">{sig}{chg:.2f} ({sig}{pct:.2f}%)</div></div>""", unsafe_allow_html=True)
            
        # AI
        dates, prices, slope = predict_trend(df, forecast_days)
        exp_price = prices[-1]
        ai_chg = ((exp_price - curr) / curr) * 100
        
        with c2:
            col = "#00ff00" if ai_chg > 0 else "#ff0000"
            st.markdown(f"""<div class="kpi-card"><div style="color:#888; font-size:12px">PROGNOZA AI (za {forecast_days} dni)</div><div class="price-big" style="color:{col}">{exp_price:,.2f}</div><div style="color:{col}">Zmiana: {ai_chg:+.2f}%</div></div>""", unsafe_allow_html=True)

        with c3:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1]
            status, rsi_col = ("WYPRZEDANIE (Drogo)", "red") if rsi > 70 else ("OKAZJA (Tanio)", "green") if rsi < 30 else ("NEUTRALNIE", "white")
            st.markdown(f"""<div class="kpi-card"><div style="color:#888; font-size:12px">RSI (Sygnał)</div><div class="price-big" style="color:{rsi_col}">{rsi:.1f}</div><div style="color:{rsi_col}">{status}</div></div>""", unsafe_allow_html=True)

        # WYKRES
        st.markdown("---")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Historia', line=dict(color='#00d2ff', width=2)))
        fig.add_trace(go.Scatter(x=dates, y=prices, mode='lines', name='Prognoza AI', line=dict(color='#bd00ff', width=3, dash='dash')))
        fig.update_layout(template="plotly_dark", height=600, title=f"Analiza {symbol}")
        st.plotly_chart(fig, use_container_width=True)
        
        sent = "WZROSTOWY 🐂" if slope > 0 else "SPADKOWY 🐻"
        st.markdown(f"""<div class="ai-box"><h3>🧠 Komentarz AI:</h3>Trend jest <b>{sent}</b>. Cel cenowy: <b>{exp_price:,.2f}</b>.</div>""", unsafe_allow_html=True)
else:
    st.info("👈 Wpisz symbol (np. BTC-USD) i kliknij start.")
