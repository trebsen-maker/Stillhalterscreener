import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, date

st.set_page_config(page_title="Optionen Profi-Panel v64.2", layout="wide")
st.title("📊 Optionen Profi-Panel - v64.2 - Adaptiv MASTER")

# ==========================================
# --- SIDEBAR: EINGABEN & FILTER ---
# ==========================================
st.sidebar.header("📈 Basiswert & Filter")
ticker = st.sidebar.text_input("Ticker-Symbol (z.B. AAPL, MSFT, SPY, SAP.DE)", value="AAPL").upper()
use_v64_1 = st.sidebar.checkbox("Adaptive Bänder & Schockschutz aktivieren", value=True)
use_atr_pro = st.sidebar.checkbox("3-fach ATR-Schutzband zuschalten", value=False)

st.sidebar.header("⚙️ Manuelle Eingabe (Option)")
m_type = st.sidebar.selectbox("Optionstyp", ["PUT", "CALL"])
m_strike = st.sidebar.number_input("Strike-Preis", min_value=0.0, value=0.0, step=1.0)
m_price = st.sidebar.number_input("Erhaltene Prämie", min_value=0.0, value=0.0, step=0.1)
m_cost = st.sidebar.number_input("Kaufpreis Aktie (Bestand)", min_value=0.0, value=0.0, step=1.0)
m_cont = st.sidebar.number_input("Anzahl Kontrakte", min_value=0, value=0, step=1)
m_shares = st.sidebar.number_input("Bestand Aktien", min_value=0, value=0, step=1)
exp_date = st.sidebar.date_input("Verfallsdatum", value=date(2026, 12, 18))

# ==========================================
# --- DATENABRUF (KOSTENLOS ÜBER YFINANCE) ---
# ==========================================
@st.cache_data(ttl=60) # Aktualisiert Kurse alle 60 Sekunden
def load_data(symbol):
    df_d = yf.download(symbol, period="2y", interval="1d", progress=False)
    df_w = yf.download(symbol, period="2y", interval="1wk", progress=False)
    if isinstance(df_d.columns, pd.MultiIndex):
        df_d.columns = df_d.columns.get_level_values(0)
    if isinstance(df_w.columns, pd.MultiIndex):
        df_w.columns = df_w.columns.get_level_values(0)
    return df_d, df_w

df, df_w = load_data(ticker)

if df.empty or len(df) < 252:
    st.error("Nicht genügend Kursdaten gefunden. Bitte prüfe das Ticker-Symbol (z.B. AAPL oder BMW.DE).")
    st.stop()

# ==========================================
# --- INDIKATOR-BERECHNUNGEN ---
# ==========================================
# RSI (14)
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

# ATR (20 & 5)
high_low = df['High'] - df['Low']
high_close = np.abs(df['High'] - df['Close'].shift())
low_close = np.abs(df['Low'] - df['Close'].shift())
tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

df['ATR_20'] = tr.ewm(alpha=1/20, adjust=False).mean()
df['ATR_5'] = tr.ewm(alpha=1/5, adjust=False).mean()
df['Gap_Risk'] = np.where(df['ATR_20'] > 0, df['ATR_5'] / df['ATR_20'], 1.0)

# Keltner Bänder (Adaptiv)
df['KC_Basis'] = df['Close'].ewm(span=20, adjust=False).mean()
df['KC_Mult'] = np.where(use_v64_1 & (df['Gap_Risk'] > 1.4), 2.4 * 1.3, 2.4)
df['KC_Upper'] = df['KC_Basis'] + (df['KC_Mult'] * df['ATR_20'])
df['KC_Lower'] = df['KC_Basis'] - (df['KC_Mult'] * df['ATR_20'])

# 3-fach ATR Schutzband
df['ATR_Pro_Lower'] = df['KC_Basis'] - (3.2 * df['ATR_5'])
df['ATR_Pro_Upper'] = df['KC_Basis'] + (3.2 * df['ATR_5'])

# 30-Wochen SMA
df_w['SMA30W'] = df_w['Close'].rolling(window=30).mean()
df['SMA30W'] = df_w['SMA30W'].reindex(df.index, method='ffill')

# Garman-Klass Volatilität & HV Rank (252 Tage)
gk_instant = 0.5 * (np.log(df['High'] / df['Low']) ** 2) - (2 * np.log(2) - 1) * (np.log(df['Close'] / df['Open']) ** 2)
df['GK_Vol'] = np.sqrt(gk_instant.rolling(window=20).mean() * 252) * 100
df['HV_Rank'] = df['GK_Vol'].rolling(window=252).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False)

# Aktuelle Werte (letzte Kerze)
last = df.iloc[-1]
close_now = float(last['Close'])
sma30w_now = float(last['SMA30W'])
gap_risk_now = float(last['Gap_Risk'])
hv_rank_now = float(last['HV_Rank'])
rsi_now = float(last['RSI'])

# Restlaufzeit & Moneyness
dte = max(0, (exp_date - date.today()).days)
is_itm = (close_now > m_strike) if m_type == "CALL" else (close_now < m_strike)

# Entry & Exit Score
entry_score = 0
if (hv_rank_now > 30) and not (use_v64_1 and gap_risk_now > 1.4):
    entry_score += 50
if rsi_now < 45 or rsi_now > 55:
    entry_score += 25
if float(last['Low']) <= float(last['KC_Lower']) or float(last['High']) >= float(last['KC_Upper']):
    entry_score += 25

exit_score = (40 if float(last['High']) >= float(last['KC_Upper']) else 0) + \
             (30 if rsi_now > 70 else 0) + \
             (30 if close_now > sma30w_now * 1.15 else 0)

# ==========================================
# --- LAYOUT: CHART & INFOTAFEL ---
# ==========================================
col_chart, col_table = st.columns([2.2, 1])

with col_chart:
    plot_df = df.tail(150) # Zeigt die letzten 150 Handelstage
    fig = go.Figure()

    # Kerzenchart
    fig.add_trace(go.Candlestick(
        x=plot_df.index, open=plot_df['Open'], high=plot_df['High'],
        low=plot_df['Low'], close=plot_df['Close'], name="Kurs"
    ))

    # Keltner Bänder
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['KC_Upper'], line=dict(color='rgba(255,0,0,0.6)', width=1.5), name="KC Upper"))
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['KC_Lower'], line=dict(color='rgba(0,255,0,0.6)', width=1.5), fill='tonexty', fillcolor='rgba(128,128,128,0.1)', name="KC Lower"))

    # 30W SMA
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['SMA30W'], line=dict(color='yellow', width=2.5), name="30W SMA"))

    # Optional: 3-fach ATR
    if use_atr_pro:
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['ATR_Pro_Lower'], mode='markers', marker=dict(color='orange', size=4), name="ATR Pro Support"))
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['ATR_Pro_Upper'], mode='markers', marker=dict(color='orange', size=4), name="ATR Pro Resistance"))

    # Optional: Strike-Linie
    if m_strike > 0:
        fig.add_hline(y=m_strike, line_dash="dash", line_color="purple" if m_type == "CALL" else "cyan", annotation_text=f"Strike {m_strike}")

    fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

with col_table:
    st.subheader("📋 Cockpit-Dashboard")
    pct_trend = ((close_now - sma30w_now) / sma30w_now) * 100
    
    dashboard_data = {
        "METRIK": [
            "Basiswert (Aktuell)", "Optionstyp", "Abw. 30W-Trend", "Strike-Preis",
            "Restlaufzeit", "Moneyness", "Gap-Risk Ratio", "Entry Score",
            "Exit Score", "HV Rank (GK)"
        ],
        "WERT": [
            f"{ticker} ({close_now:.2f})",
            m_type,
            f"{pct_trend:+.2f} %",
            f"{m_strike:.2f}" if m_strike > 0 else "Nicht definiert",
            f"{dte} Tage",
            "In-the-Money (ITM)" if is_itm else "Out-of-the-Money (OTM)",
            f"{gap_risk_now:.2f}",
            f"{entry_score} %",
            f"{exit_score} %",
            f"{hv_rank_now:.1f} %"
        ]
    }
    st.dataframe(pd.DataFrame(dashboard_data), hide_index=True, use_container_width=True, height=400)
    if st.button("🔄 Kurse jetzt aktualisieren"):
        st.cache_data.clear()
        st.rerun()
