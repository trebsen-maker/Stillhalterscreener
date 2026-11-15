import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date

st.set_page_config(page_title="Optionen Profi-Panel v64.2", layout="wide")
st.title("📊 Optionen Profi-Panel - v64.2 - Adaptiv MASTER")

def calculate_indicators(df, df_w, use_v64_1=True):
    df = df.copy()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))

    tr = pd.concat([
        df['High'] - df['Low'],
        np.abs(df['High'] - df['Close'].shift()),
        np.abs(df['Low'] - df['Close'].shift())
    ], axis=1).max(axis=1)

    df['ATR_20'] = tr.ewm(alpha=1/20, adjust=False).mean()
    df['ATR_14'] = tr.ewm(alpha=1/14, adjust=False).mean()
    df['ATR_5'] = tr.ewm(alpha=1/5, adjust=False).mean()
    df['Gap_Risk'] = np.where(df['ATR_20'] > 0, df['ATR_5'] / df['ATR_20'], 1.0)

    df['KC_Basis'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['KC_Mult'] = np.where(use_v64_1 & (df['Gap_Risk'] > 1.4), 2.4 * 1.3, 2.4)
    df['KC_Upper'] = df['KC_Basis'] + (df['KC_Mult'] * df['ATR_20'])
    df['KC_Lower'] = df['KC_Basis'] - (df['KC_Mult'] * df['ATR_20'])

    df['ATR_Pro_Lower'] = df['KC_Basis'] - (3.2 * df['ATR_5'])
    df['ATR_Pro_Upper'] = df['KC_Basis'] + (3.2 * df['ATR_5'])

    hl2 = (df['High'] + df['Low']) / 2
    basic_ub = hl2 + (3.0 * df['ATR_14'])
    basic_lb = hl2 - (3.0 * df['ATR_14'])
    
    final_ub = np.zeros(len(df))
    final_lb = np.zeros(len(df))
    supertrend = np.zeros(len(df))
    st_dir = np.ones(len(df))
    
    c_arr = df['Close'].values
    ub_arr = basic_ub.values
    lb_arr = basic_lb.values

    for i in range(1, len(df)):
        final_ub[i] = ub_arr[i] if (ub_arr[i] < final_ub[i-1] or c_arr[i-1] > final_ub[i-1]) else final_ub[i-1]
        final_lb[i] = lb_arr[i] if (lb_arr[i] > final_lb[i-1] or c_arr[i-1] < final_lb[i-1]) else final_lb[i-1]
        if st_dir[i-1] == 1:
            st_dir[i] = -1 if c_arr[i] > final_ub[i] else 1
        else:
            st_dir[i] = 1 if c_arr[i] < final_lb[i] else -1
        supertrend[i] = final_lb[i] if st_dir[i] == -1 else final_ub[i]

    df['Supertrend'] = supertrend
    df['ST_Dir'] = st_dir

    df_w = df_w.copy()
    df_w['SMA30W'] = df_w['Close'].rolling(window=30).mean()
    df['SMA30W'] = df_w['SMA30W'].reindex(df.index, method='ffill')

    gk_inst = 0.5 * (np.log(df['High'] / df['Low']) ** 2) - (2 * np.log(2) - 1) * (np.log(df['Close'] / df['Open']) ** 2)
    df['GK_Vol'] = np.sqrt(gk_inst.rolling(window=20).mean() * 252) * 100
    df['HV_Rank'] = df['GK_Vol'].rolling(window=252).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False)
    return df

def get_scores(last_row, use_v64_1=True):
    c_now = float(last_row['Close'])
    s30_now = float(last_row['SMA30W'])
    gr_now = float(last_row['Gap_Risk'])
    hv_now = float(last_row['HV_Rank'])
    rsi_now = float(last_row['RSI'])

    e_score = 0
    if (hv_now > 30) and not (use_v64_1 and gr_now > 1.4):
        e_score += 50
    if rsi_now < 45 or rsi_now > 55:
        e_score += 25
    if float(last_row['Low']) <= float(last_row['KC_Lower']) or float(last_row['High']) >= float(last_row['KC_Upper']):
        e_score += 25

    x_score = 0
    if float(last_row['High']) >= float(last_row['KC_Upper']):
        x_score += 40
    if rsi_now > 70:
        x_score += 30
    if c_now > s30_now * 1.15:
        x_score += 30
    return e_score, x_score

def scan_single_ticker(sym, data_d, data_w, is_single, use_v64_1):
    try:
        df_sym_d = data_d.copy() if is_single else data_d[sym].dropna(how="all")
        df_sym_w = data_w.copy() if is_single else data_w[sym].dropna(how="all")
        if len(df_sym_d) < 252:
            return None
        df_calc = calculate_indicators(df_sym_d, df_sym_w, use_v64_1)
        last_r = df_calc.iloc[-1]
        e_score, x_score = get_scores(last_r, use_v64_1)
        c_price = float(last_r['Close'])
        s30 = float(last_r['SMA30W'])
        pct_30 = ((c_price - s30) / s30) * 100
        st_txt = "🟢 Bullish" if last_r['ST_Dir'] == -1 else "🔴 Bearish"
        return {
            "Ticker": sym,
            "Kurs": round(c_price, 2),
            "Entry Score (%)": int(e_score),
            "Exit Score (%)": int(x_score),
            "HV Rank (%)": round(float(last_r['HV_Rank']), 1),
            "RSI (14)": round(float(last_r['RSI']), 1),
            "Gap-Risk": round(float(last_r['Gap_Risk']), 2),
            "Abw. 30W-SMA (%)": round(pct_30, 2),
            "Supertrend": st_txt
        }
    except Exception:
        return None

def run_scanner(watchlist_str, use_v64_1):
    raw_tickers = [t.strip().upper() for t in watchlist_str.split(",") if t.strip()]
    tickers_to_scan = list(dict.fromkeys(raw_tickers))[:40]
    st.info(f"Scanne {len(tickers_to_scan)} Aktien... Bitte einen Moment Geduld.")
    
    data_d = yf.download(tickers_to_scan, period="2y", interval="1d", group_by="ticker", progress=False)
    data_w = yf.download(tickers_to_scan, period="2y", interval="1wk", group_by="ticker", progress=False)

    results = []
    progress_bar = st.progress(0)
    is_single = len(tickers_to_scan) == 1

    for idx, sym in enumerate(tickers_to_scan):
        res_item = scan_single_ticker(sym, data_d, data_w, is_single, use_v64_1)
        if res_item is not None:
            results.append(res_item)
        progress_bar.progress((idx + 1) / len(tickers_to_scan))

    if results:
        res_df = pd.DataFrame(results).sort_values(by=["Entry Score (%)", "HV Rank (%)"], ascending=[False, False])
        st.success(f"Fertig! {len(res_df)} Werte erfolgreich ausgewertet.")
        st.dataframe(res_df, hide_index=True, use_container_width=True, height=600)
    else:
        st.error("Es konnten keine Daten abgerufen werden. Bitte prüfe die Schreibweise der Ticker.")

st.sidebar.header("📈 Einzel-Chart & Filter")
ticker = st.sidebar.text_input("Ticker-Symbol (z.B. AAPL, AAL, SAP.DE)", value="AAL").upper().strip()
use_v64_1 = st.sidebar.checkbox("Adaptive Bänder & Schockschutz aktivieren", value=True)
use_atr_pro = st.sidebar.checkbox("3-fach ATR-Schutzband zuschalten", value=False)
show_supertrend = st.sidebar.checkbox("Supertrend (3.0, 14) anzeigen", value=True)

st.sidebar.header("⚙️ Manuelle Eingabe (Option)")
m_type = st.sidebar.selectbox("Optionstyp", ["PUT", "CALL"])
m_strike = st.sidebar.number_input("Strike-Preis", min_value=0.0, value=0.0, step=1.0)
m_price = st.sidebar.number_input("Erhaltene Prämie", min_value=0.0, value=0.0, step=0.1)
m_cost = st.sidebar.number_input("Kaufpreis Aktie (Bestand)", min_value=0.0, value=0.0, step=1.0)
m_cont = st.sidebar.number_input("Anzahl Kontrakte", min_value=0, value=0, step=1)
m_shares = st.sidebar.number_input("Bestand Aktien", min_value=0, value=0, step=1)
exp_date = st.sidebar.date_input("Verfallsdatum", value=date(2026, 12, 18))

@st.cache_data(ttl=60)
def load_single_data(symbol):
    df_d = yf.download(symbol, period="2y", interval="1d", progress=False)
    df_w = yf.download(symbol, period="2y", interval="1wk", progress=False)
    if isinstance(df_d.columns, pd.MultiIndex):
        df_d.columns = df_d.columns.get_level_values(0)
    if isinstance(df_w.columns, pd.MultiIndex):
        df_w.columns = df_w.columns.get_level_values(0)
    return df_d, df_w

df_raw, df_w_raw = load_single_data(ticker)

if df_raw.empty or len(df_raw) < 252:
    st.error("Nicht genügend Kursdaten gefunden. Bitte prüfe das Ticker-Symbol (z.B. AAPL oder BMW.DE).")
    st.stop()

df = calculate_indicators(df_raw, df_w_raw, use_v64_1)
last = df.iloc[-1]
close_now = float(last['Close'])
sma30w_now = float(last['SMA30W'])
gap_risk_now = float(last['Gap_Risk'])
hv_rank_now = float(last['HV_Rank'])
entry_score, exit_score = get_scores(last, use_v64_1)

dte = max(0, (exp_date - date.today()).days)
is_itm = (close_now > m_strike) if m_type == "CALL" else (close_now < m_strike)

col_chart, col_table = st.columns([2.2, 1])

with col_chart:
    plot_df = df.tail(150).copy()
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['Open'], high=plot_df['High'], low=plot_df['Low'], close=plot_df['Close'], name="Kurs"))
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['KC_Upper'], line=dict(color='rgba(255,0,0,0.6)', width=1.5), name="KC Upper"))
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['KC_Lower'], line=dict(color='rgba(0,255,0,0.6)', width=1.5), fill='tonexty', fillcolor='rgba(128,128,128,0.1)', name="KC Lower"))
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['SMA30W'], line=dict(color='yellow', width=2.5), name="30W SMA"))

    if show_supertrend:
        st_green = np.where(plot_df['ST_Dir'] == -1, plot_df['Supertrend'], np.nan)
        st_red = np.where(plot_df['ST_Dir'] == 1, plot_df['Supertrend'], np.nan)
        fig.add_trace(go.Scatter(x=plot_df.index, y=st_green, mode='lines', line=dict(color='#00e676', width=2.5), name="Supertrend (Bullish)"))
        fig.add_trace(go.Scatter(x=plot_df.index, y=st_red, mode='lines', line=dict(color='#ff5252', width=2.5), name="Supertrend (Bearish)"))

    if use_atr_pro:
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['ATR_Pro_Lower'], mode='markers', marker=dict(color='orange', size=4), name="ATR Pro Support"))
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['ATR_Pro_Upper'], mode='markers', marker=dict(color='orange', size=4), name="ATR Pro Resistance"))

    if m_strike > 0:
        fig.add_hline(y=m_strike, line_dash="dash", line_color="purple" if m_type == "CALL" else "cyan", annotation_text=f"Strike {m_strike}")

    fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

with col_table:
