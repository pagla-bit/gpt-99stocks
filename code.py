# streamlit_99_stocks_dynamic_manual_input.py
"""
99 Stocks — Dynamic Indices (Manual Ticker Input + Full Dynamic Top-33 Selection)
- User inputs tickers manually for Russell2000 (small ~2000), S&P400 (mid ~400), S&P500 (large ~500)
- Evaluates ALL input stocks per bucket using algo, selects top 33 by score
- Three algorithm modes: probabilistic, technical, hybrid
- Vectorized Monte Carlo, parallel evaluation, caching
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(layout="wide", page_title="99 Stocks — Dynamic Manual Input (Top 33 Selection)")

# -------------------- Data fetcher (yfinance) --------------------

@st.cache_data(ttl=1800)
def get_data(ticker, period="1y", interval="1d"):
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period=period, interval=interval, actions=False, auto_adjust=True)
        if hist.empty:
            return pd.DataFrame(), {"_error": "empty history"}
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not set(required_cols).issubset(hist.columns):
            return pd.DataFrame(), {"_error": "incomplete OHLCV"}
        info = tk.info or {}
        return hist, info
    except Exception as e:
        return pd.DataFrame(), {"_error": str(e)}

# -------------------- Indicators & scoring --------------------

def calc_indicators(df, rsi_period=14, macd_fast=12, macd_slow=26, macd_signal=9, sma_long=50, atr_period=14, adx_period=14):
    df = df.copy()
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(com=rsi_period-1, adjust=False).mean()
    ma_down = down.ewm(com=rsi_period-1, adjust=False).mean()
    rs = ma_up / ma_down
    df['RSI'] = 100 - (100 / (1 + rs))
    ema_fast = df['Close'].ewm(span=macd_fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=macd_slow, adjust=False).mean()
    df['MACD'] = ema_fast - ema_slow
    df['MACD_signal'] = df['MACD'].ewm(span=macd_signal, adjust=False).mean()
    df['SMA_long'] = df['Close'].rolling(sma_long).mean()
    hl = df['High'] - df['Low']
    hc = abs(df['High'] - df['Close'].shift())
    lc = abs(df['Low'] - df['Close'].shift())
    tr = np.maximum(hl, np.maximum(hc, lc))
    df['ATR'] = tr.ewm(span=atr_period, adjust=False).mean()
    plus_dm = df['High'].diff().clip(lower=0)
    minus_dm = (-df['Low'].diff()).clip(upper=0)
    atr = df['ATR'].replace(0, np.nan)
    plus_di = 100 * (plus_dm.ewm(span=adx_period).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=adx_period).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)).replace([np.inf, -np.inf], np.nan) * 100
    df['ADX'] = dx.ewm(span=adx_period).mean()
    return df

def prob_reach_target(df, current_price, target_return, sims=2500, max_days=90):
    returns = df['Close'].pct_change().dropna().values
    if len(returns) < 30:
        return 0.0
    mu, sigma = np.nanmean(returns), np.nanstd(returns)
    if sigma == 0 or np.isnan(sigma):
        return 0.0
    rand = np.random.normal(mu, sigma, (sims, max_days))
    paths = current_price * np.cumprod(1 + rand, axis=1)
    threshold = current_price * (1 + target_return)
    return float(np.mean((paths >= threshold).any(axis=1)))

def technical_score(df):
    if len(df) < 60:
        return 0.0
    latest, prev = df.iloc[-1], df.iloc[-2]
    score = 0.0
    if not np.isnan(latest['RSI']):
        if latest['RSI'] < 30: score += 0.35
        elif latest['RSI'] < 40: score += 0.15
    if not np.isnan(prev.get('MACD', np.nan)) and not np.isnan(prev.get('MACD_signal', np.nan)):
        if prev['MACD'] < prev['MACD_signal'] and latest['MACD'] > latest['MACD_signal']:
            score += 0.25
    if not np.isnan(latest.get('SMA_long', np.nan)) and latest['Close'] > latest['SMA_long']:
        score += 0.15
    if not np.isnan(latest.get('ADX', np.nan)) and latest['ADX'] > 25:
        score += 0.10
    if len(df) >= 6:
        mom = (df['Close'].iloc[-1] / df['Close'].iloc[-6]) - 1
        if mom > 0.03: score += 0.15
        elif mom > 0: score += 0.05
    return min(score, 1.0)

def hybrid_score(df, current_price, target_return, sims, max_days, alpha_prob=0.6):
    p = prob_reach_target(df, current_price, target_return, sims, max_days)
    t = technical_score(df)
    return alpha_prob * p + (1 - alpha_prob) * t

# -------------------- Ticker evaluation --------------------

def evaluate_ticker(ticker, algo, target_return, sims, max_days, period="1y", interval="1d"):
    hist, info = get_data(ticker, period, interval)
    if hist.empty:
        return {"ticker": ticker, "error": info.get("_error", "no data")}
    df = calc_indicators(hist)
    latest = df.iloc[-1]
    current_price = float(latest['Close'])
    market_cap = info.get('marketCap', None)
    prob = prob_reach_target(df, current_price, target_return, sims, max_days) if algo in ('probabilistic', 'hybrid') else np.nan
    tech = technical_score(df) if algo in ('technical', 'hybrid') else np.nan
    score = prob if algo == 'probabilistic' else tech if algo == 'technical' else hybrid_score(df, current_price, target_return, sims, max_days)
    return {"ticker": ticker, "marketCap": market_cap, "price": current_price, "prob": prob, "tech": tech, "score": score}

# -------------------- Streamlit UI --------------------

st.title("99 Stocks — Dynamic Manual Ticker Input — Top 33 Selection")

st.sidebar.header("Bucket Controls")

st.sidebar.markdown("### Small Cap (e.g., Russell 2000)")
sc_algo = st.sidebar.selectbox("Algorithm", ['probabilistic', 'technical', 'hybrid'], index=0, key='sc_algo')
sc_target = st.sidebar.number_input("Profit target %", min_value=1.0, max_value=100.0, value=5.0, step=0.5, key='sc_target')
sc_days = st.sidebar.slider("Days horizon", min_value=1, max_value=30, value=3, step=1, key='sc_days')

st.sidebar.markdown("### Mid Cap (e.g., S&P 400)")
mc_algo = st.sidebar.selectbox("Algorithm", ['probabilistic', 'technical', 'hybrid'], index=0, key='mc_algo')
mc_target = st.sidebar.number_input("Profit target %", min_value=1.0, max_value=200.0, value=10.0, step=0.5, key='mc_target')
mc_days = st.sidebar.slider("Days horizon", min_value=3, max_value=90, value=14, step=1, key='mc_days')

st.sidebar.markdown("### Large Cap (e.g., S&P 500)")
lc_algo = st.sidebar.selectbox("Algorithm", ['probabilistic', 'technical', 'hybrid'], index=0, key='lc_algo')
lc_target = st.sidebar.number_input("Profit target %", min_value=1.0, max_value=500.0, value=30.0, step=1.0, key='lc_target')
lc_days = st.sidebar.slider("Days horizon", min_value=14, max_value=365, value=90, step=1, key='lc_days')

st.sidebar.markdown("---")
st.sidebar.header("Performance")
sims = st.sidebar.selectbox("Monte Carlo sims", options=[500, 1000, 2500, 5000], index=2)
workers = st.sidebar.slider("Parallel workers", 2, 24, 8)
st.sidebar.markdown("**Warning:** Eval time depends on input sizes (up to ~2900 stocks: 5-15 min).")
st.sidebar.markdown("---")
run_request = st.sidebar.button("Run Full Evaluation")

# -------------------- Manual Ticker Inputs --------------------

st.header("Enter Tickers Manually (Comma-Separated)")

input_col1, input_col2, input_col3 = st.columns(3)

with input_col1:
    st.markdown("### Small Cap Tickers")
    small_input = st.text_area(
        "Paste tickers (e.g., ABR,ACAD,ACHC,...)", 
        placeholder="ABR, ACAD, ACHC, ADMA, ... (up to ~2000)",
        height=200,
        key='small_input'
    )

with input_col2:
    st.markdown("### Mid Cap Tickers")
    mid_input = st.text_area(
        "Paste tickers (e.g., A,ADM,ALB,...)", 
        placeholder="A, ADM, ALB, ALLE, ... (up to ~400)",
        height=200,
        key='mid_input'
    )

with input_col3:
    st.markdown("### Large Cap Tickers")
    large_input = st.text_area(
        "Paste tickers (e.g., AAPL,MSFT,NVDA,...)", 
        placeholder="AAPL, MSFT, NVDA, GOOGL, ... (up to ~500)",
        height=200,
        key='large_input'
    )

# Parse inputs
def parse_tickers(input_text):
    if not input_text:
        return []
    return sorted(set([t.strip().upper() for t in input_text.split(',') if t.strip()]))

small_universe = parse_tickers(small_input)
mid_universe = parse_tickers(mid_input)
large_universe = parse_tickers(large_input)

st.markdown("### Input Universe Sizes")
col1, col2, col3 = st.columns(3)
col1.metric("Small Cap", len(small_universe))
col2.metric("Mid Cap", len(mid_universe))
col3.metric("Large Cap", len(large_universe))

if not all([small_universe, mid_universe, large_universe]):
    st.warning("Enter tickers for all buckets to run evaluation.")
    st.stop()

# -------------------- Parallel evaluation --------------------

def evaluate_bucket_parallel(tickers, algo, target_pct, days, sims, workers, max_results=33):
    target_return = target_pct / 100.0
    results, errors = [], []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(evaluate_ticker, t, algo, target_return, sims, days): t for t in tickers}
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                r = fut.result()
                if 'error' in r:
                    errors.append((t, r['error']))
                else:
                    results.append(r)
            except Exception as e:
                errors.append((t, str(e)))
    if not results:
        return pd.DataFrame(), errors
    df = pd.DataFrame(results).sort_values('score', ascending=False).head(max_results).reset_index(drop=True)
    return df, errors

if run_request:
    st.info("Running full dynamic evaluation... Progress by bucket.")
    with st.spinner("Evaluating Small Cap..."):
        sc_df, sc_errs = evaluate_bucket_parallel(small_universe, sc_algo, sc_target, sc_days, sims, workers)
    with st.spinner("Evaluating Mid Cap..."):
        mc_df, mc_errs = evaluate_bucket_parallel(mid_universe, mc_algo, mc_target, mc_days, sims, workers)
    with st.spinner("Evaluating Large Cap..."):
        lc_df, lc_errs = evaluate_bucket_parallel(large_universe, lc_algo, lc_target, lc_days, sims, workers)

    st.header("Top 33 Candidates (Dynamic Selection by Score)")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader(f"Small Cap — {sc_target}% in {sc_days}d ({sc_algo})")
        if not sc_df.empty:
            display = sc_df[['ticker', 'price', 'marketCap', 'prob', 'tech', 'score']].round(4)
            st.dataframe(display)
        else:
            st.warning("No valid results for Small Cap.")
        if sc_errs:
            st.write(f"Errors ({len(sc_errs)}): {', '.join([f'{e[0]}:{e[1]}' for e in sc_errs[:5]])}...")

    with c2:
        st.subheader(f"Mid Cap — {mc_target}% in {mc_days}d ({mc_algo})")
        if not mc_df.empty:
            display = mc_df[['ticker', 'price', 'marketCap', 'prob', 'tech', 'score']].round(4)
            st.dataframe(display)
        else:
            st.warning("No valid results for Mid Cap.")
        if mc_errs:
            st.write(f"Errors ({len(mc_errs)}): {', '.join([f'{e[0]}:{e[1]}' for e in mc_errs[:5]])}...")

    with c3:
        st.subheader(f"Large Cap — {lc_target}% in {lc_days}d ({lc_algo})")
        if not lc_df.empty:
            display = lc_df[['ticker', 'price', 'marketCap', 'prob', 'tech', 'score']].round(4)
            st.dataframe(display)
        else:
            st.warning("No valid results for Large Cap.")
        if lc_errs:
            st.write(f"Errors ({len(lc_errs)}): {', '.join([f'{e[0]}:{e[1]}' for e in lc_errs[:5]])}...")

    st.markdown("---")
    st.write("**Notes:** Scores blend prob/tech (hybrid) or use single mode. Backtest results; not advice.")
else:
    st.info("Enter comma-separated tickers in the boxes above, adjust controls, & click 'Run Full Evaluation'.")

st.markdown("---")
st.write("Enhancements? Auto-load from CSV, sector filters, or export results?")
