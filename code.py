# streamlit_99_stocks.py
"""
99 Stocks Dashboard
- Three market-cap buckets (<=1B, 1B-10B, >10B)
- Pulls an initial universe from Wikipedia S&P500 (fallback list included)
- Live market caps from yfinance for classification
- For each bucket: choose algorithm (probabilistic, technical, hybrid),
  set profit target and time window (days)
- Up to 33 tickers per bucket (best-ranked by chosen algorithm)
- Parallel fetching, caching, vectorized Monte Carlo
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import requests
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(layout="wide", page_title="99 Stocks Dashboard")

# -------------------- Utilities & Caching --------------------

@st.cache_data(ttl=3600)
def fetch_sp500_list():
    """Try to fetch S&P500 tickers from Wikipedia. Return list of tickers."""
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        df = tables[0]
        return [t.replace('.', '-') for t in df['Symbol'].astype(str).tolist()]
    except Exception:
        # Fallback conservative list (subset) if wiki fetch fails
        return [
            "AAPL","MSFT","AMZN","GOOGL","META","NVDA","TSLA","JPM","V","JNJ","WMT","PG","DIS",
            "MA","HD","BAC","XOM","CVX","PFE","KO","PEP","CSCO","ADBE","CMCSA","NFLX","INTC",
            "T","VZ","CRM","ABBV","MRK","ABT","NKE","ORCL","TXN","QCOM","UNH","LIN","CVS","MCD",
            "LOW","MDT","PM","SCHW","RTX","NEE","COST","AMGN","HON","UPS","BMY","SBUX","INTU"
        ]

@st.cache_data(ttl=1800)
def get_data(ticker, period="1y", interval="1d"):
    """Fetch OHLCV and info for a ticker via yfinance (cached)."""
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period=period, interval=interval, actions=False, auto_adjust=True)
        # Validate columns
        if hist.empty:
            return pd.DataFrame(), {}
        required_cols = ['Open','High','Low','Close','Volume']
        if not set(required_cols).issubset(set(hist.columns)):
            return pd.DataFrame(), {"_error": "Incomplete OHLCV"}
        info = tk.info if hasattr(tk, "info") else {}
        return hist, info
    except Exception as e:
        return pd.DataFrame(), {"_error": str(e)}

# -------------------- Indicators & Signals --------------------

def calc_indicators(df, rsi_period=14, macd_fast=12, macd_slow=26, macd_signal=9, sma_long=50, atr_period=14, adx_period=14):
    """Add RSI, MACD, SMA_long, ATR, ADX to df."""
    df = df.copy()
    # RSI
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(com=rsi_period-1, adjust=False).mean()
    ma_down = down.ewm(com=rsi_period-1, adjust=False).mean()
    rs = ma_up / ma_down
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    ema_fast = df['Close'].ewm(span=macd_fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=macd_slow, adjust=False).mean()
    df['MACD'] = ema_fast - ema_slow
    df['MACD_signal'] = df['MACD'].ewm(span=macd_signal, adjust=False).mean()

    # SMA long
    df['SMA_long'] = df['Close'].rolling(sma_long).mean()

    # ATR
    hl = df['High'] - df['Low']
    hc = (df['High'] - df['Close'].shift()).abs()
    lc = (df['Low'] - df['Close'].shift()).abs()
    tr = np.maximum(hl, np.maximum(hc, lc))
    df['ATR'] = tr.ewm(span=atr_period, adjust=False).mean()

    # ADX-like
    plus_dm = df['High'].diff()
    minus_dm = -df['Low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    plus_di = 100 * (plus_dm.ewm(span=adx_period).mean() / df['ATR'])
    minus_di = 100 * (minus_dm.ewm(span=adx_period).mean() / df['ATR'])
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    df['ADX'] = dx.ewm(span=adx_period).mean()

    return df

# -------------------- Probabilistic Monte Carlo (vectorized) --------------------

def prob_reach_target(df, current_price, target_return, sims=2500, max_days=90):
    """
    Vectorized MC: simulate daily returns using historical mu/sigma (normal approx).
    Returns probability to reach target within max_days.
    """
    returns = df['Close'].pct_change().dropna().values
    if len(returns) < 30:
        return 0.0  # insufficient history
    mu = returns.mean()
    sigma = returns.std()
    if sigma == 0:
        return 0.0
    rand = np.random.normal(loc=mu, scale=sigma, size=(sims, max_days))
    price_paths = current_price * np.cumprod(1 + rand, axis=1)
    threshold = current_price * (1 + target_return)
    hits = (price_paths >= threshold).any(axis=1)
    return float(np.mean(hits))

# -------------------- Technical scoring --------------------

def technical_score(df):
    """
    Heuristic technical score 0..1
    - RSI near oversold -> positive
    - Recent MACD bullish crossover -> positive
    - Price above SMA_long -> positive
    - ADX > 25 increases score (trend)
    - Momentum (last 5-day return) adds score
    """
    if len(df) < 60:
        return 0.0
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    score = 0.0
    # RSI: lower RSI -> more likely bounce short-term
    if not np.isnan(latest['RSI']):
        if latest['RSI'] < 30:
            score += 0.35
        elif latest['RSI'] < 40:
            score += 0.15
    # MACD crossover
    if not np.isnan(prev['MACD']) and not np.isnan(prev['MACD_signal']):
        if (prev['MACD'] < prev['MACD_signal']) and (latest['MACD'] > latest['MACD_signal']):
            score += 0.25
    # Price vs SMA_long
    if not np.isnan(latest['SMA_long']):
        if latest['Close'] > latest['SMA_long']:
            score += 0.15
    # ADX
    if not np.isnan(latest['ADX']) and latest['ADX'] > 25:
        score += 0.10
    # Momentum: 5-day return
    if len(df) >= 6:
        mom = df['Close'].iloc[-1] / df['Close'].iloc[-6] - 1
        # positive momentum increases likelihood for mid/long targets
        if mom > 0.03:
            score += 0.15
        elif mom > 0:
            score += 0.05
    return min(score, 1.0)

# -------------------- Hybrid scoring --------------------

def hybrid_score(df, current_price, target_return, sims, max_days, alpha_prob=0.6):
    """
    Combine probabilistic and technical scores.
    alpha_prob: weight for probabilistic part (0..1); rest for technical.
    Returns combined score 0..1
    """
    p = prob_reach_target(df, current_price, target_return, sims=sims, max_days=max_days)
    t = technical_score(df)
    return alpha_prob * p + (1 - alpha_prob) * t

# -------------------- Fetch & Evaluate single ticker --------------------

def evaluate_ticker(ticker, period, interval, algo, target_return, max_days, sims):
    """
    Fetch data, compute indicators, then compute score based on algo.
    Returns dict with ticker, marketCap, price, score, prob, techscore, note
    """
    hist, info = get_data(ticker, period=period, interval=interval)
    if hist.empty:
        return {"ticker": ticker, "error": info.get("_error", "no data")}
    df = calc_indicators(hist)
    latest = df.iloc[-1]
    current_price = float(latest['Close'])
    market_cap = info.get('marketCap', None)
    prob = prob_reach_target(df, current_price, target_return, sims=sims, max_days=max_days) if algo in ('probabilistic','hybrid') else None
    tech = technical_score(df) if algo in ('technical','hybrid') else None
    if algo == 'probabilistic':
        score = prob
    elif algo == 'technical':
        score = tech
    else:  # hybrid
        # choose alpha based on horizon (longer -> give more weight to prob)
        score = hybrid_score(df, current_price, target_return, sims=sims, max_days=max_days, alpha_prob=0.6)
    return {
        "ticker": ticker,
        "marketCap": market_cap,
        "price": current_price,
        "prob": prob if prob is not None else np.nan,
        "tech": tech if tech is not None else np.nan,
        "score": score if score is not None else 0.0
    }

# -------------------- UI Controls --------------------

st.title("99 Stocks — Algorithmic Selection by Market Cap & Target")

# Left control area — use sidebar with three expanders to represent each bucket's control panel
st.sidebar.header("Bucket Controls")

st.sidebar.markdown("### Small Cap (<= $1B)")
sc_algo = st.sidebar.selectbox("Algorithm (Small Cap)", options=['probabilistic','technical','hybrid'], index=0, key='sc_algo')
sc_target = st.sidebar.number_input("Profit target % (Small Cap)", min_value=1.0, max_value=100.0, value=5.0, step=0.5, key='sc_target')
sc_days = st.sidebar.slider("Days horizon (Small Cap)", min_value=1, max_value=21, value=3, step=1, key='sc_days')

st.sidebar.markdown("### Mid Cap ($1B - $10B)")
mc_algo = st.sidebar.selectbox("Algorithm (Mid Cap)", options=['probabilistic','technical','hybrid'], index=0, key='mc_algo')
mc_target = st.sidebar.number_input("Profit target % (Mid Cap)", min_value=1.0, max_value=200.0, value=10.0, step=0.5, key='mc_target')
mc_days = st.sidebar.slider("Days horizon (Mid Cap)", min_value=3, max_value=60, value=14, step=1, key='mc_days')

st.sidebar.markdown("### Large Cap (> $10B)")
lc_algo = st.sidebar.selectbox("Algorithm (Large Cap)", options=['probabilistic','technical','hybrid'], index=0, key='lc_algo')
lc_target = st.sidebar.number_input("Profit target % (Large Cap)", min_value=1.0, max_value=500.0, value=30.0, step=1.0, key='lc_target')
lc_days = st.sidebar.slider("Days horizon (Large Cap)", min_value=14, max_value=365, value=90, step=1, key='lc_days')

st.sidebar.markdown("---")
st.sidebar.header("Performance & Fetching")
sims = st.sidebar.selectbox("Monte Carlo sims (vectorized)", options=[500,1000,2500,5000], index=2)
batch_limit = st.sidebar.number_input("Max tickers to evaluate from universe", min_value=50, max_value=1000, value=500)
workers = st.sidebar.slider("Parallel fetch workers", 1, 12, 6)
st.sidebar.markdown("---")
if st.sidebar.button("Run selection / Refresh"):
    run_flag = True
else:
    run_flag = False

# Main: fetch universe
with st.spinner("Preparing universe..."):
    universe = fetch_sp500_list()
    if len(universe) == 0:
        st.error("Could not fetch a ticker universe. Update code or provide a CSV.")
        st.stop()
    # limit universe size for speed — up to batch_limit
    universe = universe[:int(batch_limit)]

# -------------------- Fetch market caps & classify --------------------

st.info("Fetching live market caps and classifying tickers (this can take a moment)...")

def fetch_marketcap_to_list(tickers, period='6mo', interval='1d', workers=6):
    results = {}
    errors = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(get_data, t, period, interval): t for t in tickers}
        for fut in as_completed(futures):
            t = futures[fut]
            hist, info = fut.result()
            if hist.empty:
                errors[t] = info.get("_error", "no data")
                continue
            mc = info.get('marketCap', None)
            results[t] = mc
    return results, errors

marketcaps, mc_errors = fetch_marketcap_to_list(universe, period="1y", interval="1d", workers=workers)

# classify
small_ticks = []
mid_ticks = []
large_ticks = []
for t, mc in marketcaps.items():
    if mc is None:
        continue
    # using USD marketCap
    if mc < 1_000_000_000:
        small_ticks.append(t)
    elif mc < 10_000_000_000:
        mid_ticks.append(t)
    else:
        large_ticks.append(t)

st.write(f"Universe evaluated: {len(universe)} tickers. Classified as Small: {len(small_ticks)}, Mid: {len(mid_ticks)}, Large: {len(large_ticks)}")

# -------------------- Evaluate & rank tickers per bucket --------------------

def evaluate_bucket(tickers, algo, target_percent, days, sims, workers, cap_name):
    """
    Evaluate tickers in parallel and return DataFrame of top results (score desc) up to 33.
    """
    target_return = target_percent / 100.0
    results = []
    errors = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(evaluate_ticker, t, period="1y", interval="1d", algo=algo, target_return=target_return, max_days=days, sims=sims): t for t in tickers}
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                r = fut.result()
                if r is None:
                    errors.append((t, "no result"))
                elif 'error' in r:
                    errors.append((t, r['error']))
                else:
                    results.append(r)
            except Exception as e:
                errors.append((t, str(e)))
    if not results:
        return pd.DataFrame(), errors
    df = pd.DataFrame(results)
    # rank by score descending, show up to 33
    df_sorted = df.sort_values(by='score', ascending=False).head(33).reset_index(drop=True)
    return df_sorted, errors

# Only run evaluation if user clicked Run, otherwise show blank / instructions
if run_flag:
    with st.spinner("Evaluating Small Cap bucket..."):
        sc_df, sc_errors = evaluate_bucket(small_ticks, sc_algo, sc_target, sc_days, sims, workers, "Small")
    with st.spinner("Evaluating Mid Cap bucket..."):
        mc_df, mc_errors = evaluate_bucket(mid_ticks, mc_algo, mc_target, mc_days, sims, workers, "Mid")
    with st.spinner("Evaluating Large Cap bucket..."):
        lc_df, lc_errors = evaluate_bucket(large_ticks, lc_algo, lc_target, lc_days, sims, workers, "Large")

    # Display three tables side-by-side (or stacked if narrow)
    st.header("Selected Stocks (up to 33 each)")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader(f"Small Cap (<= $1B) — target {sc_target}% in {sc_days}d — algo: {sc_algo}")
        if sc_df is not None and not sc_df.empty:
            sc_df_display = sc_df.copy()
            sc_df_display['prob'] = sc_df_display['prob'].round(3)
            sc_df_display['tech'] = sc_df_display['tech'].round(3)
            sc_df_display['score'] = sc_df_display['score'].round(3)
            st.dataframe(sc_df_display[['ticker','price','marketCap','prob','tech','score']])
        else:
            st.info("No candidates found or not enough data.")
        if sc_errors:
            st.write("Errors:", sc_errors[:5])

    with c2:
        st.subheader(f"Mid Cap ($1B–$10B) — target {mc_target}% in {mc_days}d — algo: {mc_algo}")
        if mc_df is not None and not mc_df.empty:
            mc_df_display = mc_df.copy()
            mc_df_display['prob'] = mc_df_display['prob'].round(3)
            mc_df_display['tech'] = mc_df_display['tech'].round(3)
            mc_df_display['score'] = mc_df_display['score'].round(3)
            st.dataframe(mc_df_display[['ticker','price','marketCap','prob','tech','score']])
        else:
            st.info("No candidates found or not enough data.")
        if mc_errors:
            st.write("Errors:", mc_errors[:5])

    with c3:
        st.subheader(f"Large Cap (> $10B) — target {lc_target}% in {lc_days}d — algo: {lc_algo}")
        if lc_df is not None and not lc_df.empty:
            lc_df_display = lc_df.copy()
            lc_df_display['prob'] = lc_df_display['prob'].round(3)
            lc_df_display['tech'] = lc_df_display['tech'].round(3)
            lc_df_display['score'] = lc_df_display['score'].round(3)
            st.dataframe(lc_df_display[['ticker','price','marketCap','prob','tech','score']])
        else:
            st.info("No candidates found or not enough data.")
        if lc_errors:
            st.write("Errors:", lc_errors[:5])

    st.markdown("---")
    st.write("Notes:")
    st.write("- Probabilities from MC are NOT guarantees. Backtest before trading.")
    st.write("- Technical score is heuristic; adjust thresholds and logic if needed.")
    st.write("- If many tickers lack marketCap in yfinance, consider expanding universe or using paid data.")
else:
    st.info("Configure your bucket controls on the left and click 'Run selection / Refresh' to evaluate up to 99 stocks.")

# Footer: quick help
st.markdown("---")
st.write("Built with yfinance — fetches live marketCap to classify into buckets. Use smaller batch limits / fewer sims on limited hardware.")

