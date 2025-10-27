# streamlit_99_stocks_dynamic_fixed.py
"""
99 Stocks — Dynamic Indices (fixed fetch + headers)
- Fetch Russell2000, S&P400, S&P500 constituents from Wikipedia with a User-Agent header
- Index boundaries fixed (Russell -> small, S&P400 -> mid, S&P500 -> large)
- Three algorithm modes per bucket: probabilistic, technical, hybrid
- Up to 33 top-ranked tickers per bucket
- Vectorized Monte Carlo, parallel evaluation, caching
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import requests
from io import StringIO
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(layout="wide", page_title="99 Stocks — Dynamic Indices (Fixed)")

# -------------------- Utility: robust read_html with headers --------------------

def read_html_with_headers(url, timeout=15):
    """
    Read HTML from url using requests with a browser-like User-Agent, then parse via pandas.read_html.
    Raises RuntimeError with helpful message on failure.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        html = resp.text
        # pandas will try available parsers (lxml/bs4)
        tables = pd.read_html(StringIO(html))
        if not tables:
            raise RuntimeError("No HTML tables found on the page.")
        return tables
    except Exception as e:
        # Provide actionable guidance
        raise RuntimeError(
            f"Could not fetch/parse tables from {url}. "
            "Possible reasons: network blocked or pandas parser deps missing (lxml or beautifulsoup4+html5lib). "
            "Fixes: install 'lxml' (pip install lxml) or 'beautifulsoup4 html5lib' (pip install beautifulsoup4 html5lib). "
            f"Underlying error: {e}"
        )

# -------------------- Fetch index constituent lists (cached) --------------------

@st.cache_data(ttl=86400)
def fetch_sp500_list():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = read_html_with_headers(url)
    # Primary table is usually first
    df = tables[0]
    # Typical column name is 'Symbol' but be defensive
    col = None
    for c in df.columns:
        if str(c).strip().lower() in ("symbol", "ticker", "ticker symbol"):
            col = c
            break
    if col is None:
        # try heuristics
        possible = [c for c in df.columns if 'symbol' in str(c).lower() or 'ticker' in str(c).lower()]
        if possible:
            col = possible[0]
    if col is None:
        raise RuntimeError("Could not find a 'Symbol' column in S&P500 Wikipedia table.")
    tickers = df[col].astype(str).str.replace('.', '-', regex=False).str.strip().tolist()
    return [t for t in tickers if t and t != 'nan']

@st.cache_data(ttl=86400)
def fetch_sp400_list():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
    tables = read_html_with_headers(url)
    df = tables[0]
    col = None
    for c in df.columns:
        if str(c).strip().lower() in ("symbol", "ticker", "ticker symbol"):
            col = c
            break
    if col is None:
        possible = [c for c in df.columns if 'symbol' in str(c).lower() or 'ticker' in str(c).lower()]
        if possible:
            col = possible[0]
    if col is None:
        raise RuntimeError("Could not find a 'Symbol' column in S&P400 Wikipedia table.")
    tickers = df[col].astype(str).str.replace('.', '-', regex=False).str.strip().tolist()
    return [t for t in tickers if t and t != 'nan']

@st.cache_data(ttl=86400)
def fetch_russell2000_list():
    """
    Russell 2000 page layout can vary. Try multiple candidate pages and heuristics.
    """
    candidate_urls = [
        "https://en.wikipedia.org/wiki/Russell_2000_Index",
        "https://en.wikipedia.org/wiki/List_of_companies_in_the_Russell_2000_Index",
        "https://en.wikipedia.org/wiki/List_of_Russell_2000_companies"
    ]
    last_error = None
    for url in candidate_urls:
        try:
            tables = read_html_with_headers(url)
            # Attempt to find a ticker-like column
            for df in tables:
                # normalize column names and search
                cols = [str(c).strip().lower() for c in df.columns]
                for target in ("symbol","ticker","ticker symbol"):
                    if target in cols:
                        col = df.columns[cols.index(target)]
                        ticks = df[col].astype(str).str.replace('.', '-', regex=False).str.strip().tolist()
                        ticks = [t for t in ticks if t and t != 'nan']
                        if len(ticks) >= 30:  # some reasonable minimum
                            return ticks
            # heuristic: maybe tickers are in first column as uppercase codes
            for df in tables:
                first_col = df.iloc[:,0].astype(str).str.strip().tolist()
                cand = []
                for v in first_col:
                    if isinstance(v,str) and 1 < len(v) <= 6 and v.isupper() and not any(c.isdigit() for c in v):
                        cand.append(v.replace('.', '-'))
                if len(cand) >= 30:
                    return cand
        except Exception as e:
            last_error = e
            continue
    raise RuntimeError(f"Could not fetch Russell2000 tickers from Wikipedia. Last error: {last_error}")

# -------------------- Data fetcher (yfinance) --------------------

@st.cache_data(ttl=1800)
def get_data(ticker, period="1y", interval="1d"):
    """
    Fetch OHLCV and info via yfinance. Returns (hist_df, info_dict) or (empty_df, {'_error':..}).
    """
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period=period, interval=interval, actions=False, auto_adjust=True)
        if hist.empty:
            return pd.DataFrame(), {"_error": "empty history"}
        required_cols = ['Open','High','Low','Close','Volume']
        if not set(required_cols).issubset(set(hist.columns)):
            return pd.DataFrame(), {"_error": "incomplete OHLCV"}
        info = tk.info if hasattr(tk, "info") else {}
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
    hc = (df['High'] - df['Close'].shift()).abs()
    lc = (df['Low'] - df['Close'].shift()).abs()
    tr = np.maximum(hl, np.maximum(hc, lc))
    df['ATR'] = tr.ewm(span=atr_period, adjust=False).mean()
    plus_dm = df['High'].diff()
    minus_dm = -df['Low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    plus_di = 100 * (plus_dm.ewm(span=adx_period).mean() / (df['ATR'].replace(0, np.nan)))
    minus_di = 100 * (minus_dm.ewm(span=adx_period).mean() / (df['ATR'].replace(0, np.nan)))
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)).replace([np.inf, -np.inf], np.nan) * 100
    df['ADX'] = dx.ewm(span=adx_period).mean()
    return df

def prob_reach_target(df, current_price, target_return, sims=2500, max_days=90):
    returns = df['Close'].pct_change().dropna().values
    if len(returns) < 30:
        return 0.0
    mu = np.nanmean(returns)
    sigma = np.nanstd(returns)
    if sigma == 0 or np.isnan(sigma):
        return 0.0
    rand = np.random.normal(loc=mu, scale=sigma, size=(sims, max_days))
    price_paths = current_price * np.cumprod(1 + rand, axis=1)
    threshold = current_price * (1 + target_return)
    hits = (price_paths >= threshold).any(axis=1)
    return float(np.mean(hits))

def technical_score(df):
    if len(df) < 60:
        return 0.0
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    score = 0.0
    if not np.isnan(latest['RSI']):
        if latest['RSI'] < 30:
            score += 0.35
        elif latest['RSI'] < 40:
            score += 0.15
    if (not np.isnan(prev.get('MACD', np.nan))) and (not np.isnan(prev.get('MACD_signal', np.nan))):
        if (prev['MACD'] < prev['MACD_signal']) and (latest['MACD'] > latest['MACD_signal']):
            score += 0.25
    if not np.isnan(latest.get('SMA_long', np.nan)) and latest['Close'] > latest['SMA_long']:
        score += 0.15
    if not np.isnan(latest.get('ADX', np.nan)) and latest['ADX'] > 25:
        score += 0.10
    if len(df) >= 6:
        mom = df['Close'].iloc[-1] / df['Close'].iloc[-6] - 1
        if mom > 0.03:
            score += 0.15
        elif mom > 0:
            score += 0.05
    return min(score, 1.0)

def hybrid_score(df, current_price, target_return, sims, max_days, alpha_prob=0.6):
    p = prob_reach_target(df, current_price, target_return, sims=sims, max_days=max_days)
    t = technical_score(df)
    return alpha_prob * p + (1 - alpha_prob) * t

# -------------------- Ticker evaluation --------------------

def evaluate_ticker(ticker, algo, target_return, sims, max_days, period="1y", interval="1d"):
    hist, info = get_data(ticker, period=period, interval=interval)
    if hist.empty:
        return {"ticker": ticker, "error": info.get("_error", "no data")}
    df = calc_indicators(hist)
    latest = df.iloc[-1]
    current_price = float(latest['Close'])
    market_cap = info.get('marketCap', None)
    prob = None
    tech = None
    if algo in ('probabilistic', 'hybrid'):
        prob = prob_reach_target(df, current_price, target_return, sims=sims, max_days=max_days)
    if algo in ('technical', 'hybrid'):
        tech = technical_score(df)
    if algo == 'probabilistic':
        score = prob
    elif algo == 'technical':
        score = tech
    else:
        score = hybrid_score(df, current_price, target_return, sims=sims, max_days=max_days, alpha_prob=0.6)
    return {"ticker": ticker, "marketCap": market_cap, "price": current_price, "prob": prob if prob is not None else np.nan, "tech": tech if tech is not None else np.nan, "score": score}

# -------------------- Streamlit UI --------------------

st.title("99 Stocks — Dynamic Indices (Russell2000 / S&P400 / S&P500) — FIXED FETCH")

st.sidebar.header("Bucket Controls — Fixed by Index (Russell -> Small, S&P400 -> Mid, S&P500 -> Large)")

# Small cap controls
st.sidebar.markdown("### Small Cap (Russell 2000)")
sc_algo = st.sidebar.selectbox("Algorithm (Small Cap)", ['probabilistic','technical','hybrid'], index=0, key='sc_algo')
sc_target = st.sidebar.number_input("Profit target % (Small Cap)", min_value=1.0, max_value=100.0, value=5.0, step=0.5, key='sc_target')
sc_days = st.sidebar.slider("Days horizon (Small Cap)", min_value=1, max_value=30, value=3, step=1, key='sc_days')

# Mid cap
st.sidebar.markdown("### Mid Cap (S&P 400)")
mc_algo = st.sidebar.selectbox("Algorithm (Mid Cap)", ['probabilistic','technical','hybrid'], index=0, key='mc_algo')
mc_target = st.sidebar.number_input("Profit target % (Mid Cap)", min_value=1.0, max_value=200.0, value=10.0, step=0.5, key='mc_target')
mc_days = st.sidebar.slider("Days horizon (Mid Cap)", min_value=3, max_value=90, value=14, step=1, key='mc_days')

# Large cap
st.sidebar.markdown("### Large Cap (S&P 500)")
lc_algo = st.sidebar.selectbox("Algorithm (Large Cap)", ['probabilistic','technical','hybrid'], index=0, key='lc_algo')
lc_target = st.sidebar.number_input("Profit target % (Large Cap)", min_value=1.0, max_value=500.0, value=30.0, step=1.0, key='lc_target')
lc_days = st.sidebar.slider("Days horizon (Large Cap)", min_value=14, max_value=365, value=90, step=1, key='lc_days')

st.sidebar.markdown("---")
st.sidebar.header("Performance Controls")
sims = st.sidebar.selectbox("Monte Carlo sims (vectorized)", options=[500,1000,2500,5000], index=2)
workers = st.sidebar.slider("Parallel workers (fetch & eval)", 2, 24, 8)
st.sidebar.markdown("**Warning:** Fetching and evaluating the entire universe (Russell2000+S&P400+S&P500) can take several minutes.")
st.sidebar.markdown("---")
run_request = st.sidebar.button("Run evaluation for full indices (fetch ALL)")

# -------------------- Fetch index lists --------------------

status = st.empty()
status.info("Fetching index constituent lists from Wikipedia... (cached hourly)")

try:
    sp500_list = fetch_sp500_list()
    sp400_list = fetch_sp400_list()
    russell_list = fetch_russell2000_list()
except Exception as e:
    status.error(f"Could not fetch index constituents: {e}")
    st.stop()

status.success(f"Fetched: S&P500={len(sp500_list)} | S&P400={len(sp400_list)} | Russell2000={len(russell_list)}")

# Keep index-based fixed mapping
small_universe = sorted(list(set(russell_list)))
mid_universe = sorted(list(set(sp400_list)))
large_universe = sorted(list(set(sp500_list)))

st.markdown("### Universe sizes (pre-evaluation)")
st.write(f"Small (Russell2000): {len(small_universe)} tickers")
st.write(f"Mid (S&P400): {len(mid_universe)} tickers")
st.write(f"Large (S&P500): {len(large_universe)} tickers")

# -------------------- Run evaluation if requested --------------------

def evaluate_bucket_parallel(tickers, algo, target_pct, days, sims, workers, max_results=33):
    target_return = float(target_pct) / 100.0
    results = []
    errors = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(evaluate_ticker, t, algo, target_return, sims, days): t for t in tickers}
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
    df_sorted = df.sort_values(by='score', ascending=False).head(max_results).reset_index(drop=True)
    return df_sorted, errors

if run_request:
    st.info("Running evaluation. This may take several minutes; progress will stream in stages.")
    with st.spinner("Evaluating Small Cap (Russell2000) — may take the longest..."):
        sc_df, sc_errs = evaluate_bucket_parallel(small_universe, sc_algo, sc_target, sc_days, sims, workers, max_results=33)
    with st.spinner("Evaluating Mid Cap (S&P400)..."):
        mc_df, mc_errs = evaluate_bucket_parallel(mid_universe, mc_algo, mc_target, mc_days, sims, workers, max_results=33)
    with st.spinner("Evaluating Large Cap (S&P500)..."):
        lc_df, lc_errs = evaluate_bucket_parallel(large_universe, lc_algo, lc_target, lc_days, sims, workers, max_results=33)

    st.header("Top candidates (up to 33 per bucket)")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader(f"Small Cap (Russell2000) — target {sc_target}% in {sc_days}d — algo: {sc_algo}")
        if sc_df is not None and not sc_df.empty:
            display = sc_df.copy()
            display['prob'] = display['prob'].round(4)
            display['tech'] = display['tech'].round(4)
            display['score'] = display['score'].round(4)
            st.dataframe(display[['ticker','price','marketCap','prob','tech','score']])
        else:
            st.info("No results / insufficient data for Small Cap.")
        if sc_errs:
            st.write("Sample errors (first 10):")
            for e in sc_errs[:10]:
                st.write(f"- {e[0]}: {e[1]}")

    with c2:
        st.subheader(f"Mid Cap (S&P400) — target {mc_target}% in {mc_days}d — algo: {mc_algo}")
        if mc_df is not None and not mc_df.empty:
            display = mc_df.copy()
            display['prob'] = display['prob'].round(4)
            display['tech'] = display['tech'].round(4)
            display['score'] = display['score'].round(4)
            st.dataframe(display[['ticker','price','marketCap','prob','tech','score']])
        else:
            st.info("No results / insufficient data for Mid Cap.")
        if mc_errs:
            st.write("Sample errors (first 10):")
            for e in mc_errs[:10]:
                st.write(f"- {e[0]}: {e[1]}")

    with c3:
        st.subheader(f"Large Cap (S&P500) — target {lc_target}% in {lc_days}d — algo: {lc_algo}")
        if lc_df is not None and not lc_df.empty:
            display = lc_df.copy()
            display['prob'] = display['prob'].round(4)
            display['tech'] = display['tech'].round(4)
            display['score'] = display['score'].round(4)
            st.dataframe(display[['ticker','price','marketCap','prob','tech','score']])
        else:
            st.info("No results / insufficient data for Large Cap.")
        if lc_errs:
            st.write("Sample errors (first 10):")
            for e in lc_errs[:10]:
                st.write(f"- {e[0]}: {e[1]}")

    st.markdown("---")
    st.write("Notes & caveats:")
    st.write("- Index scraping depends on Wikipedia page structure. If Russell2000 page lacks table data, fetch may require adjustment.")
    st.write("- Running the full universe is network/CPU intensive. Reduce 'workers' or 'sims' for faster runs.")
    st.write("- Probabilities are historical-model outputs (not guarantees). Backtest before trading.")
else:
    st.info("Configure the three bucket controls in the sidebar and click 'Run evaluation for full indices' to start.")

st.markdown("---")
st.write("If you want: I can add CSV export buttons, incremental evaluation (top-N by market cap first), or persistent caching for results. Which would you like next?")
