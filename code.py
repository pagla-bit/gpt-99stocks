# streamlit_99_stocks_dynamic.py
"""
99 Stocks — dynamic index-based universe
- Russell 2000 (small), S&P 400 (mid), S&P 500 (large) — fetched from Wikipedia
- Index boundaries are FIXED by source (Russell -> small, S&P400 -> mid, S&P500 -> large)
- Three algorithm modes per bucket: probabilistic, technical, hybrid
- Up to 33 top-ranked tickers shown per bucket
- Vectorized Monte Carlo (probabilistic), heuristic technical score, hybrid combination
- Parallel fetching and caching. Expect long runtimes for full universe; adjust workers / sims.
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

st.set_page_config(layout="wide", page_title="99 Stocks — Dynamic Indices")

# ------------------------- Helper: Fetch index constituents -------------------------

@st.cache_data(ttl=3600)
def fetch_table_from_wiki(url, expected_symbol_cols=('Symbol','Ticker','Ticker symbol','Ticker Symbol'), verbose=False):
    """
    Use pandas.read_html to find a table containing a column for the ticker.
    Returns list of tickers (strings) or raises ValueError if none found.
    """
    try:
        tables = pd.read_html(url)
    except Exception as e:
        raise ValueError(f"Could not read tables from {url}: {e}")
    for t in tables:
        cols = [str(c).strip() for c in t.columns]
        # check if any expected symbol column exists
        for symcol in expected_symbol_cols:
            if symcol in cols:
                ticks = t[symcol].astype(str).str.replace('.', '-', regex=False).str.strip().tolist()
                ticks = [x for x in ticks if x and x != 'nan']
                if len(ticks) == 0:
                    continue
                return ticks
    # sometimes the symbol column might be present but with different name; try heuristics
    for t in tables:
        # flatten all text and search for uppercase symbols-like strings in first column
        first_col = t.iloc[:,0].astype(str).tolist()
        candidates = []
        for val in first_col:
            if isinstance(val, str) and 1 < len(val) <= 10 and val.isupper() and not val.isdigit():
                candidates.append(val.replace('.', '-'))
        if len(candidates) > 10:
            return candidates
    raise ValueError(f"No ticker-like column found in tables from {url}")

@st.cache_data(ttl=3600)
def fetch_sp500_list():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    return fetch_table_from_wiki(url, expected_symbol_cols=('Symbol','Ticker'))

@st.cache_data(ttl=3600)
def fetch_sp400_list():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
    return fetch_table_from_wiki(url, expected_symbol_cols=('Symbol','Ticker'))

@st.cache_data(ttl=3600)
def fetch_russell2000_list():
    """
    Try a few likely Wikipedia pages that might contain Russell 2000 constituents.
    If none are present, raise a ValueError — user must ensure internet access / page layout.
    """
    # Try some plausible pages containing lists; these pages change often so we attempt a couple of URLs.
    candidates = [
        "https://en.wikipedia.org/wiki/Russell_2000_Index",
        "https://en.wikipedia.org/wiki/List_of_companies_in_the_Russell_2000_Index",
        "https://en.wikipedia.org/wiki/List_of_Russell_2000_companies"
    ]
    last_err = None
    for url in candidates:
        try:
            ticks = fetch_table_from_wiki(url, expected_symbol_cols=('Ticker','Symbol','Ticker symbol'), verbose=True)
            # sanity check that we got many tickers (Russell is ~2000)
            if len(ticks) >= 100:
                return ticks
            # if we got a small list, still return it if >30
            if len(ticks) >= 30:
                return ticks
            # else continue trying
        except Exception as e:
            last_err = e
            continue
    # As last attempt, try to read from other wikis that may have lists (community pages)
    raise ValueError(f"Could not fetch Russell 2000 constituents from Wikipedia pages. Last error: {last_err}")

# ------------------------- Data fetching & caching -------------------------

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

# ------------------------- Indicators & scoring -------------------------

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
    plus_di = 100 * (plus_dm.ewm(span=adx_period).mean() / df['ATR'])
    minus_di = 100 * (minus_dm.ewm(span=adx_period).mean() / df['ATR'])
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    df['ADX'] = dx.ewm(span=adx_period).mean()
    return df

def prob_reach_target(df, current_price, target_return, sims=2500, max_days=90):
    returns = df['Close'].pct_change().dropna().values
    if len(returns) < 30:
        return 0.0
    mu = returns.mean()
    sigma = returns.std()
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
    if not np.isnan(prev['MACD']) and not np.isnan(prev['MACD_signal']):
        if (prev['MACD'] < prev['MACD_signal']) and (latest['MACD'] > latest['MACD_signal']):
            score += 0.25
    if not np.isnan(latest['SMA_long']):
        if latest['Close'] > latest['SMA_long']:
            score += 0.15
    if not np.isnan(latest['ADX']) and latest['ADX'] > 25:
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

# ------------------------- Ticker evaluation -------------------------

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
    score = 0.0
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

# ------------------------- Streamlit UI -------------------------

st.title("99 Stocks — Dynamic Indices (Russell2000 / S&P400 / S&P500)")

st.sidebar.header("Bucket Controls — Fixed by Index (Russell -> Small, S&P400 -> Mid, S&P500 -> Large)")

st.sidebar.markdown("### Small Cap (Russell 2000)")
sc_algo = st.sidebar.selectbox("Algorithm (Small Cap)", ['probabilistic','technical','hybrid'], index=0, key='sc_algo')
sc_target = st.sidebar.number_input("Profit target % (Small Cap)", min_value=1.0, max_value=100.0, value=5.0, step=0.5, key='sc_target')
sc_days = st.sidebar.slider("Days horizon (Small Cap)", min_value=1, max_value=21, value=3, step=1, key='sc_days')

st.sidebar.markdown("### Mid Cap (S&P 400)")
mc_algo = st.sidebar.selectbox("Algorithm (Mid Cap)", ['probabilistic','technical','hybrid'], index=0, key='mc_algo')
mc_target = st.sidebar.number_input("Profit target % (Mid Cap)", min_value=1.0, max_value=200.0, value=10.0, step=0.5, key='mc_target')
mc_days = st.sidebar.slider("Days horizon (Mid Cap)", min_value=3, max_value=60, value=14, step=1, key='mc_days')

st.sidebar.markdown("### Large Cap (S&P 500)")
lc_algo = st.sidebar.selectbox("Algorithm (Large Cap)", ['probabilistic','technical','hybrid'], index=0, key='lc_algo')
lc_target = st.sidebar.number_input("Profit target % (Large Cap)", min_value=1.0, max_value=500.0, value=30.0, step=1.0, key='lc_target')
lc_days = st.sidebar.slider("Days horizon (Large Cap)", min_value=14, max_value=365, value=90, step=1, key='lc_days')

st.sidebar.markdown("---")
st.sidebar.header("Performance Controls")
sims = st.sidebar.selectbox("Monte Carlo sims (vectorized)", options=[500,1000,2500,5000], index=2)
workers = st.sidebar.slider("Parallel workers (fetch & eval)", 2, 24, 8)
st.sidebar.markdown("**Important:** Fetching and evaluating ALL tickers (Russell2000 + S&P400 + S&P500) can take many minutes depending on workers/sims and API rate limits.")
st.sidebar.markdown("---")
run_request = st.sidebar.button("Run evaluation for full indices")

# ------------------------- Fetch tickers for each index -------------------------
status_placeholder = st.empty()
status_placeholder.info("Fetching index constituent lists from Wikipedia... (this occurs once per hour cached)")

try:
    with st.spinner("Fetching S&P 500 tickers..."):
        sp500_list = fetch_sp500_list()
    with st.spinner("Fetching S&P 400 tickers..."):
        sp400_list = fetch_sp400_list()
    with st.spinner("Fetching Russell 2000 tickers..."):
        russell_list = fetch_russell2000_list()
except Exception as e:
    st.error(f"Could not fetch index constituents: {e}")
    st.stop()

status_placeholder.success(f"Fetched lists — S&P500: {len(sp500_list)} | S&P400: {len(sp400_list)} | Russell2000: {len(russell_list)}")

# Keep index classification fixed as requested: Russell -> small, S&P400 -> mid, S&P500 -> large
small_universe = sorted(list(set(russell_list)))
mid_universe = sorted(list(set(sp400_list)))
large_universe = sorted(list(set(sp500_list)))

st.markdown("### Universe sizes (pre-evaluation)")
st.write(f"Small (Russell2000) tickers: {len(small_universe)}")
st.write(f"Mid (S&P400) tickers: {len(mid_universe)}")
st.write(f"Large (S&P500) tickers: {len(large_universe)}")

# ------------------------- Run evaluation if requested -------------------------

def evaluate_bucket_parallel(tickers, algo, target_pct, days, sims, workers, max_results=33):
    target_return = float(target_pct) / 100.0
    results = []
    errors = []
    # evaluate in parallel
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(evaluate_ticker, t, algo, target_return, sims, days): t for t in tickers}
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                r = fut.result()
                if r is None:
                    errors.append((t, "no result"))
                    continue
                if 'error' in r:
                    errors.append((t, r['error']))
                    continue
                results.append(r)
            except Exception as e:
                errors.append((t, str(e)))
    if not results:
        return pd.DataFrame(), errors
    df = pd.DataFrame(results)
    df_sorted = df.sort_values(by='score', ascending=False).head(max_results).reset_index(drop=True)
    return df_sorted, errors

if run_request:
    st.info("Running full evaluation. This may take several minutes — progress logs will appear below.")
    # Evaluate each bucket — these may take time depending on universe sizes & workers
    with st.spinner("Evaluating Small Cap (Russell2000) — this can be the longest..."):
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
            st.info("No results / insufficient data for Small Cap candidates.")
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
            st.info("No results / insufficient data for Mid Cap candidates.")
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
            st.info("No results / insufficient data for Large Cap candidates.")
        if lc_errs:
            st.write("Sample errors (first 10):")
            for e in lc_errs[:10]:
                st.write(f"- {e[0]}: {e[1]}")

    st.markdown("---")
    st.write("Notes & caveats:")
    st.write("- Index constituent scraping depends on Wikipedia page structure. If the Russell2000 list page lacks a table, the fetch may fail; check internet access or Wikipedia layout.")
    st.write("- Running the full universe is network- and CPU-heavy; consider limiting workers or using a subset for iterative development.")
    st.write("- Probabilities are model outputs from historical-return Monte Carlo (normal approx). They are NOT guarantees.")
else:
    st.info("Configure the three buckets on the left. Click 'Run evaluation for full indices' to fetch & evaluate Russell2000, S&P400 and S&P500 tickers (this fetches ALL tickers from those indices).")

st.markdown("---")
st.write("If you want: I can add CSV export buttons, incremental evaluation (evaluate top-N by market cap first), or persistent caching to avoid re-evaluating unchanged tickers. Tell me which next.")
