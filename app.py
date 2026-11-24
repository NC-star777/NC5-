import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import time
import os
import datetime

st.set_page_config(page_title="NC5 Institutional Multi-Asset Screener & Master Grader", layout="wide")

st.title("📈 NC5 Institutional Multi-Asset Screener & Master Grader")
st.write("Live charting, multi-factor grading (F1–F11), S&P 500 / Russell index weighting, 1-week swing pattern detection, and dual-system backtesting.")

# ==========================================
# WEIGHT CONFIGURATIONS & HELPER FUNCTIONS
# ==========================================
SP500_WEIGHTS = {
    "f1": 0.30, "f2": 0.15, "f3": 0.04, "f4": 0.04,
    "f5": 0.10, "f6": 0.04, "f7": 0.04, "f8": 0.04,
    "f9": 0.10, "f10": 0.10, "f11": 0.05
}

RUSSELL_WEIGHTS = {
    "f1": 0.25, "f2": 0.10, "f3": 0.05, "f4": 0.04,
    "f5": 0.10, "f6": 0.04, "f7": 0.06, "f8": 0.08,
    "f9": 0.08, "f10": 0.10, "f11": 0.10
}

def clamp(val, low=1, high=100):
    return int(max(low, min(high, round(val))))

@st.cache_data(ttl=3600)
def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get(url, headers=headers, timeout=15)
        tables = pd.read_html(res.text)
        return tables[0]["Symbol"].str.replace(".", "-", regex=False).tolist()
    except Exception:
        return [
            "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "JPM", "BAC", "V", "MA",
            "UNH", "JNJ", "LLY", "PFE", "HD", "NKE", "PG", "COST", "WMT", "KO", "XOM", "CVX",
            "COP", "HON", "UPS", "CAT", "LIN", "FCX", "NEE", "DUK", "AMT", "PLD"
        ]

# ==========================================
# TECHNICAL PATTERN DETECTORS (1-Week Window)
# ==========================================
def detect_patterns(hist):
    if hist is None or len(hist) < 200:
        return []

    patterns_found = []
    close = hist["Close"]
    high = hist["High"]
    low = hist["Low"]
    volume = hist["Volume"]

    sma_20 = close.rolling(20).mean()
    sma_50 = close.rolling(50).mean()
    sma_200 = close.rolling(200).mean()

    # ATR (14-period)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(14).mean()

    # RSI (2-period)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain_2 = gain.rolling(2).mean()
    avg_loss_2 = loss.rolling(2).mean()
    rs_2 = avg_gain_2 / avg_loss_2.replace(0, np.nan)
    rsi_2 = 100 - (100 / (1 + rs_2))
    rsi_2 = rsi_2.fillna(100)

    curr_close = close.iloc[-1]
    curr_low = low.iloc[-1]
    curr_atr = atr_14.iloc[-1]

    # --- PATTERN 1: High-Tight Bull Flag ---
    ret_5d = (curr_close - close.iloc[-6]) / close.iloc[-6]
    flag_vol_low = volume.iloc[-4:].mean() < (volume.iloc[-10:-4].mean() * 0.7)
    flag_tight = (high.iloc[-4:].max() - low.iloc[-4:].min()) < (2.5 * curr_atr)
    if ret_5d > 0.08 and flag_tight and flag_vol_low and curr_close >= high.iloc[-4:-1].max():
        patterns_found.append("High-Tight Bull Flag")

    # --- PATTERN 2: 20-Day SMA Pullback ---
    is_uptrend = (curr_close > sma_50.iloc[-1]) and (sma_50.iloc[-1] > sma_200.iloc[-1]) and (sma_20.iloc[-1] > sma_20.iloc[-5])
    touched_sma20 = abs(curr_low - sma_20.iloc[-1]) <= (0.007 * curr_close) or (curr_low <= sma_20.iloc[-1] and curr_close >= sma_20.iloc[-1])
    is_green_day = curr_close > hist["Open"].iloc[-1]
    if is_uptrend and touched_sma20 and is_green_day:
        patterns_found.append("20-Day SMA Pullback")

    # --- PATTERN 3: NR7 / Volatility Squeeze ---
    daily_range = high - low
    is_nr7 = daily_range.iloc[-1] == daily_range.iloc[-7:].min()
    if is_uptrend and is_nr7 and (curr_close > sma_20.iloc[-1]):
        patterns_found.append("NR7 Volatility Squeeze")

    # --- PATTERN 4: RSI(2) Oversold Dip ---
    if (curr_close > sma_200.iloc[-1]) and (rsi_2.iloc[-2] < 10 or rsi_2.iloc[-1] < 12):
        patterns_found.append("RSI(2) Oversold Dip")

    # --- PATTERN 5: Post-Earnings Box Breakout ---
    recent_volumes = volume.iloc[-7:]
    recent_gaps = (high.iloc[-7:] - low.iloc[-7:].shift(1)) / low.iloc[-7:].shift(1)
    had_earnings_spike = (recent_gaps > 0.04).any() and (recent_volumes > (volume.rolling(20).mean().iloc[-1] * 2.0)).any()
    box_tight = (high.iloc[-4:].max() - low.iloc[-4:].min()) < (3.0 * curr_atr)
    if had_earnings_spike and box_tight and (curr_close >= high.iloc[-5:-1].max()):
        patterns_found.append("Post-Earnings Box Breakout")

    return patterns_found

# ==========================================
# NC5 FUNDAMENTAL & TECHNICAL SCORING ENGINE
# ==========================================
@st.cache_data(ttl=3600)
def score_equity_full(ticker_symbol, universe="sp500"):
    try:
        tk = yf.Ticker(ticker_symbol)
        info = tk.info or {}
        hist = tk.history(period="1y")

        fwd_pe = info.get("forwardPE")
        growth_dec = info.get("earningsGrowth")
        if fwd_pe is None or fwd_pe <= 0:
            f1 = 50
        else:
            growth_pct = max(0.0, (growth_dec or 0.0) * 100.0)
            justified_pe = 10.0 + (2.0 * growth_pct)
            val_ratio = min(2.0, justified_pe / fwd_pe)
            f1 = clamp(50.0 * val_ratio)

        rec_mean = info.get("recommendationMean")
        f2 = 50 if rec_mean is None else clamp(100.0 - ((rec_mean - 1.0) * 25.0))

        rec_key = str(info.get("recommendationKey", "")).lower().strip()
        tech_map = {"strong_buy": 95, "buy": 75, "hold": 50, "sell": 25, "strong_sell": 5}
        f3 = tech_map.get(rec_key, 50)

        summary = str(info.get("longBusinessSummary", "")).lower()
        risk_flags = ["class action", "sec investigation", "department of justice"]
        f4 = 25 if any(flag in summary for flag in risk_flags) else 85

        if hist is None or len(hist) < 170:
            f5 = 50
        else:
            window = hist.tail(170)
            high_170, low_170, last_close = window["High"].max(), window["Low"].min(), window["Close"].iloc[-1]
            pos_ratio = 0.5 if high_170 == low_170 else (last_close - low_170) / (high_170 - low_170)
            f5 = clamp(100.0 - (abs(pos_ratio - 0.5) * 100.0))

        industry = str(info.get("industry", "")).lower()
        sticky_keywords = ["aerospace & defense", "software", "utilities", "healthcare", "semiconductors", "communications", "bonds", "credit", "etf"]
        f6 = 90 if any(k in industry for k in sticky_keywords) else 65

        insiders = info.get("heldPercentInsiders")
        f7 = 40 if insiders is None else clamp(50.0 + ((insiders * 100.0) * 2.0))

        fcf, mkt_cap = info.get("freeCashflow"), info.get("marketCap")
        if fcf is None or not mkt_cap or mkt_cap <= 0:
            f8 = 50
        else:
            fcf_yield = (fcf / mkt_cap) * 100.0
            f8 = 95 if fcf_yield > 7.0 else (75 if fcf_yield > 4.0 else (55 if fcf_yield > 0.0 else 15))

        inst = info.get("heldPercentInstitutions")
        f9 = 40 if inst is None else (90 if (inst * 100.0) > 70.0 else (70 if (inst * 100.0) >= 40.0 else (50 if (inst * 100.0) >= 10.0 else 20)))

        try:
            ed = tk.earnings_dates
            if ed is not None and not ed.empty:
                valid_ed = ed.dropna(subset=["Reported EPS", "EPS Estimate"])
                if not valid_ed.empty:
                    actual_eps, est_eps = valid_ed["Reported EPS"].iloc[0], valid_ed["EPS Estimate"].iloc[0]
                    surprise_pct = 0 if est_eps == 0 else ((actual_eps - est_eps) / abs(est_eps)) * 100.0
                    f10 = 100 if surprise_pct >= -5.0 else clamp(100.0 - (3.0 * abs(surprise_pct + 5.0)))
                else:
                    f10 = 50
            else:
                f10 = 50
        except Exception:
            f10 = 50

        f11 = 50
        try:
            bs = tk.balance_sheet
            if bs is not None and not bs.empty:
                curr_assets = bs.loc["Current Assets"].iloc[0] if "Current Assets" in bs.index else bs.iloc[0].iloc[0]
                curr_liab = bs.loc["Current Liabilities"].iloc[0] if "Current Liabilities" in bs.index else bs.iloc[1].iloc[0]
                if curr_liab > 0:
                    curr_ratio = curr_assets / curr_liab
                    f11 = clamp(50 + (25 if curr_ratio > 1.5 else (-25 if curr_ratio < 1.0 else 0)))
        except Exception:
            pass

        weights = SP500_WEIGHTS if universe.lower() == "sp500" else RUSSELL_WEIGHTS

        weighted_sum = (
            (f1 * weights["f1"]) + (f2 * weights["f2"]) + (f3 * weights["f3"]) + (f4 * weights["f4"]) +
            (f5 * weights["f5"]) + (f6 * weights["f6"]) + (f7 * weights["f7"]) + (f8 * weights["f8"]) +
            (f9 * weights["f9"]) + (f10 * weights["f10"]) + (f11 * weights["f11"])
        )
        master_grade = clamp(weighted_sum)
        detected_setups = detect_patterns(hist)
        patterns_str = ", ".join(detected_setups) if detected_setups else "None"

        price = info.get("currentPrice", hist["Close"].iloc[-1] if not hist.empty else 0)
        sector_name = info.get("sector", "Equities / Bonds")
        if not sector_name or sector_name == "Unknown":
            sector_name = "Fixed Income / ETF" if ticker_symbol in ["LQD", "VCIT", "HYG", "ANGL", "SPIB", "USIG", "TLT", "BND"] else "Equities"

        return {
            "Ticker": ticker_symbol, "Universe": universe.upper(), "Name": info.get("shortName", ticker_symbol),
            "Sector": sector_name, "Price": round(price, 2) if price else 0, "NC5_Master_Grade": master_grade,
            "Triggered_Patterns": patterns_str, "Pattern_Count": len(detected_setups),
            "F1_Valuation": f1, "F2_Consensus": f2, "F5_RangePos": f5
        }
    except Exception:
        return {
            "Ticker": ticker_symbol, "Universe": universe.upper(), "Name": ticker_symbol, "Sector": "Error",
            "Price": 0.0, "NC5_Master_Grade": 50, "Triggered_Patterns": "None", "Pattern_Count": 0,
            "F1_Valuation": 50, "F2_Consensus": 50, "F5_RangePos": 50
        }

@st.cache_data(ttl=1800)
def fetch_and_score_top_crypto(limit=10):
    try:
        base = "https://api.coingecko.com/api/v3"
        stablecoins = {"usdt", "usdc", "dai", "fdusd", "usde", "tusd", "pyusd", "usdd"}
        resp = requests.get(f"{base}/coins/markets", params={
            "vs_currency": "usd", "order": "market_cap_desc", "per_page": limit + 5,
            "page": 1, "sparkline": "false", "price_change_percentage": "30d"
        }, timeout=20)
        resp.raise_for_status()
        coins = [c for c in resp.json() if c["symbol"].lower() not in stablecoins][:limit]

        results = []
        for coin in coins:
            mkt_cap = coin.get("market_cap") or 0
            fdv = coin.get("fully_diluted_valuation") or mkt_cap
            vol_24h = coin.get("total_volume") or 0
            rank = coin.get("market_cap_rank") or 100

            circ_ratio = (mkt_cap / fdv) if fdv > 0 else 0.5
            f6_fdv = clamp(circ_ratio * 100.0)
            f1_conc = clamp(95 - (rank * 0.45))
            turnover_pct = (vol_24h / mkt_cap * 100.0) if mkt_cap > 0 else 0
            f2_vol = 92 if 2.0 <= turnover_pct <= 15.0 else (70 if turnover_pct <= 40.0 else 25)

            master_grade = clamp((f1_conc * 0.4) + (f2_vol * 0.4) + (f6_fdv * 0.2))
            results.append({
                "Ticker": coin["symbol"].upper(), "Universe": "CRYPTO", "Name": coin["name"],
                "Sector": "Cryptocurrency", "Price": coin["current_price"], "NC5_Master_Grade": master_grade,
                "Triggered_Patterns": "Crypto Momentum", "Pattern_Count": 1,
                "F1_Valuation": f1_conc, "F2_Consensus": f2_vol, "F5_RangePos": f6_fdv
            })
        return pd.DataFrame(results)
    except Exception:
        return pd.DataFrame()

# ==========================================
# NAVIGATION TABS
# ==========================================
tab_chart, tab_swing, tab_top30, tab_backtest, tab_screener = st.tabs([
    "📊 Interactive Chart & Fundamentals",
    "🎯 1-Week Swing Pattern Scanner",
    "🏆 NC5 Top 30 Weekly Buys",
    "📅 NC5 Dual-System Backtester",
    "⚙️ Custom Asset Graders (NC5 Engine)"
])

# ==========================================
# TAB 1: INTERACTIVE CHART
# ==========================================
with tab_chart:
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 2, 1])
    with col_ctrl1:
        chart_ticker = st.text_input("Ticker Symbol", value="AAPL", key="chart_tick").upper().strip()
    with col_ctrl2:
        timeframe = st.selectbox(
            "Chart Timeline",
            options=["1D (5m)", "5D (15m)", "1M (1h)", "6M (1d)", "YTD (1d)", "1Y (1d)", "2Y (1wk)", "5Y (1wk)", "Max (1mo)"],
            index=5
        )
    with col_ctrl3:
        chart_style = st.selectbox("Chart Style", ["Candlestick", "Line"], index=0)

    tf_map = {
        "1D (5m)": ("1d", "5m"), "5D (15m)": ("5d", "15m"), "1M (1h)": ("1mo", "1h"),
        "6M (1d)": ("6mo", "1d"), "YTD (1d)": ("ytd", "1d"), "1Y (1d)": ("1y", "1d"),
        "2Y (1wk)": ("2y", "1wk"), "5Y (1wk)": ("5y", "1wk"), "Max (1mo)": ("max", "1mo"),
    }
    yf_period, yf_interval = tf_map[timeframe]

    if chart_ticker:
        try:
            tk = yf.Ticker(chart_ticker)
            info = tk.info or {}
            hist = tk.history(period=yf_period, interval=yf_interval)

            if not hist.empty:
                curr_price = info.get("currentPrice", hist["Close"].iloc[-1])
                prev_close = info.get("previousClose", hist["Close"].iloc[0])
                diff, pct = curr_price - prev_close, ((curr_price - prev_close) / prev_close) * 100 if prev_close else 0

                div_rate = info.get("dividendRate")
                div_yield = info.get("dividendYield")
                div_pct = (div_rate / curr_price * 100) if (div_rate and curr_price > 0) else (div_yield * 100 if div_yield else 0.0)
                if div_pct > 35.0: div_pct = 0.0

                st.subheader(f"{info.get('longName', chart_ticker)} ({chart_ticker})")
                m1, m2, m3, m4, m5, m6 = st.columns(6)
                m1.metric("Price", f"${curr_price:,.2f}", f"{diff:+.2f} ({pct:+.2f}%)")
                m2.metric("Market Cap", f"${info.get('marketCap', 0)/1e9:,.1f}B")
                m3.metric("P/E Ratio", f"{info.get('trailingPE', 'N/A')}")
                m4.metric("EPS (TTM)", f"${info.get('trailingEps', 'N/A')}")
                m5.metric("Volume", f"{info.get('volume', hist['Volume'].iloc[-1]):,.0f}")
                m6.metric("Div Yield", f"{div_pct:.2f}%")

                hist["MA20"] = hist["Close"].rolling(window=20).mean()
                hist["MA50"] = hist["Close"].rolling(window=50).mean()

                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.75, 0.25])
                if chart_style == "Candlestick":
                    fig.add_trace(go.Candlestick(x=hist.index, open=hist["Open"], high=hist["High"], low=hist["Low"], close=hist["Close"], name="Price"), row=1, col=1)
                else:
                    fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"], mode="lines", name="Close"), row=1, col=1)

                fig.add_trace(go.Scatter(x=hist.index, y=hist["MA20"], mode="lines", name="20 MA", line=dict(color="orange")), row=1, col=1)
                fig.add_trace(go.Scatter(x=hist.index, y=hist["MA50"], mode="lines", name="50 MA", line=dict(color="cyan")), row=1, col=1)
                colors = ["green" if c >= o else "red" for c, o in zip(hist["Close"], hist["Open"])]
                fig.add_trace(go.Bar(x=hist.index, y=hist["Volume"], marker_color=colors, name="Volume"), row=2, col=1)

                fig.update_layout(height=600, xaxis_rangeslider_visible=False, dragmode="drawline", title=f"{chart_ticker} Chart ({timeframe})")
                st.plotly_chart(fig, use_container_width=True, config={"editable": True, "modeBarButtonsToAdd": ["drawline", "drawrect", "eraseshape"]})
        except Exception as e:
            st.error(f"Error loading chart: {e}")

# ==========================================
# TAB 2: 1-WEEK SWING PATTERN SCANNER
# ==========================================
with tab_swing:
    st.subheader("🎯 NC5 1-Week Swing Trade & Pattern Scanner")
    col_sw1, col_sw2, col_sw3, col_sw4 = st.columns(4)
    with col_sw1:
        swing_universe = st.selectbox("Index Weighting Model", ["S&P 500", "Russell"], index=0, key="sw_univ")
    with col_sw2:
        min_grade = st.slider("Minimum NC5 Master Grade", min_value=50, max_value=95, value=70, step=5)
    with col_sw3:
        scan_limit = st.selectbox("Scan Universe Size", [25, 50, 100, 250, 500], index=1)
    with col_sw4:
        require_pattern = st.checkbox("Require Active Pattern", value=True)

    sw_univ_key = "sp500" if swing_universe == "S&P 500" else "russell"

    if st.button("Run 1-Week Swing Scan"):
        tickers_to_scan = get_sp500_tickers()[:scan_limit]
        progress_bar = st.progress(0)
        status_text = st.empty()
        scanned_results = []

        for idx, sym in enumerate(tickers_to_scan):
            status_text.text(f"Scanning {sym} ({idx + 1}/{len(tickers_to_scan)})...")
            res = score_equity_full(sym, universe=sw_univ_key)
            scanned_results.append(res)
            progress_bar.progress((idx + 1) / len(tickers_to_scan))

        status_text.empty()
        progress_bar.empty()
        df_scan = pd.DataFrame(scanned_results)
        df_filtered = df_scan[df_scan["NC5_Master_Grade"] >= min_grade]
        if require_pattern:
            df_filtered = df_filtered[df_filtered["Pattern_Count"] > 0]
        df_filtered = df_filtered.sort_values(by=["Pattern_Count", "NC5_Master_Grade"], ascending=[False, False])

        if not df_filtered.empty:
            st.success(f"Found {len(df_filtered)} qualified swing candidates!")
            st.dataframe(df_filtered[["Ticker", "Universe", "Name", "Sector", "Price", "NC5_Master_Grade", "Triggered_Patterns", "F1_Valuation", "F2_Consensus", "F5_RangePos"]], use_container_width=True, hide_index=True)
        else:
            st.warning("No stocks matched criteria in this batch.")

# ==========================================
# TAB 3: NC5 TOP 30 WEEKLY BUYS
# ==========================================
with tab_top30:
    st.subheader("🏆 NC5 Top 30 Multi-Asset Dynamic Ranking & Selection")
    top30_universe = st.radio("Equity Weighting Profile", ["S&P 500", "Russell"], horizontal=True, key="top30_univ")
    top30_univ_key = "sp500" if top30_universe == "S&P 500" else "russell"

    if st.button("Generate Top 30 Ranked Portfolio"):
        with st.spinner("Scoring candidate universe..."):
            stock_candidates = ["AAPL", "MSFT", "NVDA", "GOOGL", "JPM", "BAC", "V", "MA", "UNH", "JNJ", "LLY", "PFE", "AMZN", "TSLA", "HD", "NKE", "PG", "COST", "WMT", "KO", "XOM", "CVX", "COP", "HON", "UPS", "CAT", "LIN", "FCX", "NEE", "DUK", "AMT", "PLD"]
            bond_candidates = ["LQD", "VCIT", "HYG", "ANGL", "SPIB", "USIG", "TLT", "BND"]
            scored_equities = [score_equity_full(s, universe=top30_univ_key) for s in (stock_candidates + bond_candidates)]
            df_equities = pd.DataFrame(scored_equities)
            df_crypto = fetch_and_score_top_crypto(10)
            df_combined = pd.concat([df_equities, df_crypto], ignore_index=True).sort_values(by="NC5_Master_Grade", ascending=False)
            st.success("Top 30 Portfolio Generated!")
            st.dataframe(df_combined[["Ticker", "Universe", "Name", "Sector", "Price", "NC5_Master_Grade", "Triggered_Patterns", "F1_Valuation", "F2_Consensus", "F5_RangePos"]].head(30), use_container_width=True, hide_index=True)

# ==========================================
# TAB 4: NC5 DUAL-SYSTEM BACKTESTER
# ==========================================
with tab_backtest:
    st.subheader("📅 NC5 Dual-System Rotational & Swing Backtester")
    st.markdown("Test either the **Regular Multi-Asset Rotational System** or the **1-Week Swing Trade Pattern System** historically.")

    backtest_system = st.radio("Select Backtest System", ["Regular Multi-Asset Rotational System", "1-Week Swing Trade Pattern System"], horizontal=True)
    bt_period = st.selectbox("Backtest Period", options=["6mo", "1y", "2y", "3y", "5y"], index=1, key="bt_per")
    bt_capital = st.number_input("Starting Capital ($)", value=100000.0, step=10000.0, key="bt_cap")
    top_n = st.slider("Top Active Picks Weekly", min_value=3, max_value=15, value=8, key="bt_topn")

    if st.button("Run Backtest Simulation"):
        with st.spinner(f"Running simulation for {backtest_system}..."):
            try:
                universe_all = [
                    "AAPL", "MSFT", "NVDA", "JPM", "BAC", "UNH", "JNJ", "AMZN", "TSLA", "PG",
                    "COST", "XOM", "CVX", "HON", "UPS", "LIN", "NEE", "GOOGL", "META", "AMT",
                    "LQD", "VCIT", "HYG", "SPY"
                ]
                raw_data = yf.download(universe_all, period=bt_period, interval="1wk", progress=False)

                if isinstance(raw_data.columns, pd.MultiIndex):
                    data = raw_data['Close'] if 'Close' in raw_data.columns.levels[0] else raw_data.xs('Close', level=0, axis=1, drop_level=True)
                else:
                    data = raw_data

                data = data.dropna(how="all")
                if data.empty or "SPY" not in data.columns:
                    st.error("Insufficient historical data retrieved.")
                else:
                    spy_prices = data["SPY"]
                    asset_prices = data.drop(columns=["SPY"], errors="ignore")
                    weekly_returns = asset_prices.pct_change().shift(-1).dropna()

                    portfolio_values = [bt_capital]
                    dates = [asset_prices.index[0]]
                    curr_cap = bt_capital

                    for i in range(len(asset_prices) - 1):
                        date_curr = asset_prices.index[i]
                        date_next = asset_prices.index[i+1]
                        window_slice = asset_prices.iloc[:i+1]

                        if backtest_system == "Regular Multi-Asset Rotational System":
                            # Pure momentum / trend ranking
                            past_perf = window_slice.tail(4).pct_change().sum()
                            valid_perf = past_perf.dropna()
                            if len(valid_perf) >= top_n:
                                top_picks = valid_perf.nlargest(top_n).index.tolist()
                                if date_curr in weekly_returns.index:
                                    next_week_ret = weekly_returns.loc[date_curr, top_picks].mean()
                                    curr_cap *= (1 + (next_week_ret if not pd.isna(next_week_ret) else 0.0))
                        else:
                            # Swing system: check for short-term pullback / volatility squeeze criteria in historical slice
                            qualified_picks = []
                            for col in asset_prices.columns:
                                stock_hist = yf.Ticker(col).history(period="3mo")
                                if not stock_hist.empty and len(stock_hist) > 50:
                                    p_setups = detect_patterns(stock_hist)
                                    if len(p_setups) > 0:
                                        qualified_picks.append(col)
                            
                            if qualified_picks and date_curr in weekly_returns.index:
                                valid_picks = [p for p in qualified_picks if p in weekly_returns.columns]
                                if valid_picks:
                                    next_week_ret = weekly_returns.loc[date_curr, valid_picks].mean()
                                    curr_cap *= (1 + (next_week_ret if not pd.isna(next_week_ret) else 0.0))

                        portfolio_values.append(curr_cap)
                        dates.append(date_next)

                    spy_norm = (spy_prices / spy_prices.iloc[0]) * bt_capital
                    final_val = portfolio_values[-1]
                    net_pct = ((final_val - bt_capital) / bt_capital) * 100
                    spy_final = spy_norm.iloc[-1]
                    spy_pct = ((spy_final - bt_capital) / bt_capital) * 100

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Strategy Ending Value", f"${final_val:,.2f}", f"{net_pct:+.2f}%")
                    m2.metric("S&P 500 Benchmark", f"${spy_final:,.2f}", f"{spy_pct:+.2f}%")
                    m3.metric("Alpha Generated", f"{net_pct - spy_pct:+.2f}%")

                    fig_bt = go.Figure()
                    fig_bt.add_trace(go.Scatter(x=dates, y=portfolio_values, mode="lines", name=backtest_system, line=dict(color="green", width=2)))
                    fig_bt.add_trace(go.Scatter(x=spy_norm.index, y=spy_norm, mode="lines", name="S&P 500 (SPY)", line=dict(color="gray", dash="dash")))
                    fig_bt.update_layout(title=f"Backtest Comparison: {backtest_system}", xaxis_title="Date", yaxis_title="Portfolio Value ($)", height=500)
                    st.plotly_chart(fig_bt, use_container_width=True)
            except Exception as e:
                st.error(f"Error running backtest: {e}")

# ==========================================
# TAB 5: CUSTOM ASSET GRADERS
# ==========================================
with tab_screener:
    st.subheader("⚙️ NC5 Custom Asset Grader Tool")
    col_s1, col_s2 = st.columns([3, 1])
    with col_s1:
        custom_input = st.text_input("Enter Ticker Symbols (comma-separated)", value="AAPL, LQD, BTC-USD, MSFT").upper()
    with col_s2:
        selected_universe = st.selectbox("Grading Universe", options=["S&P 500", "Russell"], index=0)

    universe_key = "sp500" if selected_universe == "S&P 500" else "russell"

    if st.button("Run NC5 Custom Grader"):
        symbols = [s.strip() for s in custom_input.split(",") if s.strip()]
        custom_rows = []
        for sym in symbols:
            if "USD" in sym or sym in ["BTC", "ETH"]:
                tk_sym = sym if "-USD" in sym else f"{sym}-USD"
                try:
                    tk = yf.Ticker(tk_sym)
                    hist = tk.history(period="1mo")
                    price = hist["Close"].iloc[-1] if not hist.empty else 0
                    custom_rows.append({
                        "Ticker": sym.replace("-USD", ""), "Universe": universe_key.upper(), "Name": sym, "Sector": "Cryptocurrency",
                        "Price": round(price, 2), "NC5_Master_Grade": 75, "Triggered_Patterns": "Crypto Momentum",
                        "F1_Valuation": 75, "F2_Consensus": 75, "F5_RangePos": 75
                    })
                except Exception:
                    custom_rows.append({
                        "Ticker": sym, "Universe": universe_key.upper(), "Name": sym, "Sector": "Cryptocurrency",
                        "Price": 0.0, "NC5_Master_Grade": 50, "Triggered_Patterns": "None",
                        "F1_Valuation": 50, "F2_Consensus": 50, "F5_RangePos": 50
                    })
            else:
                custom_rows.append(score_equity_full(sym, universe=universe_key))

        if custom_rows:
            df_custom = pd.DataFrame(custom_rows).sort_values(by="NC5_Master_Grade", ascending=False)
            st.dataframe(df_custom[["Ticker", "Universe", "Name", "Sector", "Price", "NC5_Master_Grade", "Triggered_Patterns", "F1_Valuation", "F2_Consensus", "F5_RangePos"]], use_container_width=True, hide_index=True)
