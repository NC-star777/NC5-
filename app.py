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
st.write("Live charting, multi-factor grading (F1, F2, F5), dynamic multi-asset ranking, and weekly rotational backtesting.")

# ==========================================
# HELPER FUNCTIONS & NC5 SCORING ENGINES
# ==========================================
def clamp(val, low=1, high=100):
    return int(max(low, min(high, round(val))))

@st.cache_data(ttl=3600)
def score_equity_full(ticker_symbol):
    try:
        tk = yf.Ticker(ticker_symbol)
        info = tk.info or {}
        hist = tk.history(period="1y")

        # Factor 1: Growth-Adjusted Valuation (30%)
        fwd_pe = info.get("forwardPE")
        growth_dec = info.get("earningsGrowth")
        if fwd_pe is None or fwd_pe <= 0:
            f1 = 50
        else:
            growth_pct = max(0.0, (growth_dec or 0.0) * 100.0)
            justified_pe = 10.0 + (2.0 * growth_pct)
            val_ratio = min(2.0, justified_pe / fwd_pe)
            f1 = clamp(50.0 * val_ratio)

        # Factor 2: Analyst Consensus (15%)
        rec_mean = info.get("recommendationMean")
        f2 = 50 if rec_mean is None else clamp(100.0 - ((rec_mean - 1.0) * 25.0))

        # Supporting Factors (F3 - F11)
        rec_key = str(info.get("recommendationKey", "")).lower().strip()
        tech_map = {"strong_buy": 95, "buy": 75, "hold": 50, "sell": 25, "strong_sell": 5}
        f3 = tech_map.get(rec_key, 50)

        summary = str(info.get("longBusinessSummary", "")).lower()
        risk_flags = ["class action", "sec investigation", "department of justice"]
        f4 = 25 if any(flag in summary for flag in risk_flags) else 85

        # Factor 5: 170-Day Range Position (10%)
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

        master_grade = clamp(
            (f1 * 0.30) + (f2 * 0.15) + (f3 * 0.04) + (f4 * 0.04) +
            (f5 * 0.10) + (f6 * 0.04) + (f7 * 0.04) + (f8 * 0.04) +
            (f9 * 0.10) + (f10 * 0.10) + (f11 * 0.05)
        )

        price = info.get("currentPrice", hist["Close"].iloc[-1] if not hist.empty else 0)
        sector_name = info.get("sector", "Equities / Bonds")
        if not sector_name or sector_name == "Unknown":
            sector_name = "Fixed Income / ETF" if ticker_symbol in ["LQD", "VCIT", "HYG", "ANGL", "SPIB", "USIG", "TLT", "BND"] else "Equities"

        return {
            "Ticker": ticker_symbol,
            "Name": info.get("shortName", ticker_symbol),
            "Sector": sector_name,
            "Price": round(price, 2) if price else 0,
            "NC5_Master_Grade": master_grade,
            "F1_Valuation": f1,
            "F2_Consensus": f2,
            "F5_RangePos": f5
        }
    except Exception:
        return {
            "Ticker": ticker_symbol, "Name": ticker_symbol, "Sector": "Error",
            "Price": 0.0, "NC5_Master_Grade": 50, "F1_Valuation": 50, "F2_Consensus": 50, "F5_RangePos": 50
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
                "Ticker": coin["symbol"].upper(),
                "Name": coin["name"],
                "Sector": "Cryptocurrency",
                "Price": coin["current_price"],
                "NC5_Master_Grade": master_grade,
                "F1_Valuation": f1_conc,
                "F2_Consensus": f2_vol,
                "F5_RangePos": f6_fdv
            })
        return pd.DataFrame(results)
    except Exception:
        return pd.DataFrame()

# ==========================================
# NAVIGATION TABS
# ==========================================
tab_chart, tab_top30, tab_backtest, tab_screener = st.tabs([
    "📊 Interactive Chart & Fundamentals",
    "🏆 NC5 Top 30 Weekly Buys",
    "📅 NC5 Weekly Rotational Backtester",
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
                if div_rate and curr_price and curr_price > 0:
                    div_pct = (div_rate / curr_price) * 100
                elif div_yield is not None:
                    div_pct = (div_yield * 100) if div_yield <= 1.0 else div_yield
                else:
                    div_pct = 0.0
                
                if div_pct > 35.0:
                    div_pct = 0.0

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
# TAB 2: NC5 TOP 30 WEEKLY BUYS
# ==========================================
with tab_top30:
    st.subheader("🏆 NC5 Top 30 Multi-Asset Dynamic Ranking & Selection")
    st.markdown("Scans a broad universe across all 11 sectors, corporate bonds, and crypto, then automatically selects the highest-ranking assets.")

    if st.button("Generate Top 30 Ranked Portfolio"):
        with st.spinner("Scoring candidate universe across sectors, bonds, and crypto..."):
            stock_candidates = [
                "AAPL", "MSFT", "NVDA", "GOOGL", "JPM", "BAC", "V", "MA",
                "UNH", "JNJ", "LLY", "PFE", "AMZN", "TSLA", "HD", "NKE",
                "PG", "COST", "WMT", "KO", "XOM", "CVX", "COP", "HON",
                "UPS", "CAT", "LIN", "FCX", "NEE", "DUK", "AMT", "PLD"
            ]
            bond_candidates = ["LQD", "VCIT", "HYG", "ANGL", "SPIB", "USIG", "TLT", "BND"]
            
            scored_equities = [score_equity_full(s) for s in (stock_candidates + bond_candidates)]
            df_equities = pd.DataFrame(scored_equities)
            df_crypto = fetch_and_score_top_crypto(10)
            
            df_combined = pd.concat([df_equities, df_crypto], ignore_index=True).sort_values(by="NC5_Master_Grade", ascending=False)
            
            st.success("NC5 Top 30 Highest-Ranking Portfolio Generated Successfully!")
            st.dataframe(
                df_combined[["Ticker", "Name", "Sector", "Price", "NC5_Master_Grade", "F1_Valuation", "F2_Consensus", "F5_RangePos"]].head(30),
                use_container_width=True,
                hide_index=True
            )

# ==========================================
# TAB 3: NC5 WEEKLY ROTATIONAL BACKTESTER
# ==========================================
with tab_backtest:
    st.subheader("📅 NC5 Weekly Rotational Backtester")
    st.markdown("Simulates weekly buying and rotation. You can test relative time periods or isolate specific calendar months (like July 2025).")

    bt_mode = st.radio("Backtest Mode", ["Relative Period (1y, 2y, etc.)", "Specific Month (e.g., July 2025)"], horizontal=True)

    if bt_mode == "Specific Month (e.g., July 2025)":
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            sel_bt_year = st.selectbox("Year", [2024, 2025, 2026], index=1)
        with col_m2:
            months_list = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
            sel_bt_month = st.selectbox("Month", months_list, index=6) # July is index 6
            month_num = months_list.index(sel_bt_month) + 1
        with col_m3:
            bt_capital = st.number_input("Starting Capital ($)", value=100000.0, step=10000.0, key="bt_cap_m")
        top_n = st.slider("Top Picks Rotated Weekly", min_value=3, max_value=15, value=10, key="bt_topn_m")
    else:
        col_b1, col_b2, col_b3 = st.columns(3)
        with col_b1:
            bt_period = st.selectbox("Backtest Period", options=["1y", "2y", "3y", "5y"], index=1, key="bt_per")
        with col_b2:
            bt_capital = st.number_input("Starting Capital ($)", value=100000.0, step=10000.0, key="bt_cap")
        with col_b3:
            top_n = st.slider("Top Picks Rotated Weekly", min_value=3, max_value=15, value=10, key="bt_topn")

    if st.button("Run Weekly Rotation Backtest"):
        with st.spinner("Simulating weekly rotations..."):
            try:
                universe_all = [
                    "AAPL", "MSFT", "NVDA", "JPM", "BAC", "UNH", "JNJ", "AMZN", "TSLA", "PG",
                    "COST", "XOM", "CVX", "HON", "UPS", "LIN", "NEE", "GOOGL", "META", "AMT",
                    "LQD", "VCIT", "HYG", "ANGL", "SPIB", "USIG", "TLT", "BTC-USD", "ETH-USD"
                ]
                
                if bt_mode == "Specific Month (e.g., July 2025)":
                    # Download data covering a buffer around that specific month
                    start_date = f"{sel_bt_year}-{month_num:02d}-01"
                    # End date: handle December transition or next month
                    if month_num == 12:
                        end_date = f"{sel_bt_year+1}-01-31"
                    else:
                        end_date = f"{sel_bt_year}-{month_num+1:02d}-28"
                    
                    data = yf.download(universe_all + ["SPY"], start=start_date, end=end_date, interval="1wk", progress=False)["Close"].dropna(how="all")
                else:
                    data = yf.download(universe_all + ["SPY"], period=bt_period, interval="1wk", progress=False)["Close"].dropna()
                
                if data.empty:
                    st.error("Insufficient historical data for the selected period/month.")
                else:
                    spy_prices = data["SPY"]
                    asset_prices = data.drop(columns=["SPY"], errors="ignore")
                    
                    if bt_mode == "Specific Month (e.g., July 2025)":
                        # Filter strictly for the selected month and year
                        asset_prices = asset_prices[(asset_prices.index.year == sel_bt_year) & (asset_prices.index.month == month_num)]
                        spy_prices = spy_prices[(spy_prices.index.year == sel_bt_year) & (spy_prices.index.month == month_num)]

                    if len(asset_prices) < 2:
                        st.error(f"Not enough weekly price points found for {sel_bt_month} {sel_bt_year}.")
                    else:
                        weekly_returns = asset_prices.pct_change().shift(-1).dropna()
                        
                        portfolio_values = [bt_capital]
                        dates = [asset_prices.index[0]]
                        curr_cap = bt_capital
                        
                        for i in range(len(asset_prices) - 1):
                            date_curr = asset_prices.index[i]
                            date_next = asset_prices.index[i+1]
                            
                            past_perf = asset_prices.iloc[:i+1].tail(4).pct_change().sum()
                            if len(past_perf.dropna()) >= top_n:
                                top_picks = past_perf.nlargest(top_n).index.tolist()
                                next_week_ret = weekly_returns.loc[date_curr, top_picks].mean()
                                if pd.isna(next_week_ret):
                                    next_week_ret = 0.0
                                curr_cap *= (1 + next_week_ret)
                            
                            portfolio_values.append(curr_cap)
                            dates.append(date_next)
                            
                        spy_norm = (spy_prices / spy_prices.iloc[0]) * bt_capital
                        
                        final_val = portfolio_values[-1]
                        net_profit = final_val - bt_capital
                        net_pct = (net_profit / bt_capital) * 100
                        
                        spy_final = spy_norm.iloc[-1]
                        spy_profit = spy_final - bt_capital
                        spy_pct = (spy_profit / bt_capital) * 100
                        
                        title_label = f"NC5 Rotation ({sel_bt_month} {sel_bt_year})" if bt_mode.startswith("Specific") else "NC5 Weekly Multi-Asset Rotation"
                        
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Strategy Ending Value", f"${final_val:,.2f}", f"{net_pct:+.2f}%")
                        m2.metric("S&P 500 (SPY) Benchmark", f"${spy_final:,.2f}", f"{spy_pct:+.2f}%")
                        m3.metric("Alpha Generated", f"{net_pct - spy_pct:+.2f}%")
                        
                        fig_bt = go.Figure()
                        fig_bt.add_trace(go.Scatter(x=dates, y=portfolio_values, mode="lines", name=title_label, line=dict(color="green", width=2)))
                        fig_bt.add_trace(go.Scatter(x=spy_norm.index, y=spy_norm, mode="lines", name="S&P 500 (SPY)", line=dict(color="gray", dash="dash")))
                        
                        fig_bt.update_layout(
                            title=f"Performance for {sel_bt_month} {sel_bt_year}" if bt_mode.startswith("Specific") else "Weekly Rotation vs. S&P 500",
                            xaxis_title="Date",
                            yaxis_title="Portfolio Value ($)",
                            height=500
                        )
                        st.plotly_chart(fig_bt, use_container_width=True)
            except Exception as e:
                st.error(f"Error running backtest: {e}")

# ==========================================
# TAB 4: CUSTOM ASSET GRADERS (NC5 ENGINE)
# ==========================================
with tab_screener:
    st.subheader("⚙️ NC5 Custom Asset Grader Tool")
    custom_input = st.text_input("Enter Ticker Symbols (comma-separated)", value="AAPL, LQD, BTC-USD, MSFT").upper()
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
                        "Ticker": sym.replace("-USD", ""), "Name": sym, "Sector": "Cryptocurrency",
                        "Price": round(price, 2), "NC5_Master_Grade": 75, "F1_Valuation": 75, "F2_Consensus": 75, "F5_RangePos": 75
                    })
                except Exception:
                    custom_rows.append({
                        "Ticker": sym, "Name": sym, "Sector": "Cryptocurrency",
                        "Price": 0.0, "NC5_Master_Grade": 50, "F1_Valuation": 50, "F2_Consensus": 50, "F5_RangePos": 50
                    })
            else:
                custom_rows.append(score_equity_full(sym))
                
        if custom_rows:
            df_custom = pd.DataFrame(custom_rows).sort_values(by="NC5_Master_Grade", ascending=False)
            st.dataframe(
                df_custom[["Ticker", "Name", "Sector", "Price", "NC5_Master_Grade", "F1_Valuation", "F2_Consensus", "F5_RangePos"]],
                use_container_width=True,
                hide_index=True
            )
