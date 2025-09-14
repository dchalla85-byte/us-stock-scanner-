import io
import time
import math
import json
import gzip
import queue
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------
# Config – tweak to your preference
# ---------------------------------------------
MAX_TICKERS = None          # None = scan all from NASDAQ/NYSE/AMEX
THREADS = 12                # parallel workers
CHUNK_SIZE = 50             # yfinance batch size for price/quotes
LIQ_MIN_ADV_USD = 50_000_000  # avg $ volume safety screen
FALLBACK_PRICE_VOL_RULE = {"min_price": 10.0, "min_avg_shares": 1_000_000}
OUTPUT_ALL = "us_fundamentals_scanner.csv"
OUTPUT_TOP = "us_fundamentals_top.csv"

# Composite score weights
W_ROE = 0.50
W_YLD = 0.30
W_PE  = 0.20  # lower P/E is better (we invert percentile)

# ---------------------------------------------
# Helper: get all US tickers from NASDAQ Trader
# ---------------------------------------------
def get_all_us_tickers():
    sources = [
        "https://ftp.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        "https://ftp.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
    ]
    frames = []
    for url in sources:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        # Files are pipe-delimited with header/footer lines
        lines = [ln for ln in r.text.splitlines() if "File Creation Time" not in ln and "Symbol|Security Name" not in ln and "Test Issue" not in ln and "NASDAQ Symbol" not in ln]
        rows = [ln.split("|") for ln in lines if "|" in ln]
        if "nasdaqlisted" in url:
            # Symbol,Security Name,Market Category,Test Issue,Financial Status,Round Lot Size,ETF,NextShares
            cols = ["Symbol","SecurityName","MarketCategory","TestIssue","FinancialStatus","RoundLot","ETF","NextShares"]
        else:
            # ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol
            cols = ["Symbol","SecurityName","Exchange","Cqs","ETF","RoundLot","TestIssue","NasdaqSymbol"]
        df = pd.DataFrame(rows, columns=cols)
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df[~all_df["Symbol"].str.contains(r"\^|\/|=", regex=True, na=False)]
    all_df = all_df[all_df["Symbol"].str.len() > 0]
    # Exclude test issues and preferreds/units where obvious
    if "TestIssue" in all_df.columns:
        all_df = all_df[all_df["TestIssue"].fillna("N") != "Y"]
    all_df = all_df[~all_df["Symbol"].str.contains(r"\.(A|B|C|PR|WS|U|W|RT)$", regex=True)]
    tickers = sorted(all_df["Symbol"].unique().tolist())
    return tickers

# ---------------------------------------------
# Finance fetchers
# ---------------------------------------------
def fetch_batch_fast_info(tickers):
    """
    Pulls fast info (price, trailing PE, 52w…) in batch using yfinance Tickers.
    """
    tk = yf.Tickers(" ".join(tickers))
    out = {}
    for sym, obj in tk.tickers.items():
        try:
            fi = obj.fast_info
            # averageDollarVolume: use 30-day average volume * last price (approx)
            adv = None
            try:
                adv = (fi.get("last_price") or fi.get("lastPrice") or np.nan) * (fi.get("ten_day_average_volume") or fi.get("last_volume") or np.nan)
            except Exception:
                adv = np.nan
            out[sym] = {
                "price": fi.get("last_price") or fi.get("lastPrice"),
                "trailing_pe": fi.get("trailing_pe"),
                "market_cap": fi.get("market_cap"),
                "avg_dollar_volume": adv,
            }
        except Exception:
            out[sym] = {"price": np.nan, "trailing_pe": np.nan, "market_cap": np.nan, "avg_dollar_volume": np.nan}
    return out

def compute_dividend_yield_ttm(ticker_obj, price):
    """
    Dividend yield (TTM): sum dividends last 365 days / current price.
    """
    try:
        div = ticker_obj.dividends
        if div is None or div.empty or not price or price <= 0:
            return np.nan
        # last 365 days
        last = div[div.index >= (pd.Timestamp.utcnow() - pd.Timedelta(days=365))]
        ttm = last.sum()
        return float(ttm) / float(price) if price and price > 0 else np.nan
    except Exception:
        return np.nan

def compute_roe_ttm(ticker_obj):
    """
    ROE (TTM) ~ (sum of last 4 quarters Net Income) / (average Stockholders' Equity over last 2-4 quarters)
    """
    try:
        is_q = ticker_obj.quarterly_financials  # (deprecated name ‘quarterly_financials’ in older yfinance)
        if is_q is None or is_q.empty:
            is_q = ticker_obj.quarterly_income_stmt
        bs_q = ticker_obj.quarterly_balance_sheet

        # Net Income TTM
        ni_series = None
        for cand in ["Net Income", "NetIncome", "Net Income Common Stockholders", "Net Income Applicable To Common Shares"]:
            if is_q is not None and cand in is_q.index:
                ni_series = is_q.loc[cand]
                break
        if ni_series is None or ni_series.empty:
            return np.nan

        ni_ttm = pd.to_numeric(ni_series.iloc[:4], errors="coerce").sum()

        # Average equity (use Total Stockholders' Equity or Total Equity)
        eq_series = None
        for cand in ["Total Stockholder Equity", "Total Stockholders Equity", "Stockholders Equity", "Total Equity Gross Minority Interest", "Total equity"]:
            if bs_q is not None and cand in bs_q.index:
                eq_series = pd.to_numeric(bs_q.loc[cand].iloc[:4], errors="coerce")
                break
        if eq_series is None or eq_series.empty:
            return np.nan

        # average of last 2 available quarters (or up to 4 if present)
        n = min(4, eq_series.shape[0])
        if n == 0:
            return np.nan
        avg_eq = float(eq_series.iloc[:n].mean())
        if not avg_eq or math.isclose(avg_eq, 0.0):
            return np.nan

        roe = float(ni_ttm) / avg_eq
        return roe
    except Exception:
        return np.nan

def fetch_fundamentals_for_symbols(symbols):
    """
    Fetch sector, price, P/E, dividend yield (TTM), ROE (TTM) per symbol.
    """
    results = []
    # Fast batch for price/pe/adv
    for i in tqdm(range(0, len(symbols), CHUNK_SIZE), desc="Price/PE batches"):
        batch = symbols[i:i+CHUNK_SIZE]
        fi = fetch_batch_fast_info(batch)
        # per-ticker detail for sector, dividends, ROE
        with ThreadPoolExecutor(max_workers=THREADS) as pool:
            futs = {}
            for sym in batch:
                t = yf.Ticker(sym)
                futs[pool.submit(gather_single, sym, t, fi.get(sym, {}))] = sym
            for fut in as_completed(futs):
                rec = fut.result()
                results.append(rec)
    return pd.DataFrame(results)

def gather_single(sym, t, fi_row):
    price = fi_row.get("price")
    pe = fi_row.get("trailing_pe")
    mcap = fi_row.get("market_cap")
    adv = fi_row.get("avg_dollar_volume")

    # sector/name via info() (can be slow; try fast first)
    sector = np.nan
    short_name = np.nan
    try:
        inf = t.get_info()
        sector = inf.get("sector")
        short_name = inf.get("shortName") or inf.get("longName")
    except Exception:
        pass

    yld = compute_dividend_yield_ttm(t, price)
    roe = compute_roe_ttm(t)

    return {
        "ticker": sym,
        "name": short_name,
        "sector": sector,
        "price": price,
        "trailing_pe": pe,
        "dividend_yield_ttm": yld,
        "roe_ttm": roe,
        "market_cap": mcap,
        "avg_dollar_volume": adv
    }

# ---------------------------------------------
# Ranking logic (sector-aware percentiles)
# ---------------------------------------------
def winsorize(s, lower=0.01, upper=0.99):
    ql, qu = s.quantile(lower), s.quantile(upper)
    return s.clip(ql, qu)

def sector_percentiles(df):
    df = df.copy()
    # Prepare clean metrics
    df["roe_clean"] = winsorize(df["roe_ttm"].replace([np.inf, -np.inf], np.nan))
    df["yld_clean"] = winsorize(df["dividend_yield_ttm"].replace([np.inf, -np.inf], np.nan))
    df["pe_clean"]  = winsorize(df["trailing_pe"].replace([np.inf, -np.inf], np.nan))

    # Percentile ranks within sector (higher better for ROE & Yield; lower better for P/E)
    df["roe_pct"] = df.groupby("sector")["roe_clean"].rank(pct=True, method="average")
    df["yld_pct"] = df.groupby("sector")["yld_clean"].rank(pct=True, method="average")
    df["pe_pct"]  = df.groupby("sector")["pe_clean"].rank(pct=True, method="average")

    # Invert PE (lower PE => higher score)
    inv_pe = 1.0 - df["pe_pct"]
    df["composite_score"] = (W_ROE * df["roe_pct"].fillna(0) +
                             W_YLD * df["yld_pct"].fillna(0) +
                             W_PE  * inv_pe.fillna(0))
    return df

def liquidity_screen(df):
    # Primary: average dollar volume >= threshold
    liq_ok = df["avg_dollar_volume"] >= LIQ_MIN_ADV_USD
    # Fallback rule
    fallback_ok = (df["price"] >= FALLBACK_PRICE_VOL_RULE["min_price"]) & \
                  ((df["avg_dollar_volume"].isna()) | (df["avg_dollar_volume"]==0)) & \
                  (df["market_cap"].notna())  # basic sanity
    return df[liq_ok | fallback_ok]

# ---------------------------------------------
# Main
# ---------------------------------------------
def main():
    print("Fetching U.S. listed tickers…")
    tickers = get_all_us_tickers()
    if MAX_TICKERS:
        tickers = tickers[:MAX_TICKERS]
    print(f"Total symbols: {len(tickers)}")

    print("Collecting fundamentals (this can take a while; running in parallel)…")
    df = fetch_fundamentals_for_symbols(tickers)

    # Basic cleaning
    df = df.dropna(subset=["sector"])
    # Liquidity filter
    df = liquidity_screen(df)

    # Compute sector-aware percentiles and composite
    df = sector_percentiles(df)

    # Friendly columns
    out = df[[
        "ticker","name","sector","price","market_cap","avg_dollar_volume",
        "roe_ttm","dividend_yield_ttm","trailing_pe",
        "roe_pct","yld_pct","pe_pct","composite_score"
    ]].sort_values("composite_score", ascending=False)

    # Save full table
    out.to_csv(OUTPUT_ALL, index=False)

    # Top ideas (per sector top N and overall top 200)
    top_per_sector = (
        out.sort_values("composite_score", ascending=False)
           .groupby("sector")
           .head(15)
    )
    top_overall = out.head(200)

    top = pd.concat([
        top_overall.assign(bucket="Top 200 Overall"),
        top_per_sector.assign(bucket="Top 15 per Sector")
    ], ignore_index=True).sort_values(["bucket","composite_score"], ascending=[True, False])

    top.to_csv(OUTPUT_TOP, index=False)

    print(f"\nSaved:\n - {OUTPUT_ALL}\n - {OUTPUT_TOP}")
    print("\nTip: Open in Excel/Sheets; filter by sector, then sort by Composite Score. "
          "You can also tighten liquidity or change weights at the top of the file.")

if __name__ == "__main__":
    main()
