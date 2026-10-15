# MODULE 1 - DATA: download and cache daily prices
import os, time, pandas as pd, yfinance as yf
from config import TICKERS
F = "data/prices.csv"

def download(force=False):
    os.makedirs("data", exist_ok=True)
    if os.path.exists(F) and not force:
        return pd.read_csv(F, index_col=0, parse_dates=True)
    df = yf.download(TICKERS, start="2017-01-01", auto_adjust=True, progress=False)["Close"]
    df = df[TICKERS].dropna()
    if df.empty:
        raise RuntimeError("No price data downloaded. Check your internet connection.")
    df.to_csv(F)
    return df

def latest(max_age_h=6):
    """Cached prices, refreshed when older than max_age_h hours."""
    if not os.path.exists(F) or time.time() - os.path.getmtime(F) > max_age_h * 3600:
        try:
            return download(force=True)
        except Exception:
            if not os.path.exists(F):
                raise
    return download()

if __name__ == "__main__":
    print(download(force=True).tail())
