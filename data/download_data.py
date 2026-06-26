"""
Download daily closing prices for Gold (GC=F) and Bitcoin (BTC-USD)
from Yahoo Finance and save as CSV files in the data/ directory.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import yfinance as yf


START = "2015-01-01"
END = "2021-09-10"
TICKERS = {
    "GC=F": "GCF_prices.csv",
    "BTC-USD": "BTC_USD_prices.csv",
}


def download_prices(ticker: str, start: str, end: str) -> pd.Series:
    """Download daily closing prices from Yahoo Finance.

    Args:
        ticker: Yahoo Finance ticker symbol.
        start: Start date 'YYYY-MM-DD'.
        end: End date 'YYYY-MM-DD'.

    Returns:
        pd.Series of closing prices with DatetimeIndex.
    """
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"No data returned for {ticker}")
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.squeeze()
    close = close.sort_index().ffill().dropna()
    close.name = ticker
    return close


def save_prices(series: pd.Series, filepath: str) -> None:
    """Save price series to CSV.

    Args:
        series: pd.Series to save.
        filepath: Target CSV path.
    """
    series.to_csv(filepath)
    print(f"Saved {len(series)} rows to {filepath}")


def load_or_download(
    ticker: str, filepath: str, start: str, end: str
) -> pd.Series:
    """Load prices from CSV if available, otherwise download.

    Args:
        ticker: Yahoo Finance ticker symbol.
        filepath: Local CSV path.
        start: Start date.
        end: End date.

    Returns:
        pd.Series of closing prices.
    """
    if os.path.exists(filepath):
        series = pd.read_csv(filepath, index_col=0, parse_dates=True).squeeze()
        series.name = ticker
        print(f"Loaded {len(series)} rows from {filepath}")
        return series
    series = download_prices(ticker, start, end)
    save_prices(series, filepath)
    return series


if __name__ == "__main__":
    data_dir = os.path.dirname(os.path.abspath(__file__))
    for ticker, fname in TICKERS.items():
        fpath = os.path.join(data_dir, fname)
        series = load_or_download(ticker, fpath, START, END)
        print(f"{ticker}: {len(series)} observations from {series.index[0].date()} "
              f"to {series.index[-1].date()}")
