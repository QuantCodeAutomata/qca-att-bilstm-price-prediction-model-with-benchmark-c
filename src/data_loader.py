"""
Data loading and preprocessing for gold and Bitcoin price prediction.
Handles yfinance downloads, M_i computation, min-max normalization,
sliding window construction, and temporal train/test splitting.
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import warnings

warnings.filterwarnings("ignore")


def get_prices(
    ticker: str,
    start: str = "2015-01-01",
    end: str = "2021-09-10",
) -> pd.Series:
    """Download or load daily closing prices.

    Checks data/ directory for a cached CSV before downloading.
    Forward-fills missing values and drops any remaining NaN rows.

    Args:
        ticker: Yahoo Finance ticker symbol (e.g. 'GC=F', 'BTC-USD').
        start: Start date string 'YYYY-MM-DD'.
        end: End date string 'YYYY-MM-DD'.

    Returns:
        pd.Series of closing prices with DatetimeIndex.
    """
    safe_name = ticker.replace("=", "").replace("-", "_")
    filepath = os.path.join("data", f"{safe_name}_prices.csv")

    if os.path.exists(filepath):
        series = pd.read_csv(filepath, index_col=0, parse_dates=True).squeeze()
        series.name = ticker
        return series

    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(f"No data returned for ticker {ticker}")

    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.squeeze()
    close = close.sort_index()
    close = close.ffill().dropna()
    close.name = ticker

    os.makedirs("data", exist_ok=True)
    close.to_csv(filepath)
    return close


def compute_M_series(prices: pd.Series) -> pd.Series:
    """Compute daily rise/fall degree M_i = (p_i - p_{i-1}) / p_i.

    Uses the current price p_i as the denominator (NOT p_{i-1}),
    matching the paper's definition exactly.
    M_i > 0 indicates a price rise; M_i < 0 indicates a price fall.

    Args:
        prices: pd.Series of daily closing prices.

    Returns:
        pd.Series of M_i values (first observation dropped due to differencing).
    """
    M = (prices - prices.shift(1)) / prices
    return M.dropna()


def normalize_prices(
    prices: pd.Series,
    train_frac: float = 0.7,
) -> tuple:
    """Min-max normalise prices using ONLY training set statistics.

    Fits MinMaxScaler on the first `train_frac` of the price series,
    then transforms the entire series.

    Args:
        prices: pd.Series of raw closing prices.
        train_frac: Fraction used for the training set (default 0.7).

    Returns:
        Tuple of (prices_normalized: np.ndarray, scaler: MinMaxScaler).
    """
    arr = prices.values.reshape(-1, 1).astype(float)
    split = int(len(arr) * train_frac)
    scaler = MinMaxScaler()
    scaler.fit(arr[:split])
    prices_norm = scaler.transform(arr).flatten()
    return prices_norm, scaler


def create_sliding_windows(
    prices_norm: np.ndarray,
    window: int = 100,
) -> tuple:
    """Construct sliding-window input/output pairs.

    Each input window contains `window` consecutive normalised prices;
    the target is the price immediately following the window.

    Args:
        prices_norm: 1-D array of normalised prices.
        window: Number of timesteps per input window (default 100).

    Returns:
        Tuple of (X: np.ndarray shape (N-window, window, 1),
                  y: np.ndarray shape (N-window,)).
    """
    X, y = [], []
    for i in range(len(prices_norm) - window):
        X.append(prices_norm[i : i + window])
        y.append(prices_norm[i + window])
    X_arr = np.array(X, dtype=float).reshape(-1, window, 1)
    y_arr = np.array(y, dtype=float)
    return X_arr, y_arr


def train_test_split_temporal(
    X: np.ndarray,
    y: np.ndarray,
    train_frac: float = 0.7,
    shuffle_train: bool = True,
    seed: int = 42,
) -> tuple:
    """Strict temporal 7:3 split with optional training shuffle.

    Splits windows chronologically: first `train_frac` → train,
    remainder → test. Only the training set is shuffled (window order,
    not timesteps within windows).

    Args:
        X: Input windows array of shape (N, window, 1).
        y: Target array of shape (N,).
        train_frac: Fraction for training (default 0.7).
        shuffle_train: Whether to shuffle training windows (default True).
        seed: Random seed for reproducibility (default 42).

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, split_idx).
    """
    split_idx = int(len(X) * train_frac)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    if shuffle_train:
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(X_train))
        X_train = X_train[idx]
        y_train = y_train[idx]

    return X_train, X_test, y_train, y_test, split_idx
