import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def calculate_indicators(
    df: pd.DataFrame,
    short_window: int = 20,
    long_window: int = 50,
    rsi_window: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
) -> pd.DataFrame:
    """Add a standard set of technical indicators to a market DataFrame."""
    if df is None:
        raise ValueError("DataFrame is None.")

    if df.empty:
        raise ValueError("DataFrame is empty.")

    working_df = df.copy()

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in working_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    working_df = working_df.sort_index()
    working_df["Close"] = pd.to_numeric(working_df["Close"], errors="coerce")
    working_df["Volume"] = pd.to_numeric(working_df["Volume"], errors="coerce")
    working_df = working_df.dropna(subset=["Close", "Volume"]).copy()

    close = working_df["Close"]

    working_df["Daily_Return"] = close.pct_change().fillna(0.0)
    working_df["Return_5D"] = close.pct_change(periods=5).fillna(0.0)
    working_df[f"SMA_{short_window}"] = close.rolling(window=short_window, min_periods=short_window).mean()
    working_df[f"SMA_{long_window}"] = close.rolling(window=long_window, min_periods=long_window).mean()

    window_vol = close.pct_change().rolling(window=short_window, min_periods=short_window)
    working_df[f"Volatility_{short_window}"] = window_vol.std().fillna(0.0) * np.sqrt(252)

    delta = close.diff()
    gain = delta.clip(lower=0).fillna(0)
    loss = (-delta).clip(lower=0).fillna(0)

    avg_gain = gain.ewm(com=rsi_window - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=rsi_window - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    working_df[f"RSI_{rsi_window}"] = 100 - (100 / (1 + rs)).fillna(50)

    ema_fast = close.ewm(span=macd_fast, adjust=False).mean()
    ema_slow = close.ewm(span=macd_slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=macd_signal, adjust=False).mean()
    working_df["MACD"] = macd
    working_df["MACD_Signal"] = signal
    working_df["MACD_Histogram"] = macd - signal

    working_df["Price_Position"] = (
        (close - close.rolling(window=long_window, min_periods=1).min())
        / (close.rolling(window=long_window, min_periods=1).max() - close.rolling(window=long_window, min_periods=1).min().replace(0, np.nan))
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return working_df


compute_indicators = calculate_indicators


__all__ = ["calculate_indicators", "compute_indicators"]