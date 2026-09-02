from pathlib import Path

import pandas as pd
import yfinance as yf

RAW_DATA_PATH = Path("data/raw")
RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)


def _flatten_yahoo_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.columns, pd.MultiIndex):
        return df

    flat_columns = []
    for column in df.columns:
        if isinstance(column, tuple):
            flat_columns.append(column[0])
        else:
            flat_columns.append(column)

    normalized = df.copy()
    normalized.columns = flat_columns
    return normalized


def download_stocks(
    tickers: list[str],
    start: str,
    end: str,
) -> dict[str, object]:

    all_data = {}

    for ticker in tickers:

        print(f"Downloading {ticker}...")

        df = yf.download(
            ticker,
            start=start,
            end=end,
            progress=False,
            auto_adjust=False,
        )

        if df.empty:
            print(f"Failed to download {ticker}")
            continue

        df = _flatten_yahoo_columns(df)
        df.to_csv(RAW_DATA_PATH / f"{ticker}.csv")

        all_data[ticker] = df

        print(f"Downloaded {ticker}")

    return all_data