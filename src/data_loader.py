import yfinance as yf
from pathlib import Path

# Directory to store raw downloaded data
RAW_DATA_PATH = Path("data/raw")
RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)


def download_stocks(
    tickers: list[str],
    start: str,
    end: str
) -> dict[str, object]:
    """
    Downloads historical stock data from Yahoo Finance.

    Args:
        tickers: List of stock ticker symbols.
        start: Start date (YYYY-MM-DD).
        end: End date (YYYY-MM-DD).

    Returns:
        Dictionary where:
            key   -> ticker symbol
            value -> Pandas DataFrame
    """

    all_data = {}

    for ticker in tickers:

        print(f"Downloading {ticker}...")

        df = yf.download(
            ticker,
            start=start,
            end=end,
            progress=False,
            auto_adjust=False
        )

        if df.empty:
            print(f"Failed to download {ticker}")
            continue

        df.to_csv(RAW_DATA_PATH / f"{ticker}.csv")

        all_data[ticker] = df

        print(f"Downloaded {ticker}")

    return all_data