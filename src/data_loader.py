from pandas import DataFrame
import yfinance as yf
from pathlib import Path


RAW_DATA_PATH = Path("data/raw")

RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)


def downolad_stocks(tickers, start, end):

    for ticker in tickers:
        print(f'Downloading {ticker}...')

        df = yf.download(
            tickers,
            start = start,
            end = end,
            progress=False,
            auto_adjust=False
        )
        path = RAW_DATA_PATH / f"{ticker}.csv"

        df.to_csv(path)

        print(f'Downloaded {ticker}')

    return df

# fn download_data(tokens: Vec<i32>,st: &str,en: &str) -> Vec<String> {

# }
