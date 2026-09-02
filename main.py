from pathlib import Path

from src.data_loader import download_stocks
from src.indicators import calculate_indicators
from src.preprocessing import preprocess_data
from src.utils import ensure_directory

PROCESSED_DATA_PATH = Path("data/processed")
ensure_directory(PROCESSED_DATA_PATH)

stocks = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
]

market_data = download_stocks(
    stocks,
    "2020-01-01",
    "2025-01-01",
)

processed_data = {}

for ticker, df in market_data.items():
    cleaned_df = preprocess_data(df)
    enhanced_df = calculate_indicators(cleaned_df)

    enhanced_df.to_csv(PROCESSED_DATA_PATH / f"{ticker}_processed.csv")
    processed_data[ticker] = enhanced_df

print(f"Processed {len(processed_data)} stock datasets.")