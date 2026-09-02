from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_loader import download_stocks
from src.indicators import calculate_indicators
from src.preprocessing import preprocess_data
from src.utils import ensure_directory

PROCESSED_DATA_PATH = Path("data/processed")
ensure_directory(PROCESSED_DATA_PATH)


def save_stock_analysis_plot(df, ticker: str) -> Path:
    """Save a technical analysis chart for a stock to the processed folder."""
    plot_path = PROCESSED_DATA_PATH / f"{ticker}_analysis.png"

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True, gridspec_kw={"height_ratios": [3, 1]})

    close = df["Close"]
    sma_20 = df.get("SMA_20")
    sma_50 = df.get("SMA_50")

    axes[0].plot(close.index, close, label="Close", linewidth=2, color="black")
    if sma_20 is not None and sma_20.notna().any():
        axes[0].plot(sma_20.index, sma_20, label="SMA 20", linewidth=1.5, color="dodgerblue")
    if sma_50 is not None and sma_50.notna().any():
        axes[0].plot(sma_50.index, sma_50, label="SMA 50", linewidth=1.5, color="orange")

    axes[0].set_title(f"{ticker} Price Trend and Moving Averages")
    axes[0].set_ylabel("Price")
    axes[0].legend()
    axes[0].grid(True, alpha=0.2)

    rsi = df.get("RSI_14")
    if rsi is not None and rsi.notna().any():
        axes[1].plot(rsi.index, rsi, color="purple", linewidth=1.5, label="RSI 14")
        axes[1].axhline(70, color="red", linestyle="--", alpha=0.7, linewidth=1)
        axes[1].axhline(30, color="green", linestyle="--", alpha=0.7, linewidth=1)
        axes[1].set_ylim(0, 100)
        axes[1].set_ylabel("RSI")
        axes[1].set_xlabel("Date")
        axes[1].legend()
        axes[1].grid(True, alpha=0.2)
    else:
        axes[1].text(0.5, 0.5, "RSI unavailable", ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_axis_off()

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    return plot_path


user_input = input(
    "Enter stock symbols separated by commas (example: AAPL,MSFT,NVDA): "
).strip()

if not user_input:
    stocks = ["AAPL", "MSFT", "NVDA", "AMZN", "META"]
else:
    stocks = [symbol.strip().upper() for symbol in user_input.split(",") if symbol.strip()]

if not stocks:
    raise ValueError("No valid stock symbols were provided.")

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

    chart_path = save_stock_analysis_plot(enhanced_df, ticker)
    print(f"Saved chart: {chart_path}")

print(f"Processed {len(processed_data)} stock datasets.")