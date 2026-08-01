import pandas as pd
import numpy as np
from src.data_loader import downolad_stocks
from src.preprocessing import preprocess_data

stocks = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META"
]

df = downolad_stocks(stocks,start="2020-01-01",end="2025-01-01")

processed_data = preprocess_data(df)


# let mut stocks:Vec<String> = vec![]; just rust stuff


