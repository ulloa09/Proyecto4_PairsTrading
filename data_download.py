import yfinance as yf
import pandas as pd

"""
Data Loader – Pairs Trading Project
Downloads daily prices for multiple tickers and stores them in data/prices.csv
"""

def download_price_data(tickers, start_date, end_date):
    """
    Download adjusted price data for a list of tickers and save to CSV.

    Parameters
    ----------
    tickers : list[str]
        Ticker symbols to download.
    start_date : str
        'YYYY-MM-DD'
    end_date : str
        'YYYY-MM-DD'

    Returns
    -------
    pd.DataFrame
        Prices with dates as index and tickers as columns.
    """
    frames = []

    for ticker in tickers:
        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval="1d",
            auto_adjust=False,   # so we get 'Adj Close'
            progress=False
        )

        if df.empty:
            # skip ticker with no data
            continue

        if "Adj Close" in df.columns:
            series = df[["Adj Close"]].rename(columns={"Adj Close": ticker})
        elif "Close" in df.columns:
            series = df[["Close"]].rename(columns={"Close": ticker})
        else:
            # no usable price column, skip
            continue

        frames.append(series)

    if not frames:
        raise ValueError("No valid data was downloaded for the provided tickers.")

    # merge all tickers on the date index
    data = pd.concat(frames, axis=1)

    # fill small gaps (holidays, missing days)
    data = data.ffill().bfill()

    # Ensure clean numeric data, no header remnants
    data = data.reset_index()  # Convert index to column
    data.columns.name = None  # Remove potential column-level names
    data.index.name = None  # Remove index name
    data.rename(columns={'index': 'Date'}, inplace=True)

    # save to csv (single file to reuse later)
    data.to_csv("data/prices.csv", index=True)

    return data

start_date = '2010-11-01'
end_date = '2025-11-01'
tickers = [
    # --- Technology ---
    'AAPL',  # Apple Inc.
    'NVDA',  # NVIDIA Corporation
    'INTC',  # Intel Corporation
    'ORCL',  # Oracle Corporation

    # --- Communication Services ---
    'VZ',    # Verizon Communications Inc.
    'T',     # AT&T Inc.
    'DIS',   # The Walt Disney Company

    # --- Financials ---
    'JPM',   # JPMorgan Chase & Co.
    'BAC',   # Bank of America Corporation
    'C',     # Citigroup Inc.

    # --- Consumer Discretionary ---
    'MCD',   # McDonald's Corporation
    'SBUX',  # Starbucks Corporation
    'NKE',   # Nike, Inc.

    # --- Consumer Staples ---
    'KO',    # The Coca-Cola Company
    'PEP',   # PepsiCo, Inc.
    'WMT',   # Walmart Inc.

    # --- Energy ---
    'XOM',   # Exxon Mobil Corporation
    'CVX',   # Chevron Corporation

    # --- Materials ---
    'DD',    # DuPont de Nemours, Inc.
    'APD',   # Air Products & Chemicals, Inc.

    # --- Industrials ---
    'CAT',   # Caterpillar Inc.
    'GE',    # General Electric Company
    'BA',    # The Boeing Company

    # --- Healthcare ---
    'JNJ',   # Johnson & Johnson
    'PFE',   # Pfizer Inc.
    'MRK',   # Merck & Co., Inc.

    # --- Real Estate ---
    'SPG',   # Simon Property Group, Inc.

    # --- Utilities ---
    'NEE',   # NextEra Energy, Inc.
    'DUK',   # Duke Energy Corporation
    'SO'     # The Southern Company
]

download_price_data(tickers, start_date, end_date)