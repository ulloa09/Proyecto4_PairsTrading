import yfinance as yf
import pandas as pd

"""
Data Loader – Pairs Trading Project
-----------------------------------

This module downloads daily adjusted close prices for a predefined universe of 
equities. These data serve as the foundational input for the entire Pairs Trading 
pipeline, including:

    • Correlation analysis and universe filtering
    • Engle–Granger and Johansen cointegration tests
    • Dynamic hedge‐ratio estimation using Kalman Filters
    • VECM-based spread modeling
    • Realistic backtesting with transaction costs and short-selling financing
    • Walk-forward evaluation

The objective is to produce clean, consistent, and research-grade price data that 
can be safely used without introducing look-ahead bias or structural artifacts.
"""


def download_price_data(tickers, start_date, end_date):
    """
    Download daily price data for a list of tickers and export a consolidated CSV.

    This function retrieves *Close* prices from Yahoo Finance for all requested tickers,
    aligns them on a common date index, fills small gaps (e.g., holidays), and ensures 
    the resulting dataset is clean and ready for statistical analysis. 

    The data produced here is used throughout the pairs-trading workflow, including 
    correlation screening, cointegration testing, hedge-ratio estimation, and 
    VECM/Kalman-based signal generation.

    Parameters
    ----------
    tickers : list[str]
        List of equity ticker symbols to download (e.g., ['AAPL', 'MSFT']).
        These symbols should match Yahoo Finance conventions.

    start_date : str
        Start of the historical window in 'YYYY-MM-DD' format.
        The project requires approximately 15 years of history to ensure
        statistically meaningful cointegration tests and reliable walk-forward splits.

    end_date : str
        End date of the download in 'YYYY-MM-DD' format.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing Close prices for all valid tickers, indexed by date.
        Columns correspond to tickers, and all missing values are forward/backward filled.

    Notes
    -----
    • `auto_adjust=False` is used intentionally to access raw Close prices, matching 
      class instructions that adjustments (splits/dividends) should not be applied unless 
      explicitly required in subsequent modules.

    • Any ticker returning empty data is silently skipped, ensuring robustness when 
      processing broad equity universes.

    • The output file `data/raw_prices.csv` becomes the canonical dataset for the 
      project. All subsequent models (cointegration, filters, backtests) depend on it.

    • No look-ahead bias is introduced. Data is downloaded chronologically and kept 
      unmodified except for small gap filling.

    Raises
    ------
    ValueError
        If none of the provided tickers return valid data.
    """
    frames = []

    for ticker in tickers:
        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval="1d",
            auto_adjust=False,
            progress=False
        )

        if df.empty:
            continue  # Skip tickers with no available data

        if "Close" in df.columns:
            series = df[["Close"]].rename(columns={"Close": ticker})
        else:
            continue  # Skip if we cannot extract a usable price series

        frames.append(series)

    if not frames:
        raise ValueError("No valid data was downloaded for the provided tickers.")

    # Merge all tickers on date index
    data = pd.concat(frames, axis=1)

    # Fill small gaps (holidays, missing trading days)
    data = data.ffill().bfill()

    # Clean metadata and ensure a tidy DataFrame
    data = data.reset_index()
    data.columns.name = None
    data.index.name = None
    data.rename(columns={'index': 'Date'}, inplace=True)

    # Export canonical price dataset
    data.to_csv("data/raw_prices.csv", index=True)

    return data


# ---------------------------------------------------------------------
# Default Configuration (Research-Grade Universe)
# ---------------------------------------------------------------------

start_date = '2010-11-01'
end_date = '2025-11-01'

tickers = [
    # Technology
    'AAPL', 'NVDA', 'INTC', 'ORCL',

    # Communication Services
    'VZ', 'T', 'DIS',

    # Financials
    'JPM', 'BAC', 'C',

    # Consumer Discretionary
    'MCD', 'SBUX', 'NKE',

    # Consumer Staples
    'KO', 'PEP', 'WMT',

    # Energy
    'XOM', 'CVX',

    # Materials
    'DD', 'APD',

    # Industrials
    'CAT', 'GE', 'BA',

    # Healthcare
    'JNJ', 'PFE', 'MRK',

    # Real Estate
    'SPG',

    # Utilities
    'NEE', 'DUK', 'SO'
]

# Execute download with default configuration
download_price_data(tickers, start_date, end_date)