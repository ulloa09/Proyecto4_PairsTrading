import pandas as pd

from utils import clean_prices

import pandas as pd
import itertools

from utils import clean_prices


import pandas as pd
import itertools
from utils import clean_prices


def find_correlated_pairs(data: pd.DataFrame, window: int = 60, threshold: float = 0.7) -> pd.DataFrame:
    """
    Calculates rolling correlations between all asset pairs in the price DataFrame
    and returns the pairs that exceed the average correlation threshold.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing asset prices. Must include a 'Date' column or a DateTime index.
    window : int, optional
        Window size for rolling correlation (default=60).
    threshold : float, optional
        Minimum average correlation to consider a pair as a candidate (default=0.7).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['Asset_1', 'Asset_2', 'Mean_Correlation'],
        sorted from highest to lowest average correlation.
    """

    # === Read and clean data ===
    data = data.copy()

    # Ensure the index is datetime type and sorted
    if "Date" in data.columns:
        data["Date"] = pd.to_datetime(data["Date"])
        data = data.set_index("Date")
    data = data.sort_index()

    # === Calculate rolling correlations ===
    tickers = data.columns
    pairs = list(itertools.combinations(tickers, 2))
    results = []

    for a, b in pairs:
        rolling_corr = data[a].rolling(window).corr(data[b])
        mean_corr = rolling_corr.mean()
        results.append((a, b, mean_corr))

    # === Create and filter the correlation DataFrame ===
    corr_df = pd.DataFrame(results, columns=["Asset_1", "Asset_2", "Mean_Correlation"])
    corr_df = corr_df.dropna().sort_values(by="Mean_Correlation", ascending=False)

    # === Filter pairs above the threshold ===
    high_corr_pairs = corr_df[corr_df["Mean_Correlation"] > threshold]
    print(f"\nPairs with mean correlation > {threshold}:\n", high_corr_pairs.head(10))

    return high_corr_pairs

