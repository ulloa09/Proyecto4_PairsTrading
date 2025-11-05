import pandas as pd
import statsmodels.api as sm
import itertools

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen


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


    # === Guardar matriz completa de correlaciones móviles ===
    rolling_corr = data.rolling(window).corr()       # devuelve panel largo (nivel MultiIndex)
    corr_matrix = rolling_corr.groupby(level=1).mean()  # promedio por ticker global
    corr_matrix.to_csv("data/rolling_correlation_matrix.csv")

    return high_corr_pairs





def ols_and_adf(prices: pd.DataFrame, correlated_pairs: pd.DataFrame, save_path: str = "data/ols_adf_results.csv") -> pd.DataFrame:
    """
    Performs OLS regression and ADF test for all correlated pairs.

    Parameters
    ----------
    prices_path : str
        Path to the CSV file containing asset prices (Date column + tickers).
    correlated_pairs_path : str
        Path to the CSV file containing correlated pairs and Mean_Correlation.
    save_path : str, optional
        Path to save results (default='data/ols_adf_results.csv').

    Returns
    -------
    pd.DataFrame
        Results DataFrame with columns:
        ['Asset_1', 'Asset_2', 'beta1_OLS', 'ADF_stat', 'ADF_pvalue', 'n_obs', 'Mean_Correlation'].
    """

    # === Load input data ===
    prices = prices.copy()
    correlated_pairs = correlated_pairs.copy()

    # Ensure proper formatting
    if "Date" in prices.columns:
        prices["Date"] = pd.to_datetime(prices["Date"])
        prices = prices.set_index("Date")

    print("\n=== Running OLS and ADF tests for correlated pairs ===\n")
    results = []

    for _, row in correlated_pairs.iterrows():
        asset1 = row["Asset_1"]
        asset2 = row["Asset_2"]
        mean_corr = row["Mean_Correlation"]

        # Prepare data
        df_pair = prices[[asset1, asset2]].dropna()
        n_obs = len(df_pair)

        if n_obs < 50:
            print(f"⚠️  Skipping {asset1}-{asset2}: insufficient observations ({n_obs})")
            continue

        # OLS regression: Y = β0 + β1 * X
        X = sm.add_constant(df_pair[asset2])
        model = sm.OLS(df_pair[asset1], X).fit()
        beta1 = model.params[asset2]
        residuals = model.resid

        # ADF test on residuals
        adf_stat, adf_pvalue, _, _, _, _ = adfuller(residuals)

        # Check stationarity at 95% confidence (p < 0.05)
        if adf_pvalue < 0.05:
            print(f"✅ {asset1} and {asset2} have stationary spread (p={adf_pvalue:.4f})")
            stationary = True
        else:
            stationary = False

        results.append({
            "Asset_1": asset1,
            "Asset_2": asset2,
            "beta1_OLS": beta1,
            "ADF_stat": adf_stat,
            "ADF_pvalue": adf_pvalue,
            "n_obs": n_obs,
            "Mean_Correlation": mean_corr,
            "Stationary": stationary
        })

    # Build and save results DataFrame
    results_df = pd.DataFrame(results).sort_values(by="ADF_pvalue")
    results_df.to_csv(save_path, index=False)
    print(f"\nResults saved successfully at: {save_path}")
    print(f"{(results_df['ADF_pvalue'] < 0.05).sum()} pairs show cointegration (ADF p<0.05).")

    return results_df


def run_johansen_test(prices_df: pd.DataFrame, adf_results_df: pd.DataFrame,
                      save_path: str = "data/results_johansen.csv") -> pd.DataFrame:
    """
    Performs Johansen cointegration test for pairs that passed ADF (p < 0.05).
    Saves trace statistics, 5% critical values, and first eigenvector (hedge ratio).

    Parameters
    ----------
    prices_df : pd.DataFrame
        DataFrame containing price data (Date index + tickers as columns).
    adf_results_df : pd.DataFrame
        DataFrame containing ADF test results (from results_ols_adf.csv).
    save_path : str, optional
        Path to save Johansen results CSV (default='data/results_johansen.csv').

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        ['Asset_1', 'Asset_2', 'Trace_Stat', 'Crit_Value_5%',
         'Rank_detected', 'Eigenvector_1', 'Eigenvector_2']
    """

    # Filter only pairs that passed ADF (p < 0.05)
    cointegrated_pairs = adf_results_df[adf_results_df["Stationary"] == True]

    print(f"\n=== Running Johansen test for {len(cointegrated_pairs)} cointegrated pairs ===\n")
    results = []

    for _, row in cointegrated_pairs.iterrows():
        asset1 = row["Asset_1"]
        asset2 = row["Asset_2"]

        # Skip missing assets
        if asset1 not in prices_df.columns or asset2 not in prices_df.columns:
            print(f"⚠️  Skipping {asset1}-{asset2}: missing in prices.")
            continue

        # Extract pair data and drop NaN
        df_pair = prices_df[[asset1, asset2]].dropna()

        # Johansen test
        johansen = coint_johansen(df_pair, det_order=0, k_ar_diff=1)

        # Trace statistic and 5% critical value
        trace_stat = johansen.lr1[0]
        crit_5 = johansen.cvt[0, 1]

        # Rank detection: if trace_stat > critical value → rank = 1
        rank_detected = 1 if trace_stat > crit_5 else 0

        # First eigenvector (normalized)
        eigenvector = johansen.evec[:, 0]
        beta_1 = eigenvector[0]
        beta_2 = eigenvector[1]

        if rank_detected == 1:
            print(f"✅ {asset1} and {asset2}: cointegration rank=1 (trace {trace_stat:.2f} > {crit_5:.2f})")
            cointegrated = True
        else:
            cointegrated = False


        results.append({
            "Asset_1": asset1,
            "Asset_2": asset2,
            "Trace_Stat": trace_stat,
            "Crit_Value_5%": crit_5,
            "Rank_detected": rank_detected,
            "Eigenvector_1": beta_1,
            "Eigenvector_2": beta_2,
            "Cointegrated": cointegrated
        })

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(save_path, index=False)
    print(f"\nJohansen results saved successfully at: {save_path}")

    return results_df


def extract_pair(prices_df: pd.DataFrame, johansen_df: pd.DataFrame,
                               index: int, save: bool = True, path_prefix: str = "data/") -> pd.DataFrame:
    """
    Extracts and saves a pair of assets based on its row index from the Johansen results DataFrame.

    Parameters
    ----------
    prices_df : pd.DataFrame
        DataFrame with price data (Date index + tickers as columns).
    johansen_df : pd.DataFrame
        Full Johansen results DataFrame with columns ['Asset_1', 'Asset_2'].
    index : int
        Row index in johansen_df indicating which pair to extract.
    save : bool, optional
        Whether to save the pair CSV (default=True).
    path_prefix : str, optional
        Folder path prefix for saving (default='data/').

    Returns
    -------
    pd.DataFrame
        DataFrame containing ['Date', Asset_1, Asset_2'] with aligned and cleaned data.
    """

    prices_df = prices_df.copy()

    # Extract pair info by index
    row = johansen_df.iloc[index]
    asset1 = row["Asset_1"]
    asset2 = row["Asset_2"]

    if asset1 not in prices_df.columns or asset2 not in prices_df.columns:
        raise ValueError(f"Assets {asset1} or {asset2} not found in price data.")

    df_pair = prices_df[[asset1, asset2]].dropna().copy()
    if save:
        filename = f"{path_prefix}{asset1}_{asset2}_pair.csv"
        df_pair.to_csv(filename)
        print(f"💾 Saved pair to: {filename} with {len(df_pair)}")

    return df_pair