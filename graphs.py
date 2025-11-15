"""
This module provides a comprehensive suite of plotting functions for visualizing key components of the Pairs Trading strategy. It includes tools for analyzing spreads derived from Ordinary Least Squares (OLS) regression and Kalman Filters, tracking dynamic hedge ratios, visualizing normalized price movements, and interpreting VECM-based trading signals. These visualizations support realistic research workflows by facilitating the assessment of model behavior, trade signals, portfolio evolution, and performance metrics within the Pairs Trading framework.
"""

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


# ============================================================
# 1. OLS SPREAD PLOT
# ============================================================

def plot_spread_regression(df: pd.DataFrame):
    """
    Plot the residual spread from an Ordinary Least Squares (OLS) regression between two asset price series.

    This plot visualizes the spread (residuals) computed by regressing the dependent asset's price on the independent asset's price,
    which is a foundational step in pairs trading strategies to identify mean-reverting relationships.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing at least two columns representing price series of two assets.
        The first column is treated as the dependent variable (Y), and the second as the independent variable (X).

    Notes
    -----
    - The spread is calculated as the residuals from the OLS regression (Y ~ X).
    - Prices are not directly plotted; only the spread is shown.
    - Useful for preliminary analysis before applying dynamic models such as Kalman Filters.
    """
    col1 = df.columns[0]
    col2 = df.columns[1]

    Y = df[col1].astype(float)
    X = sm.add_constant(df[col2].astype(float))
    model = sm.OLS(Y, X).fit()

    spread = model.resid

    plt.figure(figsize=(12, 4))
    plt.plot(spread, color="purple", linewidth=2)
    plt.axhline(spread.mean(), color="black", linestyle="--", alpha=0.8)
    plt.title(f"OLS Regression Spread ({col1} ~ {col2})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================
# 2. NORMALIZED PRICES PLOT
# ============================================================

def plot_normalized_prices(df: pd.DataFrame, title: str = "Normalized Prices"):
    """
    Plot normalized price series of two assets to visualize relative movements and crossover points.

    Normalization is performed by scaling each price series to start at 1, enabling direct comparison of relative price changes over time.
    This visualization aids in identifying convergence or divergence patterns critical for pairs trading decisions.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame indexed by dates containing exactly two columns of asset prices.
    title : str, optional
        Title for the plot (default is "Normalized Prices").

    Notes
    -----
    - The DataFrame index is converted to datetime if not already in datetime format.
    - Normalization facilitates comparison by removing scale differences between assets.
    - Useful for visual inspection of potential trading signals based on price crossovers.
    """
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = pd.to_datetime(df.index)

    normalized_df = df / df.iloc[0]

    plt.figure(figsize=(12, 6))
    for col in normalized_df.columns:
        plt.plot(normalized_df.index, normalized_df[col], label=col)

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Normalized Price (base 1)")
    plt.legend()
    plt.grid(True)
    plt.show()


# ============================================================
# 3. DYNAMIC EIGENVECTORS (KALMAN FILTER 2)
# ============================================================

def plot_dynamic_eigenvectors(df: pd.DataFrame):
    """
    Visualize the time evolution of dynamic eigenvectors estimated via Kalman Filter 2.

    These eigenvectors represent time-varying components capturing the state dynamics in the pairs trading model,
    providing insights into the changing relationships between assets.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame indexed by dates containing columns 'e1_hat' and 'e2_hat' representing the estimated eigenvectors.

    Notes
    -----
    - The plot shows the trajectories of the first two eigenvectors over time.
    - Useful for understanding model adaptation and dynamic factor behavior in real-time trading analysis.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["e1_hat"], label="v₁ₜ", color='teal')
    plt.plot(df.index, df["e2_hat"], label="v₂ₜ", color='orange')

    plt.title("Dynamic Estimated Eigenvectors (Kalman 2)")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


# ============================================================
# 4. VECM SIGNALS (LONG / SHORT / EXIT)
# ============================================================

def plot_vecm_signals(results_df: pd.DataFrame, entry_long_idx: list[int], entry_short_idx: list[int], exit_idx: list[int], theta: float):
    """
    Plot normalized VECM values with annotated trading signals indicating long entries, short entries, and exits.

    This visualization helps in analyzing the timing and effectiveness of VECM-based trade signals within the pairs trading framework.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame indexed by dates containing a 'vecm_norm' column representing normalized VECM values (z-scores).
    entry_long_idx : list[int]
        List of integer indices corresponding to long entry signal points.
    entry_short_idx : list[int]
        List of integer indices corresponding to short entry signal points.
    exit_idx : list[int]
        List of integer indices corresponding to exit signal points.
    theta : float
        Threshold parameter used for generating trading signals.

    Notes
    -----
    - The function converts integer indices to dates for plotting.
    - Horizontal lines mark the positive and negative thresholds (+θ and -θ).
    - Useful for backtesting and validating signal generation logic.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(results_df.index, results_df["vecm_norm"], color="steelblue", lw=1.4,
            label="Normalized VECM (z-score)")

    ax.axhline(theta, color="gray", linestyle="--", lw=1, label=f"+θ = {theta}")
    ax.axhline(-theta, color="gray", linestyle="--", lw=1, label=f"-θ = {-theta}")
    ax.axhline(0, color="black", linestyle=":", lw=1)

    def _idx_to_dates(idxs, index):
        return [index[i] for i in idxs if 0 <= i < len(index)]

    x_long = _idx_to_dates(entry_long_idx, results_df.index)
    x_short = _idx_to_dates(entry_short_idx, results_df.index)
    x_exit = _idx_to_dates(exit_idx, results_df.index)

    y_long = results_df["vecm_norm"].iloc[entry_long_idx]
    y_short = results_df["vecm_norm"].iloc[entry_short_idx]
    y_exit = results_df["vecm_norm"].iloc[exit_idx]

    ax.scatter(x_long, y_long, color="red", marker="v", s=90, label="LONG Entry")
    ax.scatter(x_short, y_short, color="green", marker="^", s=90, label="SHORT Entry")
    ax.scatter(x_exit, y_exit, color="black", marker="x", s=70, label="Exit")

    ax.set_title("Trading Signals — Normalized VECM (Kalman 2)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized VECM (z-score)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


# ============================================================
# 5. SPREAD EVOLUTION (KALMAN FILTER 1)
# ============================================================

def plot_spread_evolution(results_df: pd.DataFrame, asset1: str, asset2: str):
    """
    Plot the dynamic spread estimated by Kalman Filter 1 between two assets over time.

    The spread represents the residual between the dependent asset price and the dynamically weighted independent asset price,
    reflecting the evolving relationship captured by the Kalman Filter.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame indexed by dates containing a 'spread' column representing the dynamic spread series.
    asset1 : str
        Name of the dependent asset.
    asset2 : str
        Name of the independent asset.

    Notes
    -----
    - The plot includes mean and ±1 and ±2 standard deviation bands to assess spread volatility.
    - Visualizing spread dynamics is critical for identifying entry and exit points in pairs trading.
    """
    spread_series = results_df["spread"]

    plt.figure(figsize=(12, 6))
    plt.plot(spread_series, color="steelblue", linewidth=1.8,
             label="Dynamic Spread (Kalman 1)", alpha=0.7)

    mean_spread = spread_series.mean()
    std_spread = spread_series.std()

    plt.axhline(mean_spread, color="orange", linestyle="--", label="Spread Mean")

    plt.fill_between(
        spread_series.index,
        mean_spread + std_spread,
        mean_spread - std_spread,
        color='orange', alpha=0.15, label="±1 Standard Deviation"
    )
    plt.fill_between(
        spread_series.index,
        mean_spread + 2 * std_spread,
        mean_spread - 2 * std_spread,
        color='orange', alpha=0.08, label="±2 Standard Deviations"
    )

    plt.title(f"Dynamic Spread Evolution — {asset1} vs {asset2}")
    plt.xlabel("Date")
    plt.ylabel("Spread (P1 - β_t * P2)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.show()


# ============================================================
# 6. PORTFOLIO EVOLUTION (60/20/20)
# ============================================================

def plot_portfolio_evolution(portfolio_series: pd.Series, split_ratios=(0.6, 0.2, 0.2)):
    """
    Plot the evolution of portfolio value over time, highlighting training, testing, and validation phases.

    This visualization segments the portfolio timeline according to specified split ratios (default 60% train, 20% test, 20% validation),
    facilitating performance assessment across different stages of model development and deployment.

    Parameters
    ----------
    portfolio_series : pd.Series
        Time series of portfolio values indexed by dates.
    split_ratios : tuple, optional
        Tuple of three floats representing the proportions for train, test, and validation splits respectively (default is (0.6, 0.2, 0.2)).

    Notes
    -----
    - The function assumes the portfolio_series is ordered chronologically.
    - Colored spans indicate the different phases for clarity.
    - Useful for evaluating model robustness and overfitting risks.
    """
    n = len(portfolio_series)
    train_end = int(n * split_ratios[0])
    test_end = int(n * (split_ratios[0] + split_ratios[1]))

    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_series.index, portfolio_series.values,
             label="Portfolio Value", linewidth=1.8)

    plt.axvspan(portfolio_series.index[0], portfolio_series.index[train_end],
                color='green', alpha=0.08, label='Train (60%)')
    plt.axvspan(portfolio_series.index[train_end], portfolio_series.index[test_end],
                color='yellow', alpha=0.12, label='Test (20%)')
    plt.axvspan(portfolio_series.index[test_end], portfolio_series.index[-1],
                color='red', alpha=0.08, label='Validation (20%)')

    plt.title("Portfolio Value Evolution (Train / Test / Validation)")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================
# 7. COMPARISON SPREAD VS VECM
# ============================================================

def plot_spread_vs_vecm(results_df: pd.DataFrame):
    """
    Compare the dynamic spread from Kalman Filter 1 with the normalized VECM values from Kalman Filter 2.

    This plot facilitates side-by-side visual evaluation of two complementary model components,
    aiding in understanding their interactions and relative behaviors in pairs trading.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame indexed by dates containing 'spread' and 'vecm_norm' columns.

    Notes
    -----
    - Both time series are plotted on the same axes for direct comparison.
    - Useful for diagnosing model consistency and signal alignment.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(results_df.index, results_df["spread"], label="Spread (Kalman 1)")
    plt.plot(results_df.index, results_df["vecm_norm"],
             label="Normalized VECM (Kalman 2)")

    plt.title("Comparison: Spread vs Normalized VECM")
    plt.xlabel("Date")
    plt.ylabel("Estimated Value")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================
# 8. TRADE RETURNS DISTRIBUTION
# ============================================================

def plot_trade_returns_distribution(pnl_history: list[float]):
    """
    Plot the distribution of trade profit and loss (PnL) values and print key trading performance statistics.

    This visualization and summary provide insights into trade profitability, risk, and overall strategy effectiveness.

    Parameters
    ----------
    pnl_history : list of float
        List of closed-trade PnL values representing realized gains and losses.

    Notes
    -----
    - If no trades are available, a warning message is printed and plotting is skipped.
    - The histogram includes kernel density estimation (KDE) for smooth distribution representation.
    - Printed statistics include number of trades, win rate, mean, median, standard deviation, average loss, and average gain.
    """
    if not pnl_history:
        print("⚠️ No closed trades to analyze.")
        return

    pnl = np.array(pnl_history)

    mean_pnl = pnl.mean()
    median_pnl = np.median(pnl)
    std_pnl = pnl.std()
    win_rate = (pnl > 0).mean()

    plt.figure(figsize=(12, 6))
    sns.histplot(pnl, bins=30, kde=True, color="lightblue", alpha=0.4)

    plt.axvline(mean_pnl, color="black", linestyle="--", label=f"Mean = {mean_pnl:.2f}")
    plt.axvline(median_pnl, color="red", linestyle="--", label=f"Median = {median_pnl:.2f}")

    plt.title("Distribution of Trade Returns")
    plt.xlabel("PnL per Trade ($)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("\n=== Trading Statistics ===")
    print(f"Number of Trades: {len(pnl)}")
    print(f"Win Rate: {win_rate*100:.2f}%")
    print(f"Mean PnL: {mean_pnl:.2f}")
    print(f"Median PnL: {median_pnl:.2f}")
    print(f"Std. Deviation: {std_pnl:.2f}")
    print(f"Average Loss: {pnl[pnl<0].mean():.2f}")
    print(f"Average Gain: {pnl[pnl>0].mean():.2f}")

# ============================================================
# 9. HEDGE RATIO EVOLUTION (KALMAN FILTER 1)
# ============================================================

def plot_hedge_ratio_evolution(results_df: pd.DataFrame):
    """
    Plot the time evolution of the hedge ratio estimated by Kalman Filter 1.

    The hedge ratio (βₜ) represents the dynamically estimated coefficient linking the prices of two assets,
    crucial for constructing and adjusting pairs trading positions over time.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame indexed by dates containing a 'hedge_ratio' column.

    Raises
    ------
    KeyError
        If the 'hedge_ratio' column is not present in the DataFrame.

    Notes
    -----
    - The plot includes the mean hedge ratio as a reference line.
    - Monitoring hedge ratio dynamics helps in understanding model stability and market regime changes.
    """
    if "hedge_ratio" not in results_df.columns:
        raise KeyError("results_df must contain 'hedge_ratio' column")

    hedge = results_df["hedge_ratio"]

    plt.figure(figsize=(12, 6))
    plt.plot(hedge.index, hedge.values, color="purple", linewidth=1.8, label="Hedge Ratio βₜ (Kalman 1)")

    mean_beta = hedge.mean()

    plt.axhline(mean_beta, color="orange", linestyle="--", linewidth=1.2, label="Mean βₜ")

    plt.title("Dynamic Hedge Ratio Evolution (βₜ)", fontsize=13, weight="bold")
    plt.xlabel("Date")
    plt.ylabel("Hedge Ratio βₜ")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
