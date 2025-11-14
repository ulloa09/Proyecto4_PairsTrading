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
    Computes and plots the spread from an OLS regression using the first
    two columns of the DataFrame. Prices are not plotted.

    Columns are interpreted as:
        col0 -> dependent asset (Y)
        col1 -> independent asset (X)
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
    Plots normalized prices (base 1) of two assets to visualize crossovers.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame indexed by dates with two price columns.
    title : str, optional
        Plot title.
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
    Plots dynamic eigenvectors estimated by Kalman Filter 2.
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
    Plots normalized VECM values and marks long, short and exit trading signals.

    Parameters
    ----------
    results_df : pd.DataFrame
        Must contain 'vecm_norm'.
    entry_long_idx : list[int]
        List of indices for LONG entries.
    entry_short_idx : list[int]
        List of indices for SHORT entries.
    exit_idx : list[int]
        List of indices for exits.
    theta : float
        Threshold used for signal generation.
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
    Plots the dynamic spread estimated by Kalman Filter 1.

    Parameters
    ----------
    results_df : pd.DataFrame
        Must contain 'spread'.
    asset1 : str
        Dependent asset name.
    asset2 : str
        Independent asset name.
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
    Plots the evolution of the portfolio value highlighting training,
    testing and validation phases (60%, 20%, 20%).

    Parameters
    ----------
    portfolio_series : pd.Series
        Time series of portfolio values.
    split_ratios : tuple
        Train, test, validation ratios.
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
    Plots the spread (Kalman 1) compared to the normalized VECM (Kalman 2).
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
    Plots the distribution of trade PnL values and prints trading statistics.

    Parameters
    ----------
    pnl_history : list[float]
        List of closed-trade PnL values.
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
    Plots the evolution of the hedge ratio estimated by Kalman Filter 1.

    Parameters
    ----------
    results_df : pd.DataFrame
        Must contain column 'hedge_ratio'.
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
