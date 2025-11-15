import numpy as np
import pandas as pd

"""
This module provides performance metrics calculations for financial portfolios.
It includes functions to compute annualized Sharpe ratio, maximum drawdown,
Calmar ratio, downside deviation, annualized Sortino ratio, win rate, and a
comprehensive metrics generator that aggregates these along with profit factor,
total commissions, and total borrow cost.
"""

# ============================
# --- PERFORMANCE METRICS ---
# ============================
DAYS = 252

def annualized_sharpe(mean: float, std: float) -> float:
    """
    Calculate the annualized Sharpe ratio for daily data.
    Assumes 252 trading days per year.

    Parameters
    ----------
    mean : float
        Mean daily return.
    std : float
        Standard deviation of daily returns.

    Returns
    -------
    float
        Annualized Sharpe ratio.
    """
    annual_rets = mean * DAYS
    annual_std = std * np.sqrt(DAYS)
    return annual_rets / annual_std if annual_std > 0 else 0


def maximum_drawdown(values: pd.Series) -> float:
    """
    Calculate the maximum drawdown: the largest percentage drop from a peak.

    Parameters
    ----------
    values : pd.Series
        Portfolio values over time.

    Returns
    -------
    float
        Maximum drawdown as a decimal.
    """
    roll_max = values.cummax()
    drawdown = (roll_max - values) / roll_max
    return drawdown.max()


def annualized_calmar(mean: float, values: pd.Series) -> float:
    """
    Calculate the Calmar ratio: annualized return divided by maximum drawdown.

    Parameters
    ----------
    mean : float
        Mean daily return.
    values : pd.Series
        Portfolio values over time.

    Returns
    -------
    float
        Calmar ratio.
    """
    annual_rets = mean * DAYS
    max_dd = maximum_drawdown(values)
    return annual_rets / max_dd if max_dd != 0 else 0


def downside_deviation(rets: pd.Series) -> float:
    """
    Calculate the standard deviation of negative returns only.

    Parameters
    ----------
    rets : pd.Series
        Series of returns.

    Returns
    -------
    float
        Downside deviation.
    """
    neg_rets = rets[rets < 0]
    return np.sqrt((neg_rets ** 2).mean()) if len(neg_rets) > 0 else 0


def annualized_sortino(mean: float, rets: pd.Series) -> float:
    """
    Calculate the annualized Sortino ratio for daily data.

    Parameters
    ----------
    mean : float
        Mean daily return.
    rets : pd.Series
        Series of returns.

    Returns
    -------
    float
        Annualized Sortino ratio.
    """
    annual_rets = mean * DAYS
    annual_down_std = downside_deviation(rets) * np.sqrt(DAYS)
    return annual_rets / annual_down_std if annual_down_std > 0 else 0


def win_rate(pnl_history: list[float]) -> float:
    """
    Calculate the proportion of winning trades (win rate).

    Parameters
    ----------
    pnl_history : list of float
        List of profit and loss values for trades.

    Returns
    -------
    float
        Win rate as a decimal.
    """
    pnl_array = np.array(pnl_history)
    wins = np.sum(pnl_array > 0)
    total = len(pnl_array)
    return wins / total


# =========================================
# --- COMPLETE METRICS GENERATOR ---
# =========================================

def generate_metrics(portfolio_values: pd.Series,
                     pnl_history: list[float],
                     total_borrow_cost: float,
                     total_commissions: float) -> dict:
    """
    Compute all comprehensive performance metrics for the strategy.

    This version adds:
    - Profit Factor
    - Total Commissions
    - Total Borrow Cost
    (Without modifying any existing functionality)

    Parameters
    ----------
    portfolio_values : pd.Series
        Portfolio values over time.
    pnl_history : list of float
        List of profit and loss values for trades.
    total_borrow_cost : float
        Total borrowing cost incurred.
    total_commissions : float
        Total commissions paid.

    Returns
    -------
    dict
        Dictionary containing all calculated performance metrics.
    """

    # --- Calculate daily returns ---
    rets = portfolio_values.pct_change().dropna()
    mean, std = rets.mean(), rets.std()

    # --- Existing main metrics ---
    sharpe = annualized_sharpe(mean, std)
    sortino = annualized_sortino(mean, rets)
    calmar = annualized_calmar(mean, portfolio_values)
    max_dd = maximum_drawdown(portfolio_values)
    wr = win_rate(pnl_history)

    # --- New metrics ---
    # Profit factor
    pnl_array = np.array(pnl_history)
    gains = pnl_array[pnl_array > 0].sum()
    losses = abs(pnl_array[pnl_array < 0].sum())
    profit_factor = gains / losses if losses > 0 else np.nan

    # Final metrics dictionary
    metrics = {
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Calmar": calmar,
        "Max Drawdown": max_dd,
        "Win Rate": wr,
        "Mean Daily Return": mean,
        "Std Daily Return": std,
        "Profit Factor": profit_factor,
        "Total Commissions": total_commissions,
        "Total Borrow Cost": total_borrow_cost
    }

    # --- Pretty printing ---
    print("\n--- PERFORMANCE METRICS ---")
    for k, v in metrics.items():
        if k in ["Sharpe", "Sortino", "Calmar", "Profit Factor"]:
            print(f"{k:20s}: {v:.4f}")
        elif k in ["Total Commissions", "Total Borrow Cost"]:
            print(f"{k:20s}: ${v:,.2f}")
        else:
            print(f"{k:20s}: {v*100:.2f}%")

    metricas = pd.DataFrame(metrics, index=[0]).T
    metricas.to_csv('data/metrics.csv')

    return metrics