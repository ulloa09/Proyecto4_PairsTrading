import numpy as np
import pandas as pd

# ============================
# --- MÉTRICAS DE DESEMPEÑO ---
# ============================

def annualized_sharpe(mean: float, std: float) -> float:
    """
    Calcula el índice de Sharpe anualizado para datos diarios.
    Asume 252 días hábiles por año.
    """
    annual_rets = mean * 252
    annual_std = std * np.sqrt(252)
    return annual_rets / annual_std if annual_std > 0 else 0


def maximum_drawdown(values: pd.Series) -> float:
    """
    Calcula el Drawdown máximo: la mayor caída porcentual desde un pico.
    """
    roll_max = values.cummax()
    drawdown = (roll_max - values) / roll_max
    return drawdown.max()


def annualized_calmar(mean: float, values: pd.Series) -> float:
    """
    Calcula el índice de Calmar: retorno anualizado / máximo drawdown.
    """
    annual_rets = mean * 252
    max_dd = maximum_drawdown(values)
    return annual_rets / max_dd if max_dd != 0 else 0


def downside_deviation(rets: pd.Series) -> float:
    """
    Calcula la desviación estándar solo de los retornos negativos.
    """
    neg_rets = rets[rets < 0]
    return np.sqrt((neg_rets ** 2).mean()) if len(neg_rets) > 0 else 0


def annualized_sortino(mean: float, rets: pd.Series) -> float:
    """
    Calcula el índice de Sortino anualizado para datos diarios.
    """
    annual_rets = mean * 252
    annual_down_std = downside_deviation(rets) * np.sqrt(252)
    return annual_rets / annual_down_std if annual_down_std > 0 else 0


def win_rate(rets: pd.Series) -> float:
    """
    Calcula la proporción de operaciones ganadoras (win rate).
    """
    total = len(rets)
    return (rets > 0).sum() / total if total > 0 else 0


# =========================================
# --- GENERADOR DE MÉTRICAS COMPLETAS ---
# =========================================

def generate_metrics(portfolio_values: pd.Series) -> dict:
    """
    Calcula todas las métricas principales de desempeño.

    Parameters
    ----------
    portfolio_values : pd.Series
        Serie temporal del valor del portafolio (diario).

    Returns
    -------
    dict : métricas Sharpe, Sortino, Calmar, Drawdown y Win rate.
    """
    rets = portfolio_values.pct_change().dropna()
    mean, std = rets.mean(), rets.std()

    metrics = {
        "Sharpe": annualized_sharpe(mean, std),
        "Sortino": annualized_sortino(mean, rets),
        "Calmar": annualized_calmar(mean, portfolio_values),
        "Max Drawdown": maximum_drawdown(portfolio_values),
        "Win Rate": win_rate(rets),
        "Mean Daily Return": mean,
        "Std Daily Return": std,
    }

    return metrics