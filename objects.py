"""This module defines data structures used in the pairs trading project for representing trading operations."""

from dataclasses import dataclass

@dataclass
class Operation:
    """
    Represents a trading operation within the pairs trading backtesting framework.

    Parameters
    ----------
    ticker : str
        The stock ticker symbol.
    type : str
        The type of operation, e.g., 'buy' or 'sell'.
    n_shares : int
        The number of shares involved in the operation.
    open_price : float
        The price at which the position was opened.
    close_price : float
        The price at which the position was closed.
    date : str
        The date of the operation.
    """
    ticker: str
    type: str
    n_shares: int
    open_price: float
    close_price: float
    date: str