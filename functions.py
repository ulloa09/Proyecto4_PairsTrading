"""Module providing core portfolio valuation functions for the Pairs Trading project.

This module supports the dynamic hedging strategy implemented with Kalman filters to estimate hedge ratios in a market-neutral pairs trading framework. The functions here enable realistic backtesting by accurately computing the portfolio value based on both long and short positions in the paired assets, reflecting price changes and position sizes over time.
"""

from objects import Operation

def get_portfolio_value(cash, active_long_ops: list[Operation], active_short_ops: list[Operation],
                        x_ticker: str, y_ticker: str, p1: float, p2: float):
    """
    Compute the total portfolio value given current cash and active positions in a pairs trading strategy.

    This function calculates the combined value of the portfolio by summing the cash on hand and the
    market value of active long and short operations on two paired assets. It is a critical component
    in the trading engine during backtesting, providing a realistic and up-to-date valuation of the
    portfolio that reflects changes in asset prices and position sizes.

    Parameters
    ----------
    cash : float
        The current cash balance in the portfolio.
    active_long_ops : list of Operation
        List of active long positions (buy operations) on the paired assets.
    active_short_ops : list of Operation
        List of active short positions (sell operations) on the paired assets.
    x_ticker : str
        Ticker symbol for the first asset in the pair.
    y_ticker : str
        Ticker symbol for the second asset in the pair.
    p1 : float
        Current market price of the first asset (x_ticker).
    p2 : float
        Current market price of the second asset (y_ticker).

    Returns
    -------
    float
        The total portfolio value combining cash and the mark-to-market value of all active positions.

    Notes
    -----
    - Long positions contribute positively to portfolio value by multiplying the number of shares by the current price.
    - Short positions are valued based on the difference between the open price and current price, multiplied by the number of shares, reflecting profits or losses.
    - The function assumes a market-neutral structure typical of pairs trading, where long and short exposures are balanced dynamically.
    """
    val = cash

    ## OPERACIONES LARGAS
    for position in active_long_ops:
        if position.ticker == x_ticker:
            val += p1 * position.n_shares

        if position.ticker == y_ticker:
            val += p2 * position.n_shares


    ## OPERACIONES CORTAS
    for position in active_short_ops:
        if position.ticker == x_ticker:
            val += ((position.open_price - p1) * position.n_shares)

        if position.ticker == y_ticker:
            val += ((position.open_price - p2) * position.n_shares)


    return val