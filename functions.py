from objects import Operation

def get_portfolio_value(cash, active_long_ops: list[Operation], active_short_ops: list[Operation],
                        x_ticker: str, y_ticker: str, p1: float, p2: float):

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