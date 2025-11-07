from objects import Operation

def get_portfolio_value(cash, active_long_ops: list[Operation], active_short_ops: list[Operation],
                        x_ticker: str, y_ticker: str):

    val = cash

    ## OPERACIONES LARGAS
    for position in active_long_ops:
        if position.ticker == x_ticker:
            val += position.close_price * position.n_shares

        if position.ticker == y_ticker:
            val += position.close_price * position.n_shares


    ## OPERACIONES CORTAS
    for position in active_short_ops:
        if position.ticker == x_ticker:
            val += ((position.close_price - position.open_price) * position.n_shares)

        if position.ticker == y_ticker:
            val += ((position.close_price - position.open_price) * position.n_shares)


    return val