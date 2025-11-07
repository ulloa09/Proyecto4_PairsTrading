from dataclasses import dataclass

@dataclass
class Operation:
    '''
    A class to represent the trading operation
    '''

    ticker: str
    type: str
    n_shares: int
    open_price: float
    close_price: float
    date: str