import pandas as pd

def clean_prices(data):
    data = data.copy()
    # Convert objects to numeric values
    data = data.apply(pd.to_numeric, errors="coerce")
    data = data.drop(index=0)
    data = data.drop(columns=['Price'])
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.set_index('Date')

    return data