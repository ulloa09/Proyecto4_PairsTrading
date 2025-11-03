import numpy as np
import pandas as pd

def clean_prices(data):
    data = data.copy()
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    data = data.drop(columns=['Price'])
    # Convert objects to numeric values
    for col in data.columns:
        if col != 'Date':
            data[col] = pd.to_numeric(data[col], errors='coerce')

    data = data.set_index('Date')
    data = data.dropna()
    return data