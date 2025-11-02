import pandas as pd

# Reading data CSV
data = pd.read_csv('data/prices.csv')
data = data.drop(index=0)
data = data.drop(columns=['Price'])
data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date')

print(data.head())
