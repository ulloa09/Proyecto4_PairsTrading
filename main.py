import pandas as pd

from pairs_search import find_correlated_pairs
from utils import clean_prices

# Reading data CSV
data = pd.read_csv('data/prices.csv')
data = clean_prices(data)
print(data.head())

# Encontrar pares correlacionados
correlated_pairs = find_correlated_pairs(data, window=252, threshold=0.6)
correlated_pairs.to_csv('data/correlated_pairs.csv', index=False)
print("\nFile saved correctly, path:'data/correlated_pairs.csv'")

