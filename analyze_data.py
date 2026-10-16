import pandas as pd
import numpy as np
df = pd.read_csv("data/AAPL_daily.csv", header=0)
first_col = df.columns[0]
if first_col == 'Price':
    df = df.iloc[2:].reset_index(drop=True)
    df = df.rename(columns={first_col: 'Date'})
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date']).set_index('Date')
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
print("Rows:", len(df))
print("Date range:", df.index.min(), "to", df.index.max())
print("Min close:", round(df['Close'].min(), 2), "Max close:", round(df['Close'].max(), 2))
print("Total return if bought at low, sold at high:", round((df['Close'].max()/df['Close'].min()-1)*100, 1), "%")
daily_returns = df['Close'].pct_change()
print("Best single day:", round(daily_returns.max()*100, 2), "%")
print("Worst single day:", round(daily_returns.min()*100, 2), "%")
print("Overall return from start to end:", round((df['Close'].iloc[-1]/df['Close'].iloc[0]-1)*100, 2), "%")