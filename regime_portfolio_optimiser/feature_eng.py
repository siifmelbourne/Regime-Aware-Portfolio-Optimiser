import yfinance as yf
import pandas as pd
import numpy as np

stocks = ['AAPL', 'NVDA', 'MSFT']

for stock in stocks: 
    df = pd.read_csv(f"regime_portfolio_optimiser/data/{stock}_historical_data.csv")
    n = int(np.ceil(df.shape[0]*0.05))

    df['Daily Log Return'] = df['Close'].pct_change().apply(lambda x: np.log(1 + x))
    df['20-day Rolling Vol Annualised'] = df['Daily Log Return'].rolling(window=20).std() * (252 ** 0.5)
    rolling_peak = df['Close'].rolling(window=20, min_periods=1).max()
    df['Max Drawdown'] = (df['Close'] - rolling_peak) / rolling_peak

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    df.to_csv(f"regime_portfolio_optimiser/data/{stock}_historical_data_engineered.csv", index=False)

    # this relies on yf cant be in this file 
    #ticker_data = yf.Ticker(..)
    # stock_info = ticker_data.info
    # stock_info['Historical 95% CVaR'] = -df['Daily Log Return'].nsmallest(n).mean() # averages largest 5% losses to find CVaR (expressed positive)
