import yfinance as yf
import pandas as pd
import numpy as np

def feature_eng(stocks):
    for stock in stocks: 
        df = pd.read_csv(f"regime_portfolio_optimiser/data/{stock}_historical_data.csv")
        n = int(np.ceil(df.shape[0]*0.05))

        df['Daily Log Return'] = df['Close'].pct_change().apply(lambda x: np.log(1 + x))

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        
        df.to_csv(f"regime_portfolio_optimiser/data/{stock}_historical_data_engineered.csv", index=False)
    return