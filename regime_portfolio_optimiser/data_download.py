import yfinance as yf
import pandas as pd

stocks = ['AAPL', 'NVDA', 'MSFT']

for stock in stocks:
    try: 
        ticker_data = yf.Ticker(stock)
        df = ticker_data.history(start="2010-02-10")
    except Exception as e:
        print(f"Error fetching data for {stock}: {e}")
        raise

    df.index = df.index.tz_localize(None).normalize()
    df.to_csv(f"regime_portfolio_optimiser/data/{stock}_historical_data.csv")
