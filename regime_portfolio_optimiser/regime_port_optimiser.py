import yfinance as yf 
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "market_regime_model"))
from hmm_model import main as run_hmm

def get_asset_data(stocks):
    """
    Fetch historical stock data from Yahoo Finance.

    Args:
        stocks (list): List of stock ticker symbol.

    Returns:
        stock_features (dict): A dictionary containing tuples of stock information and stock data 
        returns_combined (pd.DataFrame): A DataFrame containing the daily log returns for each stock.
    """
    stock_features = {}
    returns_combined = pd.DataFrame()
    for stock in stocks: 
        try: 
            ticker_data = yf.Ticker(stock)
            stock_info = ticker_data.info
            df = ticker_data.history(start="2010-02-10")
        except Exception as e:
            print(f"Error fetching data for {stock}: {e}")
            raise

        n = int(np.ceil(df.shape[0]*0.05))

        df['Daily Log Return'] = df['Close'].pct_change().apply(lambda x: np.log(1 + x))
        df['20-day Rolling Vol Annualised'] = df['Daily Log Return'].rolling(window=20).std() * (252 ** 0.5)
        rolling_peak = df['Close'].rolling(window=20, min_periods=1).max()
        df['Max Drawdown'] = (df['Close'] - rolling_peak) / rolling_peak

        stock_info['Historical 95% CVaR'] = -df['Daily Log Return'].nsmallest(n).mean() # averages largest 5% losses to find CVaR (expressed positive)
        stock_features[stock] = stock_info, df
        returns_combined[stock] = df['Daily Log Return']
    returns_combined.index = returns_combined.index.tz_localize(None).normalize() # normalise datetime index to match regime probabilities datetime index 

    return stock_features, returns_combined

def get_portfolio_statistics(returns, current_regime, weights, rf):
    ...

def get_asset_behaviour(data, regimes):
    ...

def optimise_weights(returns, regimes, method='CVaR'): 
    ...

def main():
    stocks = ['AAPL', 'NVDA', 'MSFT']
    weights = [.3, .4, .3]
    rf = 0.05 # hardcode risk free rate for now 

    if (len(stocks) != len(weights)): 
        print("Number of stocks and weights should be equal")
        return -1

    df_combined, model = run_hmm()

    feature_cols = [
        "Daily Log Return_scaled",
        "20-Day Rolling Volatility_scaled",
        "Distance from 200-Day MA_scaled",
        "Max Drawdown_scaled",
        "Volume Change_scaled",
        "ATR_scaled",
        "RSI_scaled",
        "MACD_scaled",
        "Close_VIX",
    ] 

    X_full = df_combined[feature_cols].values
    probs_matrix = model.predict_proba(X_full)

    # make matrix of probabilities for each regime in each day
    regimes = pd.DataFrame(
    probs_matrix,
    index=df_combined["Date"].dt.normalize(),
    columns=[f"regime{i}" for i in range(model.n_components)]
    )
    regimes.index = regimes.index.tz_localize(None)

    current_regime_probs = regimes.iloc[-1].values
    
    stock_features, log_returns = get_asset_data(stocks)
    print(f"{len(log_returns.index)}\n{print(len(regimes.index))}\n{print(len(log_returns.index.difference(regimes.index)))}\n{log_returns.index.difference(regimes.index)}")
    # todo: either forward fill regimes missing in regimes index or drop the differing ones 
    # get expected return and volatility of assets given current regime probs, covariance of assets under each regime. we use weighted returns, vol and covariance
    # result_1, result_2, result_3 = get_asset_behaviour(data_combined, regimes)
    # each r_i will be list of asset_returns, asset_vols, asset_covars under regime i  
    # get expected return and volatility of portfolio given current regime probs. expected return = weighted average of returns under each regime, 
    # port_return, port_vol = get_portfolio_statistics(adjusted_returns, current_regime_probs, weights, rf) 
    # optimise_weights
    # return ... 

if __name__ == "__main__":
    main()
