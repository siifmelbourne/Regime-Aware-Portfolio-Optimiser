import yfinance as yf 
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "market_regime_model"))
from hmm_model import main as run_hmm

def combine_log_returns(stocks):
    """
    Combines log returns of stocks in portfolio into one dataframe.

    Args:
        stocks (list): List of stock ticker symbol.

    Returns:
        returns_combined (pd.DataFrame): A DataFrame containing the daily log returns for each stock.
    """

    returns_combined = pd.DataFrame()
    for stock in stocks: 
        df = pd.read_csv(f"regime_portfolio_optimiser/data/{stock}_historical_data_engineered.csv")
        returns_combined[stock] = df["Daily Log Return"] 

    returns_combined.index = pd.to_datetime(df["Date"])
    return returns_combined

def normalise_index(df1, df2, method="ffill"): 
    """
    Normalise index of datetime indexes

    Args:
        df1 (pd.DataFrame), df2 (pd.DataFrame): Pandas dataframes containing datetime index. 
        method (str): Method of normalisation. 
                reindex - Drops differing dates
                ffill   - Forward fills dates (backward fills missing values which lead)

    Returns:
        df1_normalised (pd.DataFrame), df2_normalised (pd.DataFrame): Pandas dataframes with matching indexes
    """

    # normalise datetime index conventions
    df1_normalised = df1.copy()
    df2_normalised = df2.copy()
    df1_normalised.index = df1.index.tz_localize(None).normalize() 
    df2_normalised.index = df2.index.tz_localize(None).normalize()

    # deal with index mismatches based on method. just realised filling data is probably not viable for returns, probably reindex 
    if (method == "reindex"):
        df1_normalised, df2_normalised = df1_normalised.align(df2_normalised, join="inner", axis=0)
    elif (method == "ffill"):
        max_fill = 3
        df1_normalised, df2_normalised = df1_normalised.align(df2_normalised, join="outer", axis=0)
        df1_normalised = df1_normalised.ffill(axis=0, limit=max_fill).bfill(axis=0, limit=max_fill)
        df2_normalised = df2_normalised.ffill(axis=0, limit=max_fill).bfill(axis=0, limit=max_fill)
    else:
        raise ValueError(f"Method should be ffill or reindex, got {method}")

    valid = df1_normalised.notna().all(axis=1) & df2_normalised.notna().all(axis=1) # filter for valid fills  
    df1_normalised = df1_normalised[valid]
    df2_normalised = df2_normalised[valid]

    return df1_normalised, df2_normalised

def get_asset_behaviour(returns, regimes):
    """
    Computes asset behaviour (Expected Return, Volatility, Covariance of Assets) under each regime. 
    
    Args:
        returns (pd.DataFrame): dataframe of log returns (date x returns for each asset)
        regimes (pd.DataFrame): dataframe of regime probabilities (date x probabilities for each regime)
 
    Returns:
        regime_behaviour (dict[str, RegimeAssetBehaviour]): Dictionary of regimes which contain asset behaviour.
    """ 

    regime_behaviour = {}
    for regime_name, regime_probs in regimes.items():
        regime_behaviour[regime_name] = get_asset_behaviour_in_regime(returns, regime_probs)

    return regime_behaviour

def get_asset_behaviour_in_regime(returns, regime):
    """
    Computes historical asset behaviour (Expected Return, Volatility, Covariance of Assets - expressed in daily terms) under a specific regime, weighted based on regime probability.
    
    Args:
        returns (pd.DataFrame): dataframe of log returns (date x returns for each asset)
        regime (pd.Series): series of regime probabilities (date x probabilities for each regime)
 
    Returns:
        asset_behaviour (dict): dictionary of ev, vol and covar
            ev (pd.Series): historic expected return of assets under regime
            vol (pd.Series): historic volatility of assets under regime 
            covar (pd.DataFrame): historic covariance of assets under regime 
    """

    total_proba = regime.sum()
    normalised_weights = regime/total_proba
    ev = returns.mul(regime, axis=0).sum()/total_proba
    deviations = returns-ev
    weighted_deviations = deviations.mul(normalised_weights, axis=0)
    covar = deviations.T.dot(weighted_deviations)
    var = pd.Series(np.diag(covar), index=ev.index)
    vol = var ** 0.5

    return {"ev": ev, "vol": vol, "covar": covar}

def get_portfolio_statistics(asset_behaviour, current_regime_probs, weights):
    """
    Calculates portfolio statistics (expected return, variance, vol) given weights.
    
    Args:
        asset_behaviour (dict): dictionary containing asset behaviour (expected returns, vol, covar) under each regime
        current_regime_probs (np.ndarray): vector of regime probabilities today
        weights (pd.Series): series of weights, indexed with stock ticker
 
    Returns:
        expected_return
        var
        vol
    """

    regime_returns = []
    regime_vars = []
    for behaviour in asset_behaviour.values():
        regime_returns.append((behaviour["ev"]*weights).sum())
        regime_vars.append(weights @ behaviour["covar"] @ weights)
    regime_returns = pd.Series(regime_returns)
    regime_vars = pd.Series(regime_vars)

    expected_return = sum(
    prob * regime_return #behaviour["ev"]: series of expected return for each stock, weights: series of weights for each stock
    for prob, regime_return in zip(current_regime_probs, regime_returns)
    )

    deviation = regime_returns - expected_return
    var_of_expectation = (current_regime_probs * (deviation ** 2)).sum()
    # print(var_of_expectation) 
    # is 0 when hmm_model is certain we are under one regime 
    expectation_of_var = (regime_vars * current_regime_probs).sum()
    # print(expectation_of_var)
    total_var = var_of_expectation + expectation_of_var
    # print(total_var)

    ev_annualised = expected_return*(252)
    var_annualised = total_var*(252)
    vol_annualised = (total_var)**0.5 * (252**0.5)
    print(f"\n-------------------------------------\nexpected return: {ev_annualised}")
    print(f"-------------------------------------\nvariance: {var_annualised}")
    print(f"-------------------------------------\nvolatility: {vol_annualised}\n-------------------------------------")

    return ev_annualised, var_annualised, vol_annualised

def optimise_weights(returns, regimes, method='CVaR', rf=0.05): 
    ...

def main():
    stocks = ['AAPL', 'NVDA', 'MSFT']
    try: 
        weights = pd.Series([.7, .2, .1], index=stocks)
    except Exception as e:
        print("Invalid stocks/weights input")
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

    # get matrix of probabilities from market regime model, convert to pandas dataframe
    X_full = df_combined[feature_cols].values
    probs_matrix = model.predict_proba(X_full)
    regimes = pd.DataFrame(
    probs_matrix,
    index=df_combined["Date"], # normalise datetime index to match returns datetime index 
    columns=[f"regime{i}" for i in range(model.n_components)]
    )
    current_regime_probs = regimes.iloc[-1].values
    
    log_returns = combine_log_returns(stocks)
    log_returns, regimes = normalise_index(log_returns, regimes, method="reindex")

    asset_behaviours = get_asset_behaviour(log_returns, regimes)
    # print(asset_behaviours["regime0"]["covar"])

    get_portfolio_statistics(asset_behaviours, current_regime_probs, weights)

    # todo: 
    # make optimising weight function 
    # return ... 

    # to fix: got rid of stock info when separating downloading data from main file, fragment of sotck info in feature eng py file 
    # stocks and weights are in both regime_port_optimiser and feature eng and data download umm
    # make an exponential weighting in get_asset_behaviour_in_regime

if __name__ == "__main__":
    main()
