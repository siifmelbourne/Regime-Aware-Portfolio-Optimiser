import yfinance as yf 
import pandas as pd
import numpy as np
import sys
import os
import scipy

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "market_regime_model"))
from hmm_model import main as run_hmm

def combine_log_returns(stocks):
    """
    Combines log returns of stocks in portfolio into one dataframe.

    Args:
        stocks: list
            List of stock ticker symbol.

    Returns:
        returns_combined: pd.DataFrame
            A DataFrame containing the daily log returns for each stock.
    """

    returns_combined = pd.DataFrame()
    for stock in stocks: 
        try: 
            df = pd.read_csv(f"regime_portfolio_optimiser/data/{stock}_historical_data_engineered.csv")
            returns_combined[stock] = df["Daily Log Return"] 
        except Exception as e: 
            print(f"Error fetching data for {stock}: {e}")
            raise

    returns_combined.index = pd.to_datetime(df["Date"])
    return returns_combined

def normalise_index(df1, df2, method="ffill"): 
    """
    Normalise index of datetime indexes

    Args:
        df1: pd.DataFrame
            Pandas dataframes containing datetime index. 
        df2: pd.DataFrame
            Pandas dataframes containing datetime index. 
        method: str
            Method of normalisation. 
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
        returns: pd.DataFrame
            dataframe of log returns (date x returns for each asset)
        regime: pd.Series
            series of regime probabilities (date x probabilities for each regime)
 
    Returns:
        asset_behaviour: dict
            dictionary of ev, vol and covar
                ev: pd.Series
                    historic expected return of assets under regime
                vol: pd.Series
                    historic volatility of assets under regime 
                covar: pd.DataFrame
                    historic covariance of assets under regime 
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

def get_portfolio_statistics(asset_behaviour, current_regime_probs, weights, returns_combined, rf=0.05):
    """
    Calculates portfolio statistics (expected return, variance, vol) given weights.
    
    Args:
        asset_behaviour: dict
            dictionary containing asset behaviour (expected returns, vol, covar) under each regime
        current_regime_probs: np.array
            vector of regime probabilities today
        weights: pd.Series
            series of weights, indexed with stock ticker
 
    Returns:
        ev_annualised: float
            annualised expected return of portfolio
        var_annualised: float
            annualised variance of portfolio
        vol_annualised: float
            annualised standard deviation (volatility) of portfolio
        sharpe_annualised: float
            annualised sharpe ratio of portfolio
        max_drawdown: float
            maximum drawdown of portfolio
    """

    regime_returns = []
    regime_vars = []
    for behaviour in asset_behaviour.values():
        # behaviour["ev"]: series of expected return for each stock in regime, weights: series of weights for each stock
        # regime_returns: list of expected return for each regime 
        regime_returns.append((behaviour["ev"]*weights).sum()) 
        regime_vars.append(weights @ behaviour["covar"] @ weights)
    regime_returns = pd.Series(regime_returns)
    regime_vars = pd.Series(regime_vars)

    expected_return = sum(
    prob * regime_return 
    for prob, regime_return in zip(current_regime_probs, regime_returns)
    )

    deviation = regime_returns - expected_return
    var_of_expectation = (current_regime_probs * (deviation ** 2)).sum()
    # is 0 when hmm_model is certain we are under one regime 
    expectation_of_var = (regime_vars * current_regime_probs).sum()
    total_var = var_of_expectation + expectation_of_var

    ev_annualised = expected_return*(252)
    var_annualised = total_var*(252)
    vol_annualised = (total_var)**0.5 * (252**0.5)
    sharpe_annualised = (ev_annualised - rf) / vol_annualised
    
    weighted_returns = returns_combined @ weights
    portfolio_value = (1 + weighted_returns).cumprod()
    running_peak = portfolio_value.cummax()
    drawdown = portfolio_value / running_peak - 1
    max_drawdown = -drawdown.min()

    return ev_annualised, var_annualised, vol_annualised, sharpe_annualised, max_drawdown

def optimise_weights(asset_behaviours, current_regime_probs, initial_guess, returns_combined, L=np.pi, method='MVO'): 
    '''
    Calculates the optimal weights for a portfolio given a certain optimisation method. 
    
    Args:
        asset_behaviour: dict
            dictionary containing asset behaviour (expected returns, vol, covar) under each regime
        current_regime_probs: np.array
            current regime probabilities
        initial_guess: np.Array
            guess of optimal weights 
        method: str
            method of optimisation, 'MVO' or 'CVaR'
        L: float
            constant lambda used in MVO method, determining importance of variance minimisation in portfolio 

    Returns:
        optimal_weights: np.array
            array of weights which optimise portfolio given method. 
    '''

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0, 1) for _ in range(len(initial_guess))]
    
    if (method == 'MVO'): 

        #minimise the negative of the mean - L*variance to get maximum of mean - L*variance
        def negative_utility_mvo(weights):
            mean_annualised, var_annualised, _, _, _ = get_portfolio_statistics(asset_behaviours, current_regime_probs, weights, returns_combined)
            return L*var_annualised-mean_annualised
        
        optimal_weights = scipy.optimize.minimize(negative_utility_mvo, x0=initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

    elif (method == 'CVaR'): 

        n = int(np.ceil(returns_combined.shape[0]*0.05))
        def negative_utility_cvar(weights):
            mean_annualised, _, _, _, _ = get_portfolio_statistics(asset_behaviours, current_regime_probs, weights, returns_combined)
            port_cvar = -(returns_combined @ weights).nsmallest(n).mean()

            return L*port_cvar-mean_annualised/252

        optimal_weights = scipy.optimize.minimize(negative_utility_cvar, x0=initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

    if not optimal_weights.success:
        raise RuntimeError(f"Optimisation failed: {optimal_weights.message}")

    return optimal_weights.x


def print_portfolio_statistics(asset_behaviours, current_regime_probs, weights, returns_combined):
    '''
    Prints portfolio statistics.

    Args:
        asset_behaviours: dict
            Asset behaviour under regimes
        current_regime_probs: np.array
            current regime probabilities
        weights: np.array
            weights of each stock.
    '''

    ev_annualised, var_annualised, vol_annualised, sharpe_annualised, max_drawdown = get_portfolio_statistics(asset_behaviours, current_regime_probs, weights, returns_combined)

    print(f"*************************************\nportfolio statistics under weights\n{weights}")
    print(f"*************************************\nexpected return: {ev_annualised}")
    print(f"-------------------------------------\nvariance: {var_annualised}")
    print(f"-------------------------------------\nvolatility: {vol_annualised}")
    print(f"-------------------------------------\nsharpe ratio: {sharpe_annualised}")
    print(f"-------------------------------------\nmaximum drawdown: {max_drawdown}")
    print("-------------------------------------\n")
    return

def main():
    stocks = ['AAPL', 'NVDA', 'MSFT', 'GOOGL', 'AMZN', 'JPM']
    try: 
        weights = pd.Series([.1, .2, .1, .15, .15, .3], index=stocks)
    except Exception as e:
        print("Invalid stocks/weights input")
        return -1

    # obtain probability matrix from market regime model 
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

    # Print current portfolio statistics
    print("\n\nCurrent Portfolio Statistics:\n\n")
    print_portfolio_statistics(asset_behaviours, current_regime_probs, weights, log_returns)

    # Print optimal portfolio statistics under each method 
    # Note: CVaR method is more invariant for lambda compared to MVO method
    # since behaviour is quadratic in w for MVO, non-linear for cvar
    optimal_mvo_weights = optimise_weights(asset_behaviours, current_regime_probs, weights, log_returns, method='MVO')
    print("\n\nOptimal Portfolio Statistics:")
    print("\n\nMethod: MVO")
    print_portfolio_statistics(asset_behaviours, current_regime_probs, optimal_mvo_weights, log_returns)

    optimal_cvar_weights = optimise_weights(asset_behaviours, current_regime_probs, weights, log_returns, method='CVaR')
    print("\n\nMethod: CVaR")
    print_portfolio_statistics(asset_behaviours, current_regime_probs, optimal_cvar_weights, log_returns)

if __name__ == "__main__":
    main()
