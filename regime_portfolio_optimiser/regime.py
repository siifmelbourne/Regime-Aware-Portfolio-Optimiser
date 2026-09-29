import pandas as pd
import numpy as np

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
