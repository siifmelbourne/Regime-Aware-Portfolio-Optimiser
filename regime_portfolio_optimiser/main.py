import pandas as pd
import sys
import os

from parse_args import parse_args, validate_args
from portfolio import RegimePortfolio
from data_utils import combine_log_returns, normalise_index
from regime import get_asset_behaviour
from data_download import download
from feature_eng import feature_eng

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "market_regime_model"))
from hmm_model import main as run_hmm

def run_optimiser(stocks, weights, start_date=None, end_date=None):
    if start_date is None:
        start_date = pd.to_datetime('2010-02-10')
    else:
        try: 
            start_date = pd.to_datetime(start_date)
        except (TypeError, ValueError) as e:
            raise ValueError("end_date must be a valid date") from e

    if end_date is None:
        end_date = pd.Timestamp.today().normalize()
    else:
        try:
            end_date = pd.to_datetime(end_date)
        except (TypeError, ValueError) as e:
            raise ValueError("end_date must be a valid date") from e

    start_date = start_date.tz_localize(None)    
    end_date = end_date.tz_localize(None)

    if start_date > end_date:
        raise ValueError("Start date should be before end date")

    download(stocks)
    feature_eng(stocks)

    # connect market regime model 
    df_combined, model = run_hmm(end_date)

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
    log_returns = log_returns[(log_returns.index <= end_date) & (log_returns.index >= start_date)]

    log_returns, regimes = normalise_index(log_returns, regimes, method="reindex")
    asset_behaviours = get_asset_behaviour(log_returns, regimes)

    regime_portfolio = RegimePortfolio(asset_behaviours, current_regime_probs, log_returns)

    # Print optimal portfolio statistics under each method 
    # Note: CVaR lambda and MVO lambda should be different
    # since behaviour is quadratic in w for MVO, non-linear for cvar (CVaR is much larger than mean, causes weird behaviour)
    optimal_mvo_weights = regime_portfolio.optimise(weights, method='MVO')
    optimal_cvar_weights = regime_portfolio.optimise(weights, method='CVaR', L=0.01)

    return optimal_mvo_weights, optimal_cvar_weights, regime_portfolio

def main():
    args = parse_args()
    stocks, weights = validate_args(args.stocks, args.weights)
    optimal_mvo, optimal_cvar, regime_portfolio = run_optimiser(stocks, weights)

    print("\n\nCurrent Portfolio Statistics:\n\n")
    regime_portfolio.print_statistics(weights)

    print("\n\nOptimal Portfolio Statistics:")
    print("\n\nMethod: MVO")
    regime_portfolio.print_statistics(optimal_mvo)

    print("\n\nMethod: CVaR")
    regime_portfolio.print_statistics(optimal_cvar)

if __name__ == "__main__":
    main()
