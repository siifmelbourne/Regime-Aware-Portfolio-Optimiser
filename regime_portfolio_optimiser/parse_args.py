import argparse 
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Run regime-aware portfolio optimiser")
    parser.add_argument(
        "--stocks",
        nargs="+",
        default=None,
        help="List of stock tickers, space-separated, e.g. --stocks AAPL MSFT NVDA"
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help="Portfolio weights matching --stocks order (defaults to equal weight if omitted)"
    )
    return parser.parse_args()

def validate_args(stocks, weights):
    if not stocks:
        raise ValueError("At least one stock must be provided via --stocks")

    if weights:
        if len(weights) != len(stocks):
            raise ValueError("Number of weights must match number of stocks")

        weights = pd.Series(weights, index=stocks)
        if any(weights < 0):
            raise ValueError("Must be long-only portfolio")
        if (sum(weights) != 1):
            # normalise so weights sum to 1
            weights = weights / sum(weights)
    else:
        weights = pd.Series([1 / len(stocks)] * len(stocks), index=stocks)

    if len(weights) != len(stocks):
        raise ValueError("Number of weights must match number of stocks")

    return stocks, weights