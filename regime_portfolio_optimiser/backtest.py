import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from data_download import download
from data_utils import combine_log_returns, normalise_index
from feature_eng import feature_eng
from parse_args import parse_args, validate_args
from main import run_optimiser

REBALANCE_DAYS = 21
START_DATE = pd.to_datetime('2022-01-01')
END_DATE = pd.Timestamp.today().normalize().tz_localize(None)

args = parse_args()
stocks, weights = validate_args(args.stocks, args.weights)

download(stocks)
feature_eng(stocks)

benchmark_stocks = ['VAS.AX']

download(benchmark_stocks)
feature_eng(benchmark_stocks)
benchmark_log_returns = combine_log_returns(benchmark_stocks)

idx = benchmark_log_returns.index[benchmark_log_returns.index >= START_DATE]
rebalance_dates = idx[::REBALANCE_DAYS]

stock_log_returns = combine_log_returns(stocks)

portfolio_mvo_log_returns = pd.Series(index=stock_log_returns.index)
portfolio_cvar_log_returns = pd.Series(index=stock_log_returns.index)
for k, d in enumerate(rebalance_dates):
    end_next = rebalance_dates[k + 1] if k + 1 < len(rebalance_dates) else idx[-1]

    try:
        optimal_mvo_weights, optimal_cvar_weights, _ = run_optimiser(stocks, weights, end_date=d)
    except ValueError as e:
        print(f"skip {d}: {e}")
        continue

    period = stock_log_returns[(stock_log_returns.index > d) & (stock_log_returns.index <= end_next)]
    portfolio_cvar_log_returns.loc[period.index] = period @ optimal_cvar_weights 
    portfolio_mvo_log_returns.loc[period.index] = period @ optimal_mvo_weights 

portfolio_cvar_log_returns = portfolio_cvar_log_returns.dropna()
portfolio_mvo_log_returns = portfolio_mvo_log_returns.dropna()

portfolio_mvo_log_returns, benchmark_log_returns = normalise_index(portfolio_mvo_log_returns, benchmark_log_returns)
portfolio_cvar_log_returns, benchmark_log_returns = normalise_index(portfolio_cvar_log_returns, benchmark_log_returns)

portfolio_mvo_cum_returns = portfolio_mvo_log_returns.cumsum()
portfolio_cvar_cum_returns = portfolio_cvar_log_returns.cumsum()
benchmark_cum_returns = benchmark_log_returns.cumsum()

plt.figure(figsize=(10, 5))
plt.plot(portfolio_mvo_cum_returns, label=f"portfolio - MVO", linewidth=2)
plt.plot(portfolio_cvar_cum_returns, label=f"portfolio - CVaR", linewidth=2)
for c in benchmark_cum_returns:
    plt.plot(benchmark_cum_returns[c], label=c)
plt.legend()
plt.ylabel("Cumulative return (log)")
plt.title(f"cum_returns: {START_DATE}-{END_DATE}, rebalance: {REBALANCE_DAYS}")
plt.savefig(f"regime_portfolio_optimiser/backtest_results/backtest_{stocks}.png", dpi=300, bbox_inches="tight")