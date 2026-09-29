# Regime-Aware Portfolio Optimiser

Constructs portfolio weights using regime probabilities from a Hidden Markov Model market regime detector, subject to long-only constraints.

## Installation

Clone `regime_portfolio_optimiser` and install dependencies:

       git clone <https://github.com/siifmelbourne/Regime-Aware-Portfolio-Optimiser>
       cd Regime-Aware-Portfolio-Optimiser
       pip install -r requirements.txt

## Method

**Regime-conditional asset statistics.** For each regime $j$, let $p_{j,t} = p(\text{regime}_j \mid \text{time } t)$ and let $r_{i,t}$ be the return of asset $i$ at time $t$.

- **Expected return**
$$
\mu_{i,j} = E[\text{asset}_i \mid \text{regime}_j] = \frac{\sum_t p_{j,t} \, r_{i,t}}{\sum_t p_{j,t}}
$$

- **Variance**
$$
\sigma^2_{i,j} = \frac{\sum_t p_{j,t} \left( r_{i,t} - \mu_{i,j} \right)^2}{\sum_t p_{j,t}}
$$

- **Covariance**
$$
\mathrm{Cov}_j[\text{asset}_i, \text{asset}_k] = \frac{\sum_t p_{j,t} \left( r_{i,t} - \mu_{i,j} \right)\left( r_{k,t} - \mu_{k,j} \right)}{\sum_t p_{j,t}}
$$

**Portfolio statistics.** 

Overall expected return is the average of the regime-conditional returns, weighted by the current regime probabilities. Let $p_k$ be current probability of regime $k$, $w$ be weight of stocks, $\mu_k$ be list of expected return of stocks in regime $k$. Then $w^\top \mu_k$ is the expected portfolio return in regime $k$. 

$$
\mu_p = \sum_{k=1}^{K} p_k \, w^\top \mu_k
$$
Total variance (law of total variance):

$$
\sigma_p^2 = \mathrm{Var}(R) = E\left[\mathrm{Var}(R \mid K)\right] + \mathrm{Var}\left(E[R \mid K]\right)
$$

$$
            = \sum_{k=1}^{K} p_k \, w^\top \Sigma_k w \;+\; \sum_{k=1}^{K} p_k \left( w^\top \mu_k - \mu_p \right)^2
$$


**Optimisation.**
- Mean-Variance Optimisation (MVO): 
$$
\max_{w} \;\; (\mu_p(w) - \lambda \, \sigma_p^2(w))
\qquad \text{s.t.} \quad \sum_i w_i = 1, \;\; w_i \ge 0
$$

Adjusting lambda adjusts how much volatility matters in portfolio. 

- Conditional Value-at-Risk (CVaR):

Using 95% CVaR, take the average of $5$% worst losses (as positive number). 
$\mathcal{W}_\alpha(w)$: the set of days $t$ with the $\lceil \alpha T \rceil$ lowest portfolio returns. Dependent on $w$, since different weights change which days are worst.

$$
\mathrm{CVaR}_\alpha(w) = -\frac{1}{|\mathcal{W}_\alpha(w)|} \sum_{t \in \mathcal{W}_\alpha(w)} w^\top r_t(w)
$$

$$
\min_{w} \;\; \lambda \, \mathrm{CVaR}_\alpha(w) - \mu_p(w)
\qquad \text{s.t.} \quad \sum_i w_i = 1, \;\; w_i \ge 0
$$

## Limitations / TODO

- **Risk-free rate:** fixed at 0.05. Needs a time series of rf so Sharpe ratios are accurate and rf can be plotted over time.
- **Regime model coverage:** the regime model is based mainly on VAS (ASX), so should be careful with US heavy portfolios.
- **Expanding-window backtest:** the expanding window can include regime data that is too outdated to be trusted. The same issue affects asset behaviour statistics, which use the full historical window. Will test a rolling window instead. 
- **CVaR:** returns near equal-weight portfolios on most iterations. Possible causes:
  - CVaR iterates over full historic window instead of being regime aware
  - Test other penalties instead of lambda=0.01, other percentages? e.g. 97% CVaR 
  - Check SLSQP stopping early bug
- **Industry constraints** Add industry constraints into optimisation pipeline. 
- **Backtesting:** needs to be run on historical data from the investments team. Need to add performance against the original weighting in the matplotlib graph.
- **documentation** add more documentation regarding max drawdown, sharpe ratio etc.