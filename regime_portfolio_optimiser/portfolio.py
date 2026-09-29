import pandas as pd
import numpy as np
import scipy

class RegimePortfolio:
    def __init__(self, asset_behaviours, current_regime_probs, returns_combined, rf=0.05):
        self.asset_behaviours = asset_behaviours
        self.current_regime_probs = current_regime_probs
        self.returns_combined = returns_combined
        self.rf = rf 

    def core_statistics(self, weights): 
        '''
        Calculates core statistics - moments (expected return, variance), given weights.

        Args:
            weights: np.array
                array of weights for each stock

        Returns: 
            ev: float
                expected return
            var: float
                variance
        '''
        regime_returns = []
        regime_vars = []
        for behaviour in self.asset_behaviours.values():
            # behaviour["ev"]: series of expected return for each stock in regime, weights: series of weights for each stock
            # regime_returns: list of expected return for each regime 
            regime_returns.append((behaviour["ev"]*weights).sum()) 
            regime_vars.append(weights @ behaviour["covar"] @ weights)
        regime_returns = pd.Series(regime_returns)
        regime_vars = pd.Series(regime_vars)

        expected_return = sum(
        prob * regime_return 
        for prob, regime_return in zip(self.current_regime_probs, regime_returns)
        )

        deviation = regime_returns - expected_return

        # variance of expectation is 0 when hmm_model is certain we are under one regime 
        var_of_expectation = (self.current_regime_probs * (deviation ** 2)).sum()

        expectation_of_var = (regime_vars * self.current_regime_probs).sum()
        total_var = var_of_expectation + expectation_of_var

        ev_annualised = expected_return*(252)
        var_annualised = total_var*(252)

        return ev_annualised, var_annualised
    
    def aux_statistics(self, weights):
        '''
        Calculates portfolio statistics outside of mean, variance (volatility, sharpe ratio, max drawdown) given weights.
        Args:
            weights: np.array
                array of weights for each stock

        Returns: 
            ev: float
                expected return
            var: float
                variance
        '''
        
        ev_annualised, var_annualised = self.core_statistics(weights)
        vol_annualised = (var_annualised)**0.5
        sharpe_annualised = (ev_annualised - self.rf) / vol_annualised

        weighted_returns = self.returns_combined @ weights
        portfolio_value = (1 + weighted_returns).cumprod()
        running_peak = portfolio_value.cummax()
        drawdown = portfolio_value / running_peak - 1
        max_drawdown = -drawdown.min()

        return vol_annualised, sharpe_annualised, max_drawdown

    def optimise(self, initial_guess, L=np.pi, method='MVO'): 
        '''
        Calculates the optimal weights for a portfolio given a certain optimisation method. 
        
        Args:
            initial_guess: np.Array
                guess of optimal weights 
            L: float
                constant lambda used in MVO method, determining importance of variance minimisation in portfolio 
            method: str
                method of optimisation, 'MVO' or 'CVaR'

        Returns:
            optimal_weights: np.array
                array of weights which optimise portfolio given method. 
        '''

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0, 1) for _ in range(len(initial_guess))]

        if (method == 'MVO'): 

            #minimise the negative of the mean - L*variance to get maximum of mean - L*variance
            def negative_utility_mvo(weights):
                mean_annualised, var_annualised = self.core_statistics(weights)
                return L*var_annualised-mean_annualised
            
            optimal_weights = scipy.optimize.minimize(negative_utility_mvo, x0=initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

        elif (method == 'CVaR'): 

            n = int(np.ceil(self.returns_combined.shape[0]*0.05))
            def negative_utility_cvar(weights):
                mean_annualised, _ = self.core_statistics(weights)
                port_cvar = -(self.returns_combined @ weights).nsmallest(n).mean()

                return L*port_cvar-mean_annualised/252

            optimal_weights = scipy.optimize.minimize(negative_utility_cvar, x0=initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

        if not optimal_weights.success:
            raise RuntimeError(f"Optimisation failed: {optimal_weights.message}")

        return optimal_weights.x

    def print_statistics(self, weights):
        '''
        Prints portfolio statistics.

        Args:
            weights: np.array
                weights of each stock.
        '''

        ev_annualised, var_annualised = self.core_statistics(weights)
        vol_annualised, sharpe_annualised, max_drawdown = self.aux_statistics(weights)

        print(f"*************************************\nportfolio statistics under weights\n{weights}")
        print(f"*************************************\nexpected return: {ev_annualised}")
        print(f"-------------------------------------\nvariance: {var_annualised}")
        print(f"-------------------------------------\nvolatility: {vol_annualised}")
        print(f"-------------------------------------\nsharpe ratio: {sharpe_annualised}")
        print(f"-------------------------------------\nmaximum drawdown: {max_drawdown}")
        print("-------------------------------------\n")
        return
