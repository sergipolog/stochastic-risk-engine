'''
This module contains functions to optimize a given portfolio.
'''

import numpy as np
import pandas as pd
from scipy.optimize import minimize

def generate_simulated_returns(mean_returns: np.ndarray, cov_matrix: np.ndarray, M: int, df: int = 4) -> np.ndarray:
	'''
	Generates 1-day simulated return multipliers using a t-Student distribution.
	This function utilizes a stochastic Monte Carlo engine, combining a multivariate normal distribution 
	with a chi-square distribution to model extreme market shocks (fat tails) that standard normal 
	distributions fail to capture.
	
		Parameters:
			mean_returns (np.ndarray): A 1D array containing the historical mean logarithmic returns for each asset.
			cov_matrix (np.ndarray): The 2D historical covariance matrix representing asset volatility and correlation.
			M (int): The number of Monte Carlo scenarios to simulate (e.g., 200,000).
			df (int): Degrees of freedom for the t-distribution. Lower values dictate fatter tails (more extreme outliers). Defaults to 4.

		Returns:
			np.ndarray: A 2D array of simulated daily return multipliers (e.g., 1.02 for a +2% return) across M scenarios for all assets.
	'''
	num_stocks = len(mean_returns)
	Z = np.random.multivariate_normal(np.zeros(num_stocks), cov_matrix, size=(M, 1))
	U = np.random.chisquare(df, size=(M, 1, 1))
	
	simulated_log_returns = mean_returns + Z * np.sqrt(df / U)
	
	# Returns the multipliers (e.g., 1.02 for +2%, 0.98 for -2%)
	return np.exp(np.sum(simulated_log_returns, axis=1))


def calculate_cvar_with_penalty(weights: np.ndarray, simulated_multipliers: np.ndarray, P: float, current_weights: np.ndarray, penalty: float) -> float:
	'''
	Calculates the 95% CVaR applying a turnover given the current weights and the new ones and applies a penalty.
	The objective function transforms the portfolio weights into absolute Euro values, identifies the 
	worst 5% of simulated outcomes, and averages them. It then calculates the absolute turnover required 
	to shift from the current weights to the proposed weights, adding a monetary penalty to strictly 
	limit unnecessary rebalancing.

		Parameters:
			weights (np.ndarray): A 1D array of the proposed target asset weights being evaluated by the optimizer.
			simulated_multipliers (np.ndarray): The 2D array of Monte Carlo return multipliers generated for the market.
			P (float): Total portfolio capital in Euros.
			current_weights (np.ndarray): A 1D array representing the starting asset allocation before optimization.
			penalty (float): The monetary scalar for the transaction fee (derived from the broker commission rate).

		Returns:
			float: The objective score combining the base 95% CVaR and the transaction penalty.
	'''

	portfolio_final_values = np.sum((P * weights) * simulated_multipliers, axis=1)
	profit = portfolio_final_values - P
	
	var_95 = np.percentile(profit, 5)
	cvar_95 = profit[profit <= var_95].mean()
	base_cvar = np.abs(cvar_95)

	turnover = np.sum(np.abs(weights - current_weights))

	return base_cvar + (penalty* turnover)


def optimize_portfolio_weights_with_penalty(simulated_multipliers: np.ndarray, P: float, num_stocks: int, current_weights: np.ndarray, penalty: float) -> np.ndarray:
	'''
	Finds the optimal asset allocation that minimizes tail risk.

	Utilizes SciPy's Sequential Least Squares Programming (SLSQP) to navigate the stochastic return space. 
	The solver is strictly constrained to a long-only position (no short selling) and requires all output 
	weights to sum exactly to 1.0 (100% capital allocation).

		Parameters:
			simulated_multipliers (np.ndarray): The pre-computed 2D matrix of stochastic market scenarios.
			P (float): Total portfolio capital in Euros.
			num_stocks (int): The total number of unique assets in the portfolio.
			current_weights (np.ndarray): The starting allocation percentages used to calculate the turnover penalty.
			penalty (float): The monetary scalar for the transaction fee (derived from the broker commission rate).

		Returns:
			np.ndarray: A 1D array containing the mathematically optimized target weights for the portfolio.
	'''
	
	# Start with equal weights (20% each for 5 stocks)
	init_guess = current_weights.copy()
	
	# No short selling (weights must be between 0 and 1)
	bounds = tuple((0.0, 1.0) for _ in range(num_stocks))
	
	# The sum of all weights must exactly equal 1.0
	constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
	
	# Optimizer
	optimized_result = minimize(
		fun=calculate_cvar_with_penalty,
		x0=init_guess,
		args=(simulated_multipliers, P, current_weights, penalty),
		method='SLSQP',
		bounds=bounds,
		constraints=constraints
	)

	return optimized_result.x