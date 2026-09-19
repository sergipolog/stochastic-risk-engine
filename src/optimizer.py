'''
This module contains functions to optimize a given portfolio.
'''

import numpy as np
import pandas as pd

def generate_simulated_returns(mean_returns: np.ndarray, cov_matrix: np.ndarray, M: int, df: int = 4) -> np.ndarray:
	'''Generates 1-day simulated return multipliers using a t-Student distribution.'''
	num_stocks = len(mean_returns)
	Z = np.random.multivariate_normal(np.zeros(num_stocks), cov_matrix, size=(M, 1))
	U = np.random.chisquare(df, size=(M, 1, 1))
	
	simulated_log_returns = mean_returns + Z * np.sqrt(df / U)
	
	# Returns the multipliers (e.g., 1.02 for +2%, 0.98 for -2%)
	return np.exp(np.sum(simulated_log_returns, axis=1))

def calculate_cvar(weights: np.ndarray, simulated_multipliers: np.ndarray, P: float) -> float:
	'''Calculates the 95% CVaR given a set of weights and simulated market conditions.'''

	portfolio_final_values = np.sum((P * weights) * simulated_multipliers, axis=1)
	profit = portfolio_final_values - P
	
	var_95 = np.percentile(profit, 5)
	cvar_95 = profit[profit <= var_95].mean()
	
	# Returns the absolute value.
	# The optimizer's goal is to drive this positive number as close to 0 as possible.
	return abs(cvar_95)

from scipy.optimize import minimize

def optimize_portfolio_weights(simulated_multipliers: np.ndarray, P: float, num_stocks: int) -> np.ndarray:
	'''Finds the optimal asset weights to minimize the Expected Shortfall (CVaR).'''
	
	# Start with equal weights (20% each for 5 stocks)
	init_guess = np.ones(num_stocks) / num_stocks
	
	# No short selling (weights must be between 0 and 1)
	bounds = tuple((0.0, 1.0) for _ in range(num_stocks))
	
	# The sum of all weights must exactly equal 1.0
	constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
	
	# Optimizer
	optimized_result = minimize(
		fun=calculate_cvar,
		x0=init_guess,
		args=(simulated_multipliers, P),
		method='SLSQP',
		bounds=bounds,
		constraints=constraints
	)
	
	# Returns the array of optimal weights
	return optimized_result.x

def rolling_engine_optimized(returns: pd.DataFrame, window_size: int, M: int, P: float, df: int = 4) -> pd.DataFrame:
	'''
	Performs a rolling 1-day VaR/CVaR backtest utilizing an EWMA covariance matrix 
	and daily algorithmic portfolio optimization.
	'''
	n = len(returns)
	num_stocks = returns.shape[1]
	results = []

	print(f"Initiating optimized rolling backtest over {n - window_size} days...")

	for i in range(window_size, n):
		
		historical_slice = returns.iloc[i - window_size : i]
		
		# Volatility Calibration (EWMA)
		window_mean = historical_slice.mean().values
		# ewm().cov() returns a stacked MultiIndex DataFrame; we slice the final num_stocks rows for the most recent matrix
		window_cov_matrix = historical_slice.ewm(span=30).cov().iloc[-num_stocks:, :].values 
		
		# Generates Market Simulations (Run ONCE per day)
		simulated_multipliers = generate_simulated_returns(window_mean, window_cov_matrix, M, df)
		
		# Algorithmically Optimize Portfolio Weights
		# The optimizer finds the specific allocation that minimizes the Expected Shortfall
		optimal_weights = optimize_portfolio_weights(simulated_multipliers, P, num_stocks)
		
		# Calculate the Risk Limits (VaR & CVaR) based on those exact optimal weights
		# Applies the optimized weights to the simulated matrix to find the risk thresholds
		portfolio_final_values = np.sum((P * optimal_weights) * simulated_multipliers, axis=1)
		profit = portfolio_final_values - P
		
		var_95 = np.percentile(profit, 5)
		cvar_95 = profit[profit <= var_95].mean()
		
		# Applies the optimized weights to the ACTUAL market returns
		actual_log_returns = returns.iloc[i].values
		actual_multipliers = np.exp(actual_log_returns) - 1 # Pure percentage change
		
		# Actual P&L is the sum of (Euro allocation per stock * actual percentage change)
		actual_pnl = np.sum((P * optimal_weights) * actual_multipliers)
		
		# Breaches are flagged if the actual P&L is below the VaR threshold
		breach = 1 if actual_pnl < var_95 else 0
		
		# Log the daily metrics, including the weights for later analysis
		results.append({
			'Date': returns.index[i],
			'VaR_95': var_95,
			'CVaR_95': cvar_95,
			'Actual_PnL': actual_pnl,
			'Breach': breach,
			'Optimized_Weights': optimal_weights # This allows to see how the engine shifted capital
		})

		# Status checkpoint
		if (i - window_size) % 100 == 0:
			print(f"Processed {i - window_size} / {n - window_size} days...")

	return pd.DataFrame(results).set_index('Date')


def calculate_cvar_with_penalty(weights: np.ndarray, simulated_multipliers: np.ndarray, P: float, current_weights: np.ndarray, penalty: float) -> float:
	'''Calculates the 95% CVaR applying a turnover given the current weights and the new ones and applies a penalty.'''

	portfolio_final_values = np.sum((P * weights) * simulated_multipliers, axis=1)
	profit = portfolio_final_values - P
	
	var_95 = np.percentile(profit, 5)
	cvar_95 = profit[profit <= var_95].mean()
	base_cvar = np.abs(cvar_95)

	turnover = np.sum(np.abs(weights - current_weights))

	return base_cvar + (penalty* turnover)


def optimize_portfolio_weights_with_penalty(simulated_multipliers: np.ndarray, P: float, num_stocks: int, current_weights: np.ndarray, penalty: float = 500) -> np.ndarray:
	'''Finds the optimal asset weights to minimize the Expected Shortfall (CVaR).'''
	
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