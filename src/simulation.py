'''
Simulation module for the VAR_ENGINE project.
'''

import numpy as np
import pandas as pd

def simulate_portfolio_returns(returns: pd.DataFrame, cov_matrix: pd.DataFrame, T: int, M: int, P: float) -> np.ndarray:
	'''
	Simulates the portfolio returns using a multivariate normal distribution based on the mean and covariance of the stock returns.

	Parameters:
		returns (pd.DataFrame): DataFrame containing the logarithmic returns.
		cov_matrix (pd.DataFrame): DataFrame containing the covariance matrix of the returns.
		T (int): Time horizon in days.
		M (int): Number of simulations.
		P (float): Portfolio value in euros.

	Returns:
		np.ndarray: Array containing the simulated portfolio final values.
	'''

	simulated_returns = np.random.multivariate_normal(returns.mean().values, cov_matrix.values, size=(M,T))

	cumulative_returns = np.exp(np.sum(simulated_returns, axis=1)) # Sum of returns over the time horizon for each simulation

	portfolio_final_values = np.sum(P/returns.shape[1] * cumulative_returns, axis=1) # Portfolio total values

	return portfolio_final_values

def simulate_portfolio_returns_roll(mean_returns: np.ndarray, cov_matrix: np.ndarray, M: int, P: float, num_stocks: int) -> tuple[float,float]:
	'''
	Auxiliary function for rolling_engine.
	Simulates a 1-day portfolio VaR using a multivariate normal distribution.

	Parameters:
		mean_returns (np.ndarray): 1D array containing the mean logarithmic returns.
		cov_matrix (np.ndarray): 2D array containing the daily covariance matrix.
		M (int): Number of simulations.
		P (float): Starting portfolio value in euros.
		num_stocks (int): Number of stocks in the portfolio.

	Returns:
		tuple[float, float]: The 95% Confidence Value at Risk (VaR) and Conditional Value at Risk (CVaR) for a single day.
	'''

	simulated_returns = np.random.multivariate_normal(mean_returns, cov_matrix, size=(M, 1))

	cumulative_returns = np.exp(np.sum(simulated_returns, axis=1)) 

	portfolio_final_values = np.sum((P / num_stocks) * cumulative_returns, axis=1) 

	profit = portfolio_final_values - P 

	var_95 = np.percentile(profit, 5)

	cvar_95 = profit[profit <= var_95].mean()

	return float(var_95), float(cvar_95)

def rolling_engine(returns: pd.DataFrame, window_size: int, M: int, P: float) -> pd.DataFrame:
	'''
	Performs a rolling backtesting of the portfolio returns using a multivariate normal distribution based on the mean and covariance of the stock returns.

	Parameters:
		returns (pd.DataFrame): DataFrame containing the logarithmic returns.
		window_size (int): Size of the rolling window.
		M (int): Number of simulations.
		P (float): Portfolio value in euros.

	Returns:
		pd.DataFrame: DataFrame containing the simulated portfolio final values for each rolling window.
	'''

	n = len(returns)
	results = []

	for i in range(window_size,n):
		historical_slice = returns.iloc[i-window_size:i] # Gets the historical slice of returns for the current window

		window_mean = historical_slice.mean().values # Calculates the mean of the returns for the current window
		window_cov_matrix = (historical_slice.cov()).values # Calculates the covariance matrix for the current window

		var_95, cvar_95 = simulate_portfolio_returns_roll(window_mean, window_cov_matrix, M, P, num_stocks=returns.shape[1]) # Simulates the portfolio returns for the current window

		actual_log_returns = returns.iloc[i] # Gets the actual log returns for the current day

		# Convert the actual daily log returns to standard multipliers, applied to the cash weight
		cash_per_stock = P / returns.shape[1]
		actual_pnl = np.sum((np.exp(actual_log_returns) - 1) * cash_per_stock)

		# Check if there is a breach of the VaR threshold
		breach = 1 if actual_pnl < var_95 else 0

		results.append({
			'Date': returns.index[i],
			'VaR_95': var_95,
			'CVaR_95': cvar_95,
			'Actual_PnL': actual_pnl,
			'Breach': breach
		})

	
	return pd.DataFrame(results).set_index('Date')

if __name__ == "__main__":

	T = 30 # Time horizon in days
	M = 10000 # Number of simulations
	P = 100000 # Portfolio value in euros - Equally weighted portfolio of 5 stocks

	np.random.seed(2026) # For reproducibility

	data = pd.read_csv('data/processed/historical_stock_cleaned.csv', index_col=0, header=[0,1])
	returns = pd.read_csv('data/processed/log_returns.csv', index_col=0, header=0, decimal='.')
	cov_matrix = pd.read_csv('data/processed/covariance_matrix.csv', index_col=0, header=0, decimal='.')
	# Annualized covariance matrix needs to be divided by 252 to get daily covariance
	cov_matrix = cov_matrix / 252

	portfolio_final_values = simulate_portfolio_returns(returns, cov_matrix, T, M, P)

	profit = portfolio_final_values - P # Profit distribution

	var_95 = np.percentile(profit, 5) # 5th percentile of the profit distribution

	print(var_95)


