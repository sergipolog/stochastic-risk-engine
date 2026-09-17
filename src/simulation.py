'''
Simulation module for the VAR_ENGINE project.
'''

import numpy as np
import pandas as pd


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

	simulated_returns = np.random.multivariate_normal(returns.mean().values, cov_matrix.values, size=(M,T))

	cumulative_returns = np.exp(np.sum(simulated_returns, axis=1)) # Sum of returns over the time horizon for each simulation

	portfolio_final_values = np.sum(P/5 * cumulative_returns, axis=1) # Portfolio total values

	profit = portfolio_final_values - P # Profit distribution

	var_95 = np.percentile(profit, 5) # 5th percentile of the profit distribution

	print(var_95)


