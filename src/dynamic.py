'''
This module contains functions to dynamically optimize and adjust the portfolio as data is introduced.
'''

from optimizer import *
import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st

@st.cache_data(show_spinner=False)
def dynamic_portfolio(current_weights: dict = None, penalty: float = 0.1, M: int = 200000, df: int = 4) -> tuple[dict, float, float, pd.DataFrame, float]:
	''' 
	This function builds the next pipeline: fetching live market data, calculating 
    dynamic covariance, and computing the optimal tail-risk minimized asset allocation.

    This controller normalizes absolute capital inputs, downloads a 1-year trailing window of 
    daily closing prices via yfinance, and calculates an Exponentially Weighted Moving Average 
    (EWMA) covariance matrix (60-day span) to heavily weight recent volatility clusters. It then 
    executes a 200,000-scenario Monte Carlo simulation to find the CVaR-minimized target weights, 
    accounting for real-world transaction penalties.

		Parameters:
			current_weights (dict): A dictionary mapping asset tickers to their current capital 
									allocation (e.g., {'SAN.MC': 1500, 'IBE.MC': 2500}).
			penalty (float): The broker's transaction fee expressed as a percentage (e.g., 0.1 for 0.10%). 
							This is mathematically scaled into an absolute Euro penalty inside the engine. Defaults to 0.1.
			M (int): The number of stochastic Monte Carlo futures to simulate. Defaults to 200,000.
			df (int): Degrees of freedom for the fat-tailed Student's t-distribution. Defaults to 4.

		Returns:
			- dict: The mathematically optimized target allocations in Euros, sorted in descending order.
			- float: The 1-Day 95% Expected Shortfall (CVaR) in Euros.
			- float: The 1-Day 95% Value at Risk (VaR) in Euros.
			- pd.DataFrame: The historical daily logarithmic returns for the requested assets.
			- np.ndarray: An array of the M simulated 1-day profit/loss outcomes in Euros (used for the histogram).
				
		Raises:
			ValueError: If the yfinance API returns an empty dataset or if invalid ticker symbols are detected.
	'''

	total_capital = sum(current_weights.values())

	penalty = total_capital * (penalty/100)

	tickers = list(current_weights.keys())

	num_stocks = len(tickers)

	current_weights = np.array(list(current_weights.values())) / total_capital

	print(f"Pulling 1-year live market data for {len(tickers)} assets...")
	# Donwloads 1 year data for stocks selected
	data = yf.download(tickers, period="1y")
	data = pd.DataFrame(data)

	if data.empty:
		raise ValueError("Critical Data Failure: None of the entered tickers exist on Yahoo Finance.")
	
	data = data.filter(like='Close', axis=1) # Keep only the closing prices

	if data.isnull().all().any():
		raise ValueError("Invalid Ticker Detected: One or more assets could not be found. Please verify your symbols (e.g., use 'SAN.MC' for European equities).")


	# Daily logarithmic returns
	returns = np.log(data / data.shift(1)).dropna()

	# Covariance matrix
	latest_mean = returns.mean().values
	latest_cov_matrix = returns.ewm(span=60).cov().iloc[-num_stocks:, :].values

	print("Optimizing portfolio...")

	np.random.seed(2026)
	# Simulated Tomorrow's market conditions and Optimize
	simulated_multipliers = generate_simulated_returns(latest_mean, latest_cov_matrix, M, df)
	optimal_weights = optimize_portfolio_weights_with_penalty(simulated_multipliers, total_capital, num_stocks, current_weights, penalty)

	# Expected worst 5% outcomes
	portfolio_final_values = np.sum((total_capital * optimal_weights) * simulated_multipliers, axis=1)
	profit = portfolio_final_values - total_capital

	var_95 = np.percentile(profit, 5)
	cvar_95 = profit[profit <= var_95].mean()

	# Results
	allocation = {tickers[i]: round(optimal_weights[i]*total_capital,2) for i in range(num_stocks)}

	return dict(sorted(allocation.items(), key=lambda item: item[1], reverse= True)), round(cvar_95,2), round(var_95,2), returns, profit


if __name__ == "__main__":
	my_portfolio = {"SAN": 50000, "SIE.DE": 20000, "IBE.MC":5000 , "ITX.MC": 5000, "REP.MC": 5000, "BBVA.MC": 5000, "AIR": 5000, "IAG.MC": 5000 }

	tomorrows_weights, cvar, _, _,_ = dynamic_portfolio(my_portfolio, 0.1)

	for ticker, weight in tomorrows_weights.items():
		print(f"{ticker}: {weight} €")
	print(f"Expected Shortfall (CVaR): {cvar}€")





