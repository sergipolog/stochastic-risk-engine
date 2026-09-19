'''
This module contains functions to dynamically optimize and adjust the portfolio every day.
'''

from optimizer import *
from simulation import *
import yfinance as yf
import pandas as pd

def dynamic_portfolio(current_weights: dict = None, M: int = 200000, df: int = 4, penalty: float = 500) -> tuple[dict, float]:
	''' Pulls live data at market close and calculates tomorrow's CVaR minimized weights.'''

	tickers = list(current_weights.keys())

	num_stocks = len(tickers)

	P = sum(current_weights.values())

	current_weights = np.array(list(current_weights.values())) / P
	
	print(f"Pulling 1-year live market data for {len(tickers)} assets...")
	# Donwloads 1 year data for stocks selected
	data = yf.download(tickers, period="1y")
	data = pd.DataFrame(data)
	data = data.filter(like='Close', axis=1) # Keep only the closing prices

	# Daily logarithmic returns
	returns = np.log(data / data.shift(1)).dropna()

	# Covariance matrix
	latest_mean = returns.mean().values
	latest_cov_matrix = returns.ewm(span=60).cov().iloc[-num_stocks:, :].values

	print("Optimizing portfolio...")

	# Simulated Tomorrow's market conditions and Optimize
	simulated_multipliers = generate_simulated_returns(latest_mean, latest_cov_matrix, M, df)
	optimal_weights = optimize_portfolio_weights_with_penalty(simulated_multipliers, P, num_stocks, current_weights, penalty)

	# Expected worst 5% outcomes
	portfolio_final_values = np.sum((P * optimal_weights) * simulated_multipliers, axis=1)
	profit = portfolio_final_values - P

	var_95 = np.percentile(profit, 5)
	cvar_95 = profit[profit <= var_95].mean()

	# Results
	allocation = {tickers[i]: round(optimal_weights[i]*P,2) for i in range(num_stocks)}

	return dict(sorted(allocation.items(), key=lambda item: item[1], reverse= True)), round(cvar_95,2)


if __name__ == "__main__":
	my_portfolio = {"SAN": 50000, "SIE.DE": 20000, "IBE.MC":5000 , "ITX.MC": 5000, "REP.MC": 5000, "BBVA.MC": 5000, "AIR": 5000, "IAG.MC": 5000 }

	tomorrows_weights, cvar = dynamic_portfolio(my_portfolio)

	for ticker, weight in tomorrows_weights.items():
		print(f"{ticker}: {weight} €")
	print(f"Expected Shortfall (CVaR): {cvar}€")





