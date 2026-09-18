'''
This module contains functions to dynamically optimize and adjust the portfolio every day.
'''

from optimizer import *
from simulation import *
import yfinance as yf
import pandas as pd

def dynamic_portfolio(tickers: list, P: float, M: int = 200000, df: int = 4) -> tuple[dict, float]:
	''' Pulls live data at market close and calculates tomorrow's CVaR minimized weights.'''

	print(f"Pulling 1-year live market data for {len(tickers)} assets...")
	# Donwloads 1 year data for stocks selected
	data = yf.download(tickers, period="1y")
	data = pd.DataFrame(data)
	data = data.filter(like='Close', axis=1) # Keep only the closing prices

	# Daily logarithmic returns
	returns = np.log(data / data.shift(1)).dropna()
	num_stocks = len(tickers)

	# Covariance matrix
	latest_mean = returns.mean().values
	latest_cov_matrix = returns.ewm(span=60).cov().iloc[-num_stocks:, :].values

	print("Optimizing weights...")

	# Simulated Tomorrow's market conditions and Optimize
	simulated_multipliers = generate_simulated_returns(latest_mean, latest_cov_matrix, M, df)
	optimal_weights = optimize_portfolio_weights(simulated_multipliers, P, num_stocks)

	# Expected worst 5% outcomes
	portfolio_final_values = np.sum((P * optimal_weights) * simulated_multipliers, axis=1)
	profit = portfolio_final_values - P

	var_95 = np.percentile(profit, 5)
	cvar_95 = profit[profit <= var_95].mean()

	# Results
	allocation = {tickers[i]: round(optimal_weights[i]*100,2) for i in range(num_stocks)}

	return dict(sorted(allocation.items(), key=lambda item: item[1], reverse= True)), round(cvar_95,2)


if __name__ == "__main__":
	my_portfolio = ["SAN", "SIE.DE", "IBE.MC", "ITX.MC", "REP.MC", "BBVA.MC", "AIR", "IAG.MC"]

	tomorrows_weights, cvar = dynamic_portfolio(my_portfolio, 100000)

	for ticker, weight in tomorrows_weights.items():
		print(f"{ticker}: {weight}%")
	print(f"Expected Shortfall (CVaR): {cvar}€")





