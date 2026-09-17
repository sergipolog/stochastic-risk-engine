'''
This Python Script holds the necessary functions to calculate logarithmic returns for stock data as well as the covariance matrix of the returns, 
needed for Value at Risk (VaR) calculations.
'''

import numpy as np
import pandas as pd

def calculate_log_returns(df: pd.DataFrame)-> pd.DataFrame:
	'''
	Calculates the logarithmic returns of the stock data given in a DataFrame.

	Parameters:
		df (pd.Dataframe): DataFrame containing the stock data with a MultiIndex (Date, Ticker).
	'''

	data = df.copy()
	data_shifted = data.shift(1)

	returns = pd.DataFrame(np.log(data/data_shifted))

	return returns.dropna() # The first row will be NaN since there is no previous day to compare to.

def calculate_covariance_matrix(returns: pd.DataFrame) -> pd.DataFrame:
	'''
	Calculates the covariance matrix of the logarithmic returns annualized.
	
	Parameters:
		returns (pd.DataFrame): DataFrame containing the logarithmic returns.
	'''
	return returns.cov()*252 # The general trading year has roughly 252 days of open market.