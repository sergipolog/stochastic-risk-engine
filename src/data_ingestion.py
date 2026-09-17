'''
This Python Script makes the data ingest, downloading 3 years of daily historical closing prices
for 5 major European stocks.
'''

import yfinance as yf
import pandas as pd

if __name__ == "__main__":

	destination = 'data/raw/historical_stock.csv'

	try:
		data = yf.download(['SAN', 'ITX.MC', 'AIR.PA', 'SIE.DE', 'IBE.MC'], start="2000-01-01", interval='1d', group_by='ticker')
		df = pd.DataFrame(data)
		df = df.filter(like='Close', axis=1) # Keep only the closing prices
		df.to_csv(destination)
		print(df.head(10))
		print(f"Data downloaded successfully. Saved in {destination}")
	except Exception as e:
		print(f'Error downloading data: {e}')


	

