import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from dynamic import *


## HEADER ##

st.title("Stochastic Risk-Management Engine")
st.markdown("""
**System Scope:** This engine functions as a mathematical shock absorber. It actively monitors live volatility clustering and dynamically reallocates capital to minimize the 5% Expected Shortfall (tail risk) during severe market downturns under uncertainty.
""")

## SIDEBAR ##

if 'row_count' not in st.session_state:
	st.session_state.row_count = 3

st.sidebar.header("Portfolio")

user_portfolio = {}

for i in range(st.session_state.row_count):
	col1,col2 = st.sidebar.columns([1,1.2])

	ticker_val = col1.text_input(
			f"Ticker {i}",
			key= f"ticker_{i}",
			label_visibility= "collapsed",
			placeholder="SAN"
	)

	money_val = col2.number_input(
			f"Money {i}",
			key= f"money_{i}",
			min_value= 0,
			step= 100,
			label_visibility="collapsed"
	)

	if ticker_val:
		user_portfolio[ticker_val.strip().upper()] = money_val

if st.sidebar.button("➕ Add Stock"):
	st.session_state.row_count += 1
	st.rerun()

st.sidebar.divider()
penalty = st.sidebar.number_input(
	f"Turnover Penalty",
	key="turnover",
	min_value=0,
)

if st.sidebar.button("Run Engine"):
	if not user_portfolio:
		st.sidebar.error("Please enter at least one asset.")
	else:
		with st.spinner("Executing live market pull & optimization..."):
			st.write("Portfolio sent to backend", user_portfolio)
