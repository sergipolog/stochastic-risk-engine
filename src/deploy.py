import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.express as px
import uuid
from dynamic import *


## HEADER ##

st.title("Stochastic Risk-Management Engine")
st.markdown("""
**System Scope:** This engine functions as a mathematical shock absorber. It monitors volatility clustering and dynamically reallocates capital to minimize the 5% Expected Shortfall (tail risk) during severe market downturns under uncertainty.
""")

## LANDING PAGE ##

if 'initialized' not in st.session_state:
	id0, id1, id2 = str(uuid.uuid4()), str(uuid.uuid4()), str(uuid.uuid4())
	st.session_state.row_ids = [id0, id1, id2]
	
	# This flag forces the dashboard to render immediately on load
	st.session_state.show_dashboard = True 
	
	# Test portfolio
	st.session_state[f"ticker_{id0}"] = "SAN.MC"
	st.session_state[f"money_{id0}"] = 1500
	st.session_state[f"ticker_{id1}"] = "IBE.MC"
	st.session_state[f"money_{id1}"] = 2500
	st.session_state[f"ticker_{id2}"] = "ITX.MC"
	st.session_state[f"money_{id2}"] = 3500
	
	st.session_state.initialized = True

## SIDEBAR ##


st.sidebar.header("Portfolio")

user_portfolio = {}

def delete_row(row_id):
	if len(st.session_state.row_ids) > 1:

		st.session_state.row_ids.remove(row_id)

		if f"ticker_{row_id}" in st.session_state:
			del st.session_state[f"ticker_{row_id}"]
		if f"money_{row_id}" in st.session_state:
			del st.session_state[f"money_{row_id}"]
		
	else:
			st.sidebar.warning("You must have at least one asset.")


for row_id in st.session_state.row_ids:
	col1,col2,col3 = st.sidebar.columns([0.8,1.2, 0.6])

	ticker_val = col1.text_input(
			f"Ticker",
			key= f"ticker_{row_id}",
			label_visibility= "collapsed",
			placeholder="SAN.MC"
	)

	money_val = col2.number_input(
			f"Money",
			key= f"money_{row_id}",
			min_value= 0,
			step= 100,
			label_visibility="collapsed"
	)

	col3.button("❌", key=f"del_{row_id}", on_click=delete_row, args=(row_id,))

	if ticker_val:
		user_portfolio[ticker_val.strip().upper()] = money_val

if st.sidebar.button("➕ Add Stock", use_container_width= True):
	st.session_state.row_ids.append(str(uuid.uuid4()))
	st.rerun()

st.sidebar.divider()
penalty = st.sidebar.number_input(
	f"Broker Comission Fee (%)",
	key="comission",
	min_value=0.0,
	value=0.1,
	step=0.05,
	help="The percentage fee your broker charges per transaction."
)

if st.sidebar.button("Run Engine"):
	st.session_state.show_dashboard = True

if st.session_state.show_dashboard:
	if not user_portfolio:
		st.warning("Please enter at least one asset in the sidebar.")
	else:

## MAIN ##

		with st.spinner("Executing live market pull & optimization..."):

			try:

				total_capital = sum(user_portfolio.values())

				target_allocations, cvar_95, var_95, df_returns, simulated_profit = dynamic_portfolio(user_portfolio, penalty)

				total_turnover = sum([abs(target_allocations.get(ticker, 0) - current_val) for ticker, current_val in user_portfolio.items()])
				total_fees = total_turnover * (penalty / 100)

				st.subheader("1-Day Risk Assessment")

				col1,col2,col3 = st.columns(3)
				col1.metric("Current Capital", f"€{total_capital:,.2f}")
				col2.metric("95% Expected Shortfall", f"€{cvar_95:,.2f}")
				col3.metric("Total Fees", f"€{total_fees:,.2f}")

				st.subheader("Target Allocation for Tomorrow")

				df_weights = pd.DataFrame.from_dict(target_allocations,orient='index', columns = ["Target Distribution"])

				st.bar_chart(df_weights)

				if isinstance(df_returns.columns, pd.MultiIndex):
					df_returns.columns = df_returns.columns.droplevel(0)
		
				st.divider()
				st.subheader("Engine Analytics")
				
				tab1, tab2, tab3 = st.tabs(["Correlation Matrix", "Historical Stress Test", "Monte Carlo Distribution"])
				
				with tab1:
					st.markdown("**Asset Risk Profile**")
					with st.expander("ℹ️ How to read this chart"):
						st.write("""
						**The Goal:** Build a structural shock absorber that prevents systemic failure while mathematically accounting for current market momentum.
						* **Heatmap:** Dark red indicates assets that crash together. Lighter colors indicate independent variables that provide structural safety.
						* **Risk vs. Return Profile:** The X-axis represents the severity of standalone crashes (Tail Risk), while the Y-axis tracks recent market momentum (Expected Return). 
						
						*Engine Logic:* The optimizer does not blindly buy the "safest" uncorrelated asset. Instead, it balances Correlation, Standalone Tail Risk, and Expected Return. The algorithm will aggressively allocate capital to a highly volatile asset if its positive mathematical drift offsets its tail risk.
						""")
					
					corr_matrix = df_returns.corr() 

					# Heatmap
					fig_corr = px.imshow(
						corr_matrix, 
						text_auto=".2f", 
						aspect="auto", 
						color_continuous_scale='RdBu_r',
						zmin=-1, 
						zmax=1,
						title = 'Asset Correlation'
					)

					asset_metrics = []
					recent_returns = df_returns.tail(60)

					for ticker in recent_returns.columns:
						asset_returns = recent_returns[ticker].values
						
						# Calculates Magnitude (Risk)
						var_risk = np.percentile(asset_returns, 5)
						cvar_risk = abs(asset_returns[asset_returns <= var_risk].mean()) * 100
						
						# Calculates Drift (Expected Return)
						expected_return = asset_returns.mean() * 100 
						
						asset_metrics.append({'Ticker': ticker, 'Tail Risk (%)': cvar_risk, 'Expected Return (%)': expected_return})

					df_metrics = pd.DataFrame(asset_metrics)

					# Scatter plot
					fig_scatter = px.scatter(
						df_metrics,
						x='Tail Risk (%)',
						y='Expected Return (%)',
						text='Ticker',
						color='Expected Return (%)',
						color_continuous_scale='Blues',
						title="Risk vs. Expected Return (60-Day)",
						labels={
							'Tail Risk (%)': 'Worst 5% Avg Daily Loss (%)',
							'Expected Return (%)': 'Avg Daily Return (%)'
						}
					)

					# Adjust text position so it doesn't overlap the dots
					fig_scatter.update_traces(textposition='top center', marker=dict(size=12))
					fig_scatter.update_layout(
						plot_bgcolor="rgba(0,0,0,0)",
						paper_bgcolor="rgba(0,0,0,0)",
						coloraxis_showscale=False
					)


					st.plotly_chart(fig_corr, use_container_width=True)
					st.plotly_chart(fig_scatter, use_container_width=True)


				with tab2:
					st.markdown("**Simulated One Year Daily P&L vs. VaR Limit**")
					with st.expander("ℹ️ How to read this chart"):
						st.write("""
						**The Goal:** Prove the mathematical limit holds up against historical reality.
						This is a static stress test for current target weights, it applies tomorrow's optimized target weights to the past year of actual market data. 
						* **The Blue Line:** What your daily Euro profit/loss would have been holding this exact allocation.
						* **The Red Dots:** Market days where the crash was so severe it breached the 95% Value at Risk (VaR) limit.
						
						*Engine Logic:* We expect breaches roughly 5% of the time. If there are massive clusters of red dots, the market experienced sustained extreme volatility.
						""")
					
					weights_array = np.array([target_allocations.get(col, 0) for col in df_returns.columns])
					
					# What the daily Euro profit/loss would have been historically
					daily_pnl_euros = df_returns.dot(weights_array)
					
					df_pnl = pd.DataFrame({'Date': df_returns.index, 'Daily PnL (€)': daily_pnl_euros})
					
					# Identifies breaches
					breaches = df_pnl[df_pnl['Daily PnL (€)'] <= var_95]

					# Plot time series
					fig_line = px.line(df_pnl, x='Date', y='Daily PnL (€)')
					fig_line.add_hline(y=var_95, line_dash="dash", line_color="red", annotation_text="95% VaR Limit")
					
					# Plot breaches
					fig_line.add_scatter(
						x=breaches['Date'], y=breaches['Daily PnL (€)'], 
						mode='markers', marker=dict(color='darkorange', size=8), name='VaR Breach'
					)
					st.plotly_chart(fig_line, use_container_width=True)
					
				with tab3:
					st.markdown("**200,000 Synthetic Futures (1-Day P&L Distribution)**")
					with st.expander("ℹ️ How to read this chart"):
						st.write("""
						**The Goal:** Visualize the probability of extreme tail-risk events.
						Unlike standard models that assume perfect bell curves, this engine generates 200,000 future scenarios using a fat-tailed distribution to account for unprecedented market crashes.
						* **The Red Line:** The 95% Value at Risk (VaR) cutoff.
						* **The Orange Line:** The Expected Shortfall (CVaR). This is the mathematical average of the worst 5% of simulated futures.
						
						*Engine Logic:* The optimizer's sole objective is to push that Orange Line as far to the right (closer to zero) as mathematically possible, minimizing catastrophe.
						""")
					
					# Plot the 200000 Monte Carlo outcomes
					fig_dist = px.histogram(
						x=simulated_profit, nbins=150, 
						labels={'x': '1-Day Profit / Loss (€)', 'y': 'Frequency'}
					)
					
					fig_dist.add_vline(x=var_95, line_dash="dash", line_color="red", annotation_text="95% VaR", annotation_position="top right")
					fig_dist.add_vline(x=cvar_95, line_dash="dash", line_color="orange", annotation_text="95% CVaR", annotation_position="top left")
					
					st.plotly_chart(fig_dist, use_container_width=True)


			except ValueError as ve:
				st.warning(ve)
			except Exception as e:
				st.error(f"Engine Fault: {e}")


