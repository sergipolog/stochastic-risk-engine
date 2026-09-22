# Stochastic Risk-Management Engine

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://stochastick-risk-engine.streamlit.app)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ⚠️ Academic & Architectural Disclaimer

This project is a technical portfolio piece designed to demonstrate proficiency in quantitative modeling, python usage, and full-stack data application development.

**This is not a predictive trading algorithm.** Real-world financial markets are highly complex systems influenced by liquidity constraints, macroeconomic shifts, and market microstructure that cannot be entirely captured by historical covariance or a static Student's t-distribution.

This engine is engineered strictly to showcase the architectural translation of academic risk constraints (such as $L_1$ norm turnover penalties and Expected Shortfall minimization) into a functional, interactive software environment. It is not intended, nor should it be used, for live capital deployment or actual financial decision-making.

## 📌 System Scope

This project is an interactive, quantitative portfolio optimization engine designed to act as a mathematical shock absorber for asset allocation. It actively monitors historical volatility clustering and dynamically reallocates capital to minimize the **5% Expected Shortfall (CVaR)** during severe market downturns under uncertainty.

The application bridges complex risk-modeling with an intuitive frontend, demonstrating end-to-end data science and software engineering principles.

## 🚀 Live Demo

[Launch the Live Dashboard on Streamlit Community Cloud](https://stochastic-risk-engine.streamlit.app)

`![Dashboard Preview](docs/main_1.png)`

## 🧠 Mathematical Architecture

Unlike standard mean-variance optimizers that assume normally distributed returns, this engine is built to survive fat-tail market crashes using three core quantitative pillars:

* **Monte Carlo Simulation (Fat-Tail Risk):** Generates 200,000 synthetic future market scenarios using a Student's t-distribution. This captures the extreme, unprecedented market shocks that standard normal distributions ignore.
* **Expected Shortfall (CVaR) Minimization:** The SciPy optimizer's objective function ignores average volatility and strictly targets the left tail of the distribution, minimizing the mathematical average of the worst 5% of simulated outcomes.
* **Dynamic Turnover Penalty ($\lambda$):** Bridges theoretical math with real-world trading constraints. The engine incorporates an $L_1$ norm penalty derived directly from the user's broker commission fee. The optimizer runs a strict cost-benefit analysis, refusing to execute trades unless the reduction in risk mathematically exceeds the real-world transaction cost.
* **Dynamic Volatility Modeling (EWMA):** To account for volatility clustering—where market shocks arrive in sudden, concentrated waves—the engine calculates a 60-day Exponentially Weighted Moving Average (EWMA) covariance matrix. Rather than diluting recent market crashes with months of calm historical data, this dynamic memory forces the optimizer to react aggressively to current market turbulence.

## ⚙️ Engineering & System Design

The application follows a strict Model-View-Controller (MVC) architecture, decoupling the heavy matrix mathematics from the frontend rendering.

* **Backend Engine (`dynamic.py`):** Handles the data ingestion (`yfinance`), covariance matrix calculations, Monte Carlo generation, and the SciPy optimization pipeline.
* **Frontend UI (`app.py`):** An interactive Streamlit dashboard featuring dynamic session-state management for portfolio inputs and real-time error handling for API timeouts or invalid ticker queries.
* **Performance Optimization:** Utilizes Streamlit's `@st.cache_data` decorator to cache API responses and stochastic results, ensuring instantaneous recalculations when users toggle frontend parameters.

## 📊 Analytics Dashboard

The engine translates abstract matrix math into actionable insights via three interactive Plotly visualizations:

1. **Correlation Matrix:** A normalized heatmap proving the engine's diversification logic by highlighting inversely correlated assets.
2. **Historical Stress Test:** A static backtest projecting tomorrow's optimized weights against the past 252 days of actual market data, visually isolating historical VaR breaches.
3. **Monte Carlo Distribution:** A logarithmic histogram mapping the 200,000 simulated futures, clearly delineating the 95% VaR cutoff and the Expected Shortfall zone.

## 🛠️ Technology Stack

* **Language:** Python
* **Data Processing & Math:** NumPy, Pandas, SciPy
* **Market Data API:** yfinance
* **Data Visualization:** Plotly
* **Web Framework & Hosting:** Streamlit, Streamlit Community Cloud

## 💻 Local Installation

To run this engine locally on your machine:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/sergipolog/stochastic-risk-engine.git](https://github.com/sergipolog/stochastic-risk-engine.git)
   cd stochastic-risk-engine
   ```
