# 004 - Pairs Trading

## Mauricio Martíenez Ulloa

## Project Overview

**Proyecto4_PairsTrading** is a highly modular and professional platform for analyzing and evaluating _pairs trading_ strategies, tailored for quantitative research and algorithmic portfolio management. The framework is designed for rigorous backtesting utilizing advanced cointegration methods and dynamic parameter estimation via Kalman filters and Vector Error Correction Models (VECM).

## Key Features

- **Dynamic Cointegration Analysis:** Johansen test for identifying long-term relationships and optimal pair selection.
- **Adaptive Kalman Filtering:** Dual filter design for on-the-fly hedge ratio & spread estimation and VECM signal refinement.
- **Signal Generation:** Quantitative Z-score-based regime switching for robust trade execution logic.
- **Realistic Portfolio Simulation:** Professional handling of long/short positions, commissions, borrowing costs, and ongoing portfolio valuation through time.
- **Professional Metrics Suite:** Sharpe Ratio, Sortino, Calmar, maximum drawdown, win rate, profit factor, and trade return distribution, suited for institutional quant analysis.
- **Expert Diagnostics & Visualization:** Full suite of graphical outputs for signals, spreads, and performance.

## Modular Code Structure

Designed for extensibility and maintainability:
- **backtest.py:** Core backtesting logic, execution, and metric aggregation.
- **kalman_filters.py:** State-of-the-art Kalman filter implementations for adaptive parameter tracking.
- **metrics.py:** Comprehensive professional calculation of portfolio metrics.
- **functions.py:** Core functions for portfolio valuation and trade management.
- **graphs.py:** Expert visualizations for all key strategy components.
- **objects.py:** Clean data structures for trading operations and records.
- **utils.py:** Utilities for data cleaning, partitioning, and cointegration summary.

## Professional Installation & Usage

The following instructions guide professional users and quantitative researchers through setup, cloning, environment creation, and running their first backtests.

### 1. Clone the Repository

```bash
git clone https://github.com/ulloa09/Proyecto4_PairsTrading.git
cd Proyecto4_PairsTrading
```

### 2. Set Up a Python Virtual Environment

Using `venv` (recommended for professional reproducibility):

```bash
python3 -m venv env
source env/bin/activate       # On Unix/MacOS
env\Scripts\activate          # On Windows
```

### 3. Install Dependencies

> All core dependencies are industry-standard for quantitative analytics.

```bash
pip install -r requirements.txt
```
If `requirements.txt` is not provided, manually install the essentials:
```bash
pip install numpy pandas matplotlib statsmodels seaborn
```

### 4. Prepare Data

- Ensure your financial time series data is available as a CSV with columns representing tickers and a 'Date' column.
- Place your input data in the `data/` folder (recommended for organizational clarity).

### 5. Run Backtesting and View Results

Execute the main pipeline using the provided scripts. For quick portfolio evaluation:

```bash
python main.py
```
- This will run the cointegration analysis, select optimal pairs, conduct the backtest, and print a full suite of performance metrics along with visual diagnostics.
- Modify parameters (`WINDOW`, `THETA`, `Q`, `R`, etc.) in `main.py` for custom experimentation.

### 6. Explore Diagnostic Outputs

After execution, key performance metrics and visualizations will be displayed, including:
- Portfolio value evolution across train/test/validation phases.
- Signal timing (entry/exit), spread dynamics, and trade returns histograms.

## Extensibility & Best Practices

- *Modular Expansion*: Easily add new cointegration methods, prediction models, or custom metrics.
- *Type-Safe Development*: Use the clean dataclass and function design for robust, maintainable code.
- *Professional Research Integration*: Suitable for institutional-style research, strategy prototyping, and validation.


---

_For inquiries, collaboration, or advanced strategy consulting, see author's GitHub profile._
