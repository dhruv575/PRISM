import pandas as pd
import numpy as np
import random

# Small epsilon for Sharpe calculation
eps = 1e-8
ANNUAL_TRADING_DAYS = 252

def run_equal_weight(data_df: pd.DataFrame) -> pd.Series:
    """Calculates daily returns for a static equal-weight portfolio.

    Args:
        data_df (pd.DataFrame): DataFrame with dates as index, ticker returns,
                                and an 'rf' column.

    Returns:
        pd.Series: Daily returns of the equal-weight portfolio.
    """
    stock_returns = data_df.drop(columns=['rf'], errors='ignore')
    if stock_returns.empty:
        return pd.Series(dtype=float, name="EqualWeightReturn")
    # Calculate the mean return across all stocks for each day
    daily_returns = stock_returns.mean(axis=1)
    return daily_returns.rename("EqualWeightReturn")

def run_random_portfolio(
        data_df: pd.DataFrame,
        num_stocks: int = 3,
        rebalance_days: int = 20
    ) -> pd.Series:
    """Calculates daily returns for a randomly selected portfolio,
       rebalanced periodically.

    Args:
        data_df (pd.DataFrame): DataFrame with dates as index, ticker returns,
                                and an 'rf' column.
        num_stocks (int): Number of stocks to randomly select.
        rebalance_days (int): How often to re-select stocks.

    Returns:
        pd.Series: Daily returns of the random portfolio.
    """
    stock_returns = data_df.drop(columns=['rf'], errors='ignore')
    if stock_returns.empty or stock_returns.shape[1] < num_stocks:
        print("Warning: Not enough stocks available for random portfolio.")
        return pd.Series(dtype=float, name="RandomPortfolioReturn")

    tickers = stock_returns.columns.tolist()
    portfolio_returns = pd.Series(index=data_df.index, dtype=float)
    selected_tickers = []

    for i, date in enumerate(data_df.index):
        # Rebalance check
        if i % rebalance_days == 0 or not selected_tickers:
            if len(tickers) >= num_stocks:
                 selected_tickers = random.sample(tickers, num_stocks)
            else: # Should not happen based on initial check, but safe
                 selected_tickers = tickers
            # print(f"Rebalancing Random Portfolio on {date.date()}: {selected_tickers}")

        # Calculate return for the day using selected tickers
        daily_returns = stock_returns.loc[date, selected_tickers]
        portfolio_returns[date] = daily_returns.mean() # Equal weight among selected

    return portfolio_returns.rename("RandomPortfolioReturn")

# --- Performance Metrics --- 

def calculate_cumulative_returns(returns_series: pd.Series) -> pd.Series:
    """Calculates cumulative returns from a daily returns series."""
    return (1 + returns_series.fillna(0)).cumprod()

def calculate_performance_metrics(returns_series: pd.Series, rf_series: pd.Series) -> dict:
    """Calculates annualized Sharpe Ratio and Max Drawdown."""
    if returns_series.empty or returns_series.isnull().all():
        return {"Annualized Sharpe Ratio": 0.0, "Max Drawdown": 0.0, "Cumulative Return": 1.0}
    
    cumulative_return = (1 + returns_series.fillna(0)).cumprod().iloc[-1]
    
    # Align risk-free rate series to the returns series index
    aligned_rf = rf_series.reindex(returns_series.index).fillna(0)
    
    # Calculate Excess Returns
    excess_returns = returns_series - aligned_rf
    
    # Annualized Sharpe Ratio
    # Use np.sqrt(ANNUAL_TRADING_DAYS) for annualization factor
    mean_excess_return = excess_returns.mean()
    std_dev_excess_return = excess_returns.std()
    sharpe_ratio = (mean_excess_return / (std_dev_excess_return + eps)) * np.sqrt(ANNUAL_TRADING_DAYS)
    
    # Max Drawdown
    cumulative = calculate_cumulative_returns(returns_series)
    peak = cumulative.expanding(min_periods=1).max()
    drawdown = (cumulative - peak) / (peak + eps) # Drawdown is negative or zero
    max_drawdown = abs(drawdown.min()) # Max drawdown is positive
    
    return {
        "Annualized Sharpe Ratio": round(sharpe_ratio, 4),
        "Max Drawdown": round(max_drawdown, 4),
        "Cumulative Return": round(cumulative_return, 4)
    } 