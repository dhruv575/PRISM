**STAT 4830 Frontend Technical Description**  
---

Dhruv Gupta, Kelly Wang, Didrik Wiig-Andersen, Aiden Lee, Frank Ma  
STAT 4830, Project PRISM 							      	        		   

**Project Proposal**

As part of our class, we have created an online gradient ascent based model for portfolio allocation and optimization. We want to make an interactive frontend for the project that allows users to simulate and test performance on several different year ranges, hyperparameter combinations, and stock market universes.

They should receive quick and immediate feedback, as well as comparisons with several different benchmarks in terms of both cumulative returns and annualized sharpe ratio.

We should try and include as many relevant and valid graphs as possible to make the website visually appealing.

**Proposed Technical Stack**

* **Hugging Face:** Our Python backend will be hosted in a Huggingface space, which will be in charge of actually running the model and providing back the results  
* **React:** The frontend will be all react. The app has been created using npx create-react-app to begin with.

**Backend Setup**

We have cloned our Hugging Face space into our project (a blank Gradio template project). We have renamed the folder it is in to be called "backend"

**Necessary Dependencies**  
import torch  
import pandas as pd  
import matplotlib as mpl  
import matplotlib.pyplot as plt  
import matplotlib.dates as mdates  
import yfinance as yf  
from datetime import datetime  
import numpy as np  
import seaborn as sns  
import wrds
import random

**Backend Development Steps**

These steps are to be followed by Cursor Agent running Claude 3.7 Sonnet. Each step should only be completed one at a time, and after each step is completed, the readme file should be updated accordingly. Do NOT go ahead at all and do not set up extra steps in advance. We have begun by cloning our HuggingFace spcae into our project and named the folder backend (Gradio blank template).

1. Set up the file directory and all necessary introductory files for the project. Ensure we have installed and are able to run any necessary dependencies.  
   *Completed: Created `backend/requirements.txt` and `backend/app.py`.*  
2. Store the following list of stock tickers, organized by sector, in a JSON file  
   *Completed: Created `backend/data/tickers_by_sector.json`.*  

\[  
  {  
    "sector": "Technology",  
    "tickers": \["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AVGO", "ORCL", "IBM", "CSCO", "TSM", "ASML", "AMD", "TXN", "INTC", "MU", "QCOM", "LRCX", "NXPI", "ADI"\]  
  },  
  {  
    "sector": "Consumer Discretionary",  
    "tickers": \["AMZN", "TSLA", "NKE", "MCD", "SBUX", "YUM", "GM", "F", "RIVN", "NIO", "TTWO", "EA", "GME", "AMC"\]  
  },  
  {  
    "sector": "Financials",  
    "tickers": \["JPM", "V", "MA", "GS", "MS", "BAC", "C", "AXP", "SCHW"\]  
  },  
  {  
    "sector": "Health Care",  
    "tickers": \["UNH", "JNJ", "LLY", "PFE", "MRNA", "BMY", "GILD", "CVS", "VRTX", "ISRG"\]  
  },  
  {  
    "sector": "Consumer Staples",  
    "tickers": \["WMT", "PG", "TGT", "KO", "PEP", "TSN", "CAG", "SYY", "HRL", "MDLZ"\]  
  },  
  {  
    "sector": "Energy",  
    "tickers": \["XOM", "CVX", "NEE", "DUK", "SO", "D", "ENB", "SLB", "EOG", "PSX"\]  
  },  
  {  
    "sector": "Industrials",  
    "tickers": \["DE", "LMT", "RTX", "BA", "CAT", "GE", "HON", "UPS", "EMR", "NOC", "FDX", "CSX", "UNP", "DAL"\]  
  },  
  {  
    "sector": "Real Estate",  
    "tickers": \["PLD", "AMT", "EQIX", "O", "SPG", "VICI", "DLR", "WY", "EQR", "PSA"\]  
  },  
  {  
    "sector": "Materials",  
    "tickers": \["ADM", "BG", "CF", "MOS", "FMC"\]  
  },  
  {  
    "sector": "Communication Services",  
    "tickers": \["NFLX", "DIS", "PARA", "WBD", "CMCSA", "SPOT", "LYV"\]  
  }  
\]

3. We have stored the data for each of these tickers from 1-1-2007 to 4-1-2025 in a file called data/stock_data.csv and the risk free returns values for each day in data/risk_free_data.csv. Please read them in and save them as df for future reference  
   *Completed: Loaded data into global DataFrames in `backend/app.py`.*  
4. Create a route that, given optional inputs of start date, end date, and tickers, creates a dataframe that only contains data on the given tickers for the given timeframe  
   *Completed: Created `filter_data` function in `backend/utils.py` and test interface in `backend/app.py`.*  
5. Build out a route that runs our OGD on a given dataframe which takes in hyperparameters and returns both the day to day weights and day to day returns (both cumulative and by ticker)  
   *Completed: Created `run_ogd` function in `backend/optimization.py` and integrated into Gradio interface in `backend/app.py`.*

\# Objective function  
def calculate\_sharpe(  
        returns: torch.tensor,  
        risk\_free\_rate: torch.tensor \= None  
    ):  
    if risk\_free\_rate is not None:  
        excess\_returns \= returns \- risk\_free\_rate  
    else:  
        excess\_returns \= returns  
    sharpe \= torch.mean(excess\_returns, dim=0) / torch.std(excess\_returns, dim=0)  
    return sharpe

def calculate\_sortino(  
        returns: torch.tensor,  
        min\_acceptable\_return: torch.tensor  
    ):  
    if min\_acceptable\_return is not None:  
        excess\_returns \= returns \- min\_acceptable\_return  
    downside\_deviation \= torch.std(  
        torch.where(excess\_returns \< 0, excess\_returns, torch.tensor(0.0)),  
    )  
    sortino \= torch.mean(excess\_returns, dim=0) / (downside\_deviation \+ eps\*\*2)  
    return sortino

def calculate\_max\_drawdown(  
        returns: torch.tensor  
):  
    """calculates max drawdown for the duration of the returns passed  
    i.e. expects returns to be trimmed to the period of interest

    max drawdown is defined to be positive, takes the range \[0, \\infty)  
    """  
    cum\_returns \= (returns \+ 1).cumprod(dim=0)  
    return (cum\_returns.max() \- cum\_returns\[-1\]) / (cum\_returns.max() \+ eps \*\*2)

def calculate\_turnover(  
        new\_weights: torch.tensor,  
        prev\_weights: torch.tensor  
):  
    """Turnover is defined to be the sum of absolute differences  
    between the new weights and the previous weights, divided by 2\.  
    Takes the range \[0, \\infty)

    This value should be minimized  
    """  
    return torch.sum(torch.abs(new\_weights \- prev\_weights)) / 2

def calculate\_objective\_func(  
        returns: torch.tensor,  
        risk\_free\_rate: torch.tensor,  
        new\_weights,  
        prev\_weights,  
        alphas \= \[1,1,1\]  
    ):  
    return (  
        a\[0\] \* calculate\_sortino(returns, risk\_free\_rate)   
        \- a\[1\] \* calculate\_max\_drawdown(returns)  
        \- a\[2\] \* calculate\_turnover(  
            new\_weights,  
            prev\_weights  
        )  
    )

\# set up  
window\_size \= 10

return\_logs \= torch.zeros(  
    size \= (returns.shape\[0\],),   
    dtype=torch.float32  
)  
rolling\_return\_list \= \[\]

\# returns.shape\[1\] \- 1 because we don't allow investing in   
\# risk free asset for the moment   
print(f"Initializing optimization...")  
weights \= torch.rand(  
    size \= (returns.shape\[1\] \- 1,),   
    requires\_grad=True  
)  
optimizer \= torch.optim.SGD(\[weights\], lr=0.5)  
weights\_log \= torch.zeros((returns.shape\[0\], returns.shape\[1\] \- 1))

for i, date in enumerate(returns.index):  
    if i % 5 \== 0:  
        print(f"Step {i} of {returns.shape\[0\]}", end \= '\\r')

    normalized\_weights \= torch.nn.functional.softmax(weights, dim=0)  
    daily\_returns \= torch.tensor(  
        returns.loc\[date\].T\[:-1\],  
        dtype=torch.float32  
    )  
    ret \= torch.dot(normalized\_weights, daily\_returns)

    \# for logging  
    return\_logs\[i\] \= ret.detach()  
    rolling\_return\_list.append(ret)

    if len(rolling\_return\_list) \> window\_size:  
        rolling\_return\_list.pop(0)  
        past\_returns \= torch.stack(rolling\_return\_list)  
        past\_rf \= torch.tensor(  
            returns.iloc\[max(0, i \- window\_size):i\]\['rf'\].values,  
            dtype=torch.float32  
        )  
        objective \= \-calculate\_objective\_func(  
            past\_returns,  
            past\_rf,  
            normalized\_weights,  
            weights\_log\[i \- 1\]  
        )  
        optimizer.zero\_grad()  
        objective.backward(retain\_graph=True)  
        optimizer.step()  
      
    weights\_log\[i\] \= normalized\_weights

6. Build out a route that given a dataframe returns the day to day returns for an equal weight portfolio  
7. Build out a route that given a dataframe returns the day to day returns for a "random portfolio" – you fully (and equally) invest in 3 randomly selected stocks on any given day  
8. Build out a unified route that, given a set of hyperparameters, start date, and end date, creates the dataframe, runs OGD, and also runs both the benchmarks, then returns all the data
9. Bundle it all into a well structured API

**Frontend Development Steps**

1. Create a file directory with images, components, data and pages. Create a global API variable that is set and can be edited for where the server is hosted  
2. Develop a header and footer for the project  
3. Create the layout for a dashboard that will take up exactly 100vh  
   1. On the right 1/4th we should have a list of our 111 stock tickers, grouped by sector. There should be a way to select our stock tickers in batches (i.e. toggle all, toggle by sector, etc) or individually  
4. On the left hand side Top 2/3, we should have 2 graphs; cumulative returns and weight evolution. These graphs must be highly reflexive, running across the run's time and necessary tickers, etc  
   2. On the top of the left hand side above the stock tickers, we should have a horizontal menu to set our 4 variables  
   3. There should also be a smart and relevant place to run "Allocate Portfolio"  
5. On the bottom 1/3rd of the left, we should have 3 graphs in a row that provide more specific statistics. I will leave it up to you to decide what these graphs should show

**Frontend Considerations**

1. We want the frontend to be as clean and modern as possible, considering our target audience is 16-24 year olds. Take heavy inspiration from the UI of Notion. Have it be by default in a "dark mode"  
2. We want the frontend to feel responsive and provide micro or fake feedback while we're waiting for the OGD as it may take quite long. Maybe run some fake simulations through geometric brownian motions while we're waiting

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