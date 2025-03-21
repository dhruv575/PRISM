# Portfolio Refinement Through Iterative Sequential Modeling (PRISM)

## 1. Problem Statement

Our goal is to optimize a daily portfolio of assets to achieve high risk-adjusted returns while respecting certain constraints on leverage, drawdown, and volatility. Specifically, our project focuses on maximizing a combination of financial risk metrics and ratios while limiting maximum drawdown over a given historical period.

### Why This Matters
- Traditional mean-variance optimization oversimplifies risk by only using variance.
- Real-world portfolios must manage multiple risk dimensions (drawdown, volatility) and practical constraints (leverage, short selling) at the same time.
- Variability in the real-world and extreme events (like COVID) don't follow a nice and familiar distribution, so due to their unprediciatbility, traditional trading strategies have different tolerance levels to these unexpected events.
- By addressing these complexities, we aim to create a more realistic and robust decision-making strategy for creating a dynamic portfolio.

### Success Metric
**Fitness Score** - What goes into the score
- Sortino Ratio = $\frac{E[R_p - R_f]}{\sigma_d}$

Where: $R_p$ = Return of the portfolio | $R_f$ = Risk-free rate | $\sigma_d$ = Standard deviation of negative asset returns (i.e., downside deviation where returns below a minimum acceptable return or MAR)
- Expected returns: either estimated with mean historical return or exponentially weighted historical return. 
- Max drawdown: a strategy that generates sufficient risk adjusted returns may not be favorable if it leads to prolonged and sharp decline in portfolio value. We penalize max drawdown by substracting it from our fitness function. We use an annual window for max drawdown calculation.
- Factor Exposure - this refers to the sensitivity of an investment, portfolio, or asset to specific risk factors or systematic drivers of returns (not currently implemented)
- Portfolio Constraints - we ask the model to minimize transaction costs, minimize the change in our stock positions between days, and try to maintain stable returns over time

### Constraints
- **Leverage**: May exceed 1 but within a specified maximum (e.g., 1.5–2.0).
- **Drawdown**: Must remain below a specified percentage (e.g., 20% max drawdown).
- **Transaction costs**: We implement a ceiling on transaction costs everyday. We scale the change in weights by a scaler such that the overall sum of net changes in weights is controlled below a specified ceiling.
- **Sparsity**: We use L1 regularization to control the number of active positions in the portfolio.

### Data Requirements
- Daily price data from YFinance.
- Sufficient history to handle training and validation. The goal is to be able to test our model against extreme events like the 2008 Financial Crisis or COVID.

### Potential Pitfalls
- Overfitting to historical data (backtest bias).
- Incorrect handling of missing data or survivorship bias.
- High computational costs if too many assets or constraints are added.

---

## 2. Technical Approach

### Mathematical Formulation
Let $w_i$ denote the weight of asset $i$ in the portfolio. We define the portfolio return as 

$$R_P = \sum_{i} w_i \cdot R_i$$

The portfolio volatility $\sigma_p$ is computed as the annualized standard deviation of the daily portfolio returns, and the maximum drawdown of the portfolio, hereby abbreviated to $MDD(p)$, is computed from the cumulative returns. Given this, our objective function is

$$
\textbf{Maximize } \quad \alpha_1 \cdot \frac{E[R_p - R_f]}{\sigma_d} + \alpha_2 \cdot \big(-MDD(p)\big) - \alpha_3 \sum_i \left|w_i - w_i^{\text{prev}}\right| - \alpha_4 \sum_i |w_i|
$$

Where:
- $R_f$ is the risk-free rate, so that $\frac{E[R_p - R_f]}{\sigma_d}$ represents the Sortino ratio.
- $\alpha_1$ scales the impact of Sortino ratio.
- $\alpha_2$ scales the impact of the drawdown term.
- $\alpha_3$ is the transaction cost penalty, which penalizes large changes in portfolio weights between periods.
- $\alpha_4$ is the sparsity factor, which encourages a more concentrated portfolio through L1 regularization.
- $w_i^{\text{prev}}$ denotes the weight of asset $i$ in the previous period.

The portfolio weights are subject to the following constraints: 

- If no leverage: $\sum_i w_i = 1 \quad \text{and} \quad w_i \geq 0 \quad \forall\ i$ 
- If allowing leverage: $\sum_i |w_i| \leq L_{\max}\,$ where $L_{\max}$ is the maximum allowable leverage.
- Short-selling limit: $w_i \geq -\delta \quad \forall\ i $ where $ \delta \geq 0$ specifies the maximum allowed short position per asset.

### Algorithm & PyTorch Strategy
- Represent weights $\mathbf{w}$ as a PyTorch tensor.
- Compute portfolio returns and risk measures (volatility, drawdown) within the computational graph.
- Use online gradient descent with momentum to optimize the objective function.
- Apply a decaying learning rate, where the initial learning rate and the rate of decay are hyperparameters of the model.
- Implement L1 regularization for sparsity control.

### Validation Methods
- **In-Sample Optimization**: Train on a subset of historical data (2022-2023).
- **Out-of-Sample Backtest**: Test on later data (2024 to present).
- Compare results to a baseline (e.g., equal weights) and individual stock investments.

### Resource Requirements
- Python 3.8+, PyTorch, NumPy, pandas, matplotlib, yfinance.
- Sufficient CPU/GPU time for iterative optimization and backtesting with large historical stock market prices.

---

## 3. Initial Results

### Evidence of Working Implementation
- **Expanded Universe**: We've expanded from the original Magnificent 7 stocks to a comprehensive universe of 109 stocks across multiple sectors:
  - Technology & AI (10 stocks)
  - Financials & Banking (9 stocks)
  - Healthcare & Pharma (10 stocks)
  - Consumer Goods & Retail (10 stocks)
  - Energy & Utilities (10 stocks)
  - Industrials & Defense (10 stocks)
  - Real Estate & Infrastructure (10 stocks)
  - Transportation & Automotive (10 stocks)
  - Semiconductors (10 stocks)
  - Agriculture & Food (10 stocks)
  - Entertainment & Media (9 stocks)
  - Meme Stocks (2 stocks)

- **Hyperparameters**: We tested two different optimization strategies:
  - **OGD_Long_Window**: Higher learning rate (0.01), 0.8 momentum, 90-day window
  - **OGD_Conservative**: Lower learning rate (0.005), 0.85 momentum, 90-day window

- **Online Gradient Descent**:
  Our testing considers the stock market in 2022-2024 and produced weights that resulted in a portfolio that outperforms most individual stocks and the equal-weight strategy.

### Performance Metrics 
- **Returns**: Cumulative returns performed by the OGD models consistently outperform most single-stock investments and the equal-weight portfolio.
- **Portfolio Concentration**: We now track concentration metrics including:
  - Active positions (>0.5%)
  - Top 10 concentration percentage
  - Herfindahl-Hirschman Index (HHI) to measure portfolio concentration

### Test Case Results
- The model successfully identifies strong performers like NVDA and adjusts weights dynamically based on market conditions.
- The addition of L1 regularization helps control the number of active positions in the portfolio.
- Transaction cost penalties help reduce portfolio turnover.

### Current Limitations
- The model may still be sensitive to the choice of hyperparameters.
- The expanded universe increases computational requirements.
- Further testing across different market regimes is needed.

### Resource Usage Measurements
- CPU-bound for small datasets; no GPU acceleration used yet.
- Optimization completes in reasonable time for 109 assets but could be further optimized.

### Unexpected Challenges
- Balancing sparsity with diversification requires careful tuning of the L1 regularization parameter.
- Transaction cost modeling needs to be realistic without overly constraining the optimizer.

---

## 4. Intermediate Results
We have successfully expanded our selection universe from the Magnificent 7 to 109 stocks across various industries. We've also implemented L1 regularization to control portfolio sparsity and added transaction cost penalties to the objective function.

### Test Case Results
- The model identifies NVDA as a key position, aligning with its strong performance in 2023-2024.
- Portfolio concentration metrics show reasonable diversification while still maintaining focused positions.
- The OGD strategies demonstrate adaptability to changing market conditions.

### Portfolio Concentration Analysis
- Active positions (>0.5%): The model maintains a focused portfolio with a manageable number of positions.
- Top 10 concentration: Typically ranges from 70-90% depending on the strategy.
- HHI concentration index: Provides a quantitative measure of portfolio concentration.

### Unexpected Challenges
- The effects of L1 regularization require careful tuning to balance concentration and diversification.
- Transaction cost penalties need to be calibrated to reflect realistic trading costs without overly constraining the optimizer.

## 5. Next Steps

1. **Further Expand Data Universe**  
   - Consider adding international stocks, ETFs, and other asset classes.
   - Test the model with longer historical windows, especially during market stress periods.
   - Evaluate the optimal historical window length for different market regimes.

2. **Refine Constraints**  
   - Continue to optimize the transaction cost penalty as part of our objective function.
   - Enforce leverage up to 1.5, short selling up to 30% of portfolio. 
   - Integrate more advanced risk measures like conditional value-at-risk (CVaR).
   - Fine-tune the L1 regularization parameter to achieve optimal portfolio concentration.

3. **Rolling Optimization**  
   - Implement a time-series approach to rebalance daily/monthly/quarterly.
   - Experiment with different rebalancing frequencies and compare performance.

4. **Short Selling**
   - Test the optimizer under conditions allowing short positions.
   - Evaluate the impact of short selling constraints on portfolio performance.
     
5. **Advanced Validation**  
   - Perform a walk-forward validation to reduce overfitting risk.
   - Compare with multiple baselines (index funds, risk-parity strategy).
   - Test our model with real-time data implementation.

6. **Alternative Methods to Consider**
   - Online Multiplicative Weights
   - Online Mirror Descent
   - Consider sector-level allocation followed by stock selection within sectors.

7. **Code Refactoring**
   - Tidy up code base, currently we have some functions that are similar in functionality but sitting inside/outside of our optimizer class.
   - Improve documentation and create a more modular design.

**What We've Learned So Far**  
- Multi-objective optimization in finance requires balancing competing goals (returns, risk, transaction costs, sparsity).
- PyTorch's auto-differentiation helps but requires careful handling of constraints.
- Portfolio concentration metrics provide valuable insights into the diversification profile.
- Online gradient descent with momentum and L1 regularization offers a flexible framework for dynamic portfolio optimization.


# Literature Review

We have applied three different queries across **EconLit**, **Scopus**, and **Web of Science**. See the queries below:

| **Database**        | **Query**                                                                                                                                                                                                                                                                                                                                                |
|---------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Web of Science**  | ("stock" OR "equity") AND ("portfolio optimization" OR "asset allocation") AND ("risk measures" OR "risk metrics") AND ("drawdown" OR "value-at-risk" OR "conditional value-at-risk" OR liquidity OR skewness OR kurtosis) AND (optimization OR "gradient descent" OR "adam" OR "rmsprop")                                                               |
| **Scopus**          | TITLE-ABS-KEY( "portfolio optimization" OR "asset allocation" ) AND ( "risk measures" OR "risk metrics" ) AND ( "drawdown" OR "value-at-risk" OR "conditional value-at-risk" OR liquidity OR skewness OR kurtosis ) AND ( "stock" OR "equity" )                                                                                                         |
| **EconLit**         | ("portfolio optimization" OR "portfolio selection") AND ( "risk measures" OR "risk metrics" OR "alternative risk premia" ) AND ( "drawdown" OR "value-at-risk" OR "conditional value-at-risk" OR "tail risk" OR "drawdown range" OR "short selling" ) AND ( "liquidity" OR "factor-based" OR "multi-factor" OR "factor selection" OR "factor exposure" ) |

The queries resulted in **192 articles** which we imported into Zotero for review. The final breakdown is:

| **Web of Science** | **Scopus** | **EconLit** |
|--------------------|-----------:|------------:|
| 50                 | 120        | 7           |

We will analyze further based on the relevance of the title and abstract. Relevance is determined by whether they introduce new risk strategies we can implement in our optimization function. After that, we can see if we can incorporate them in the respective factor-based environment.
