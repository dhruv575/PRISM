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
<!-- - **Data**: Historical daily/weekly returns for selected assets (10–30). -->
- **Transaction costs**: We implement a ceiling on transaction costs everyday. We scale the change in weights by a scaler such that the overall sum of net changes in weights is controlled below a specified ceiling.

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
\textbf{Maximize } \quad \alpha \cdot \frac{E[R_p - R_f]}{\sigma_d} + \beta \cdot \big(-MDD(p)\big) - \lambda \sum_i \left|w_i - w_i^{\text{prev}}\right|\,
$$

Where:
- $R_f$ is the risk-free rate, so that $\frac{E[R_p - R_f]}{\sigma_d}$ represents the Sortino ratio.
- $\alpha$ scales the impact of Sortino ratio.
- $\beta$ scales the impact of the drawdown term.
- $\lambda$ is the transaction cost penalty, which penalizes large changes in portfolio weights between periods.
- $w_i^{\text{prev}}$ denotes the weight of asset $i$ in the previous period.

The portfolio weights are subject to the following constraints: 

- If no leverage: $\sum_i w_i = 1 \quad \text{and} \quad w_i \geq 0 \quad \forall\ i$ 
- If allowing leverage: $\sum_i |w_i| \leq L_{\max}\,$ where $L_{\max}$ is the maximum allowable leverage.
- Short-selling limit: $w_i \geq -\delta \quad \forall\ i $ where $ \delta \geq 0$ specifies the maximum allowed short position per asset.

### Algorithm & PyTorch Strategy
- Represent weights $\mathbf{w}$ as a PyTorch tensor.
- Compute portfolio returns and risk measures (volatility, drawdown) within the computational graph.
- Use gradient-based methods (e.g., Adam, LBFGS) to optimize $-\text{objective}$ (because PyTorch minimizes by default).
- Right now, we are using a decaying learning rate, where the initial learning rate and the rate of decay are hyperparameters of the model.

### Validation Methods
- **In-Sample Optimization**: Train on a subset of historical data.
- **Out-of-Sample Backtest**: Test on later data (walk-forward or simple split).
- Compare results to a baseline (e.g., equal weights).

### Resource Requirements
- Python 3.8+, PyTorch, NumPy, pandas, matplotlib, yfinance.
- Sufficient CPU/GPU time for iterative optimization and backtesting with large historical stock market prices.

---

## 3. Initial Results

### Evidence of Working Implementation
- **Basic Test**: A small 7-asset dataset was loaded into our PyTorch pipeline. \
  Companies: Tesla, Google, Microsoft, Amazon, Apple, Meta, NVIDIA
- **Hyperparameters**: We experimented with random values and chose the set that resulted in the best model specifications.
- **Online Gradient Descent** \
Our testing only considers the stock market in 2022-2024 and produced weights that resulted in a portfolio that was only outperformed by a portfolio that only contained META and only continaed NVDA. With our updated fitness function using a working OGD algorithm, the score now outperforms all single investment stocks except for NVDA.

### Performance Metrics 
- **Returns**: Cumulative returns performed by the OGD model is consistently returning around 1.5x, which is above all the singular stock investments except NVDA.
- See the notebook for visuals.

### Test Case Results
- Verified the objective function calculates returns and volatility correctly.
- Observed that adding a drawdown penalty can shift weights toward lower-volatility assets.

### Current Limitations
- Minimal data usage (only 24 months of daily returns).
- An unclear transaction cost function is included in the model, which may not reflect real-world applicability.

### Resource Usage Measurements
- CPU-bound for small datasets; no GPU acceleration used yet.
- Optimization completes in ~1 second for 7 assets but could scale up with more assets.

### Unexpected Challenges
- Handling negative weights for short selling in PyTorch required a custom clip function.
- Integrating maximum drawdown in the computational graph introduced complexity in gradient calculation.

---

## 4, Intermediate results
We added more stocks to our selection universe. Instead of a toy sample with Mag 7, we expanded to 109 stocks from a range of industries. Additionally, we also added entropy within our model specification to control the number of stocks that the optimizer chooses to take a position in. 

### Test Case Results
Model identified NVDA as the stock to take the largest position in. Interestingly the model also picked up Gamestop in May 2024, about a month after its price shot up. 

### Unexpected Challenges
- The effects of adding entropy to objective function does not match what we expected. 


## 5. Next Steps

1. **Expand Data Universe**  
   - Through proper entropy / sparsity controls, we hope to eventually increase our stock universe further. Intermediate results suggest the model is somewhat successful in stock picking. 
   - Acquire a longer historical window and consider testing our model against time periods where there where sudden shocks to the market due to extreme events.
     -  We will also consider the question: How long of historical window matters?


2. **Refine Constraints**  
   - Right now, we are setting a ceiling on total transaction costs. Ideally, we would want to penalize transaction costs or turnover as part of our objective function.
   - Enforce leverage up to 1.5, short selling up to 30% of portfolio. 
   - Evaluate how these constraints interact with drawdown penalty.
   - Integrate more advanced risk measures like conditional value-at-risk.
     - Review the literature on alternative risk measures we should consider incorporating.
     - VaR, expected loss (expected loss given loss)

3. **Rolling Optimization**  
   - Implement a time-series approach to rebalance daily/monthly/quarterly.
   - We can potentially experiment with different rebalancing frequencies and compare performance. 

4. **Short Selling**
    - We know that the possibility to go short on certain assets is a key assumption in traditional mean variance portfolio analysis. We would want to test our optimizer under similar conditions.
     
6. **Transaction Costs**  
   - Add a penalty for changing weights significantly between rebalances.

7. **Advanced Validation**  
   - Perform a walk-forward validation to reduce overfitting risk.
   - Compare with multiple baselines (index funds, risk-parity strategy).
   - Test our model with real-time data implementation

8. **Other Methods to Consider**

   We need to account for cases when the weights we put on our stocks are too sparse. There should be a mechancism to add a penalty for the sum of the weights $w_i$. We need more flexibility in our objective function to improve the way we account for the risk in the stock market.
   - Online Multiplicative Weights
     - Look into using multiplicative experts.
   - Online Mirror Descent
     - Think of stocks at the sector level. Then, the goal is to maximize entropy $\sum_i w_i \cdot \log(w_i)$. Look into the Bregman divergence.

10. **Some code refactoring**
- Tidy up code base, currently we have some functions that are similar in functionality but sitting inisde / outside of our optimizer class depending on whether they are working with train or test data. 


**What We’ve Learned So Far**  
- Multi-objective optimization in finance can quickly become complex.
- PyTorch’s auto-differentiation helps but requires careful handling of constraints.
- Good data hygiene (cleaning, consistent date alignment) is critical.
- We are influencing our portfolio with hindsight bias due to the fact that we chose the 7 firms that it can invest in. The choices we made depend on our knowledge of the past, so it is necessary to remove our influence on the model.
