
# STAT 4830 Final Report

# Portfolio Refinement Through Iterative Sequential Modeling (PRISM)

**Team PRISM:** Dhruv Gupta, Aiden Lee, Frank Ma, Kelly Wang, Didrik Wiig-Andersen

---

## 1. The Stock Market Problem

The stock market operates under extreme uncertainty, influenced by a complex web of macroeconomic, geopolitical, and behavioral factors. Traditional portfolio strategies, while useful in theory, often fail under real-world stresses like the 2007-2008 financial, COVID-19 global lockdown, and the current rapid changes in tariff policies that the second Trump administration has been proposing.

### Our Challenge
- **Challenge:** Develop a trading algorithm that is robust to market shocks and adapts to changing market conditions.
- **Goal:** Achieve consistently strong returns relative to a "safe" baseline (e.g., index or risk-free investments) while controlling for risk measures.
- **Outcome:** Build an optimizer that actively balances returns, volatility, risk of extreme losses (drawdowns), and portfolio stability.

### Constraints We Address
- **Drawdown:** Significant drops in portfolio value erode investor confidence. Our model explicitly penalizes portfolios with high drawdown.
- **Turnover:** Excessive trading leads to high transaction costs. We penalize high turnover to maintain realistic, implementable portfolios.
- **Concentration Risk:** Investing too heavily in a few assets can be catastrophic. We introduce penalties to ensure a diversified set of investments.

---

## 2. Technical Approach

### Literature Review

This section explains fundamental portfolio management strategies that have been used since the mid-20th century. Specifically, in 1952, Harry Markowitz published a paper on Mean-Variance Optimization that revolutionized investment theory.

- **Mean-Variance Optimization (Markowitz, 1952):**
  - The founding idea of portfolio theory: balance returns vs. variance.
  - Highlights diversification benefits: adding imperfectly correlated assets reduces risk.
  - Leads to the "efficient frontier" of optimal risk-return trade-offs.

Building on Markowitz's work, William Sharpe developed a metric to evaluate portfolio performance:
  
- **Sharpe Ratio (Sharpe, 1966):**
  - $$SR = \frac{R_p - r_f}{\sigma_p}$$
  - Measures the risk-adjusted return considering both return and standard deviation.
  - The portfolio maximizing Sharpe is called the "tangency portfolio."

However, in industry, multiple metrics are incorporated for portfolio selection. In 2000, Rockafellar and Uryasev released an important paper that introduced optimization with Conditional Value-at-Risk. This was one of the first successful attempts to combine numerous metrics together.

- **Conditional Value-at-Risk (CVaR):**
  - Focuses on tail risk: the expected loss during the worst $\alpha\%$ of cases.
  - Gained popularity post-2008 crisis for modeling rare but extreme events.

Combining metrics is what we did in our project as well. To identify the metrics to combine, we performed a comprehensive literature review across three major academic databases using the following queries:

- **Web of Science:**
  ("stock" OR "equity") AND ("portfolio optimization" OR "asset allocation") AND ("risk measures" OR "risk metrics") AND ("drawdown" OR "value-at-risk" OR "conditional value-at-risk" OR liquidity OR skewness OR kurtosis) AND (optimization OR "gradient descent" OR "adam" OR "rmsprop")

- **Scopus:**
  TITLE-ABS-KEY("portfolio optimization" OR "asset allocation") AND ("risk measures" OR "risk metrics") AND ("drawdown" OR "value-at-risk" OR "conditional value-at-risk" OR liquidity OR skewness OR kurtosis) AND ("stock" OR "equity")

- **EconLit:**
  ("portfolio optimization" OR "portfolio selection") AND ("risk measures" OR "risk metrics" OR "alternative risk premia") AND ("drawdown" OR "value-at-risk" OR "conditional value-at-risk" OR "tail risk" OR "drawdown range" OR "short selling") AND ("liquidity" OR "factor-based" OR "multi-factor" OR "factor selection" OR "factor exposure")

This systematic search yielded 177 relevant articles across these databases. After analyzing this corpus of literature, we identified four key metrics that are commonly used in portfolio optimization but are underrepresented in traditional academic models: Sortino ratio, maximum drawdown, concentration penalty, and turnover costs.

---

## 3. Formulation

### Objective Function
Given our portfolio returns 

$$R_p = \sum_{i=1}^n w_{i,t}\cdot R_{i,t}$$

Our full optimization problem is:

$$
\max_{w_t} \alpha_1 \cdot Sortino_t(R_p, r_f) - \alpha_2 \cdot MaxDD_t(R_p) - \alpha_3 \cdot Turnover(w_t, w_{t-1}) - \alpha_4 \cdot CP(w_t)
$$

where:
- $\alpha_i$ are tunable hyperparameters weighting each objective.
- $w_t$ is the vector of portfolio weights at time $t$.

### Constraints

- **Budget constraint:**
  $$\sum_{i=1}^N w_{t,i} = 1$$

- **Long-only (no short selling):**
  $$w_{t,i} \geq 0 \quad \text{for all} \quad i, t$$

Additional leverage and short-selling scenarios are being explored as next steps.

---

## 4. Variable Definitions

- **$R_p$:** Portfolio returns.
- **$n$** Number of assets held in the portfolio.
- **$w_{i,t}$:** Weight assigned to asset $i$ at time $t$.
- **$R_{i,t}$:** Return of asset $i$ at time $t$.
- **$r_{f,t}$:** Risk-free rate at time $t$.
- **$m$:** Size of the rolling window for metrics calculation.
- **$\epsilon$:** Small constant added to denominators for numerical stability.

### Metric Formulas

- **Sortino Ratio:**

  $$Sortino_t(R_p, r_f) = \frac{E[R_p - r_f]}{\sigma_{\text{downside}} + \epsilon}$$
  - **Downside Variability**:
  
  $$\sigma_{\text{downside}} = \sqrt{\frac{1}{T}\sum_{t=1}^{T} \min(R_p - r_{f,t}, \text{NA})^2}$$

- **Maximum Drawdown:**

  $$MaxDD_t(R_p) = \frac{\max_{T \in [t-m,t]} CR_T - CR_t}{\max_{T \in [t-m,t]} CR_T + \epsilon}$$

- **Turnover:**

  $$Turnover(w_t, w_{t-1}) = \frac{1}{2} \sum_{i=1}^{N} |w_{i,t} - w_{i,t-1}|$$

- **Concentration Penalty (CP):**

  $$CP(w_t) = \max(ENP_{min} - ENP(w_t), 0) + \max(ENP(w_t) - ENP_{max}, 0)$$

  - **Effective Number of Positions (ENP):**

    $$ENP(w_t) = \frac{1}{\sum_{i=1}^{N} (w_{i,t})^2 + \epsilon}$$

---

## 5. Methodology

### Algorithm Selection

- **Online Gradient Descent (OGD):**
  - Sequential learning: updates portfolio one day at a time.
  - Adaptable to shocks without retraining on entire history.
  - Handles differentiable objectives like Sortino, Turnover dynamically.

### Hyperparameter Tuning

- **Objective Weights:** Balancing $\alpha_1$ through $\alpha_4$.
- **Learning Rate:** Key for stability vs. responsiveness.
- **Window Sizes:** Chosen based on trading horizons (daily, monthly, yearly).
- **Concentration Controls:** Tuning $ENP_{min}$ and $ENP_{max}$ to maintain diversification.

Explored via grid search and Bayesian optimization (Gaussian Process Minimization) to find near-optimal combinations.

---

## 6. Implementation

### Key Design Choices

- **Framework:**
  - PyTorch for automatic differentiation and efficient tensor computation.

- **Techniques:**
  - **Softmax normalization:** Enforces the budget constraint smoothly.
  - **Rolling Window:** Metrics calculated over moving historical windows.
  - **Custom Stability Fixes:** Adding epsilons to prevent divide-by-zero errors.

### Challenges Faced

- **Computational Cost:** Scaling to 100+ assets strained memory and processing.
- **Concentration Handling:** Balancing return optimization with diversification penalties required iterative tuning.
- **Drawdown Differentiability:** Maximum drawdown involves non-smooth operations, making gradient calculation difficult.

Solutions included memory-efficient batching, adaptive learning rates, and careful mathematical approximations.

---

## 7. Demonstration

We hosted a working demo on Hugging Face Spaces to showcase model behavior:
- **[Link to Demo](https://droov-opt.hf.space/)**

---

## 8. Results

### Observations

- **Full Objective:**
  - Consistently outperformed equal-weighted benchmarks.
  - Concentration in tech stocks (AMZN, NVDA) still problematic.

- **Sortino Only:**
  - High returns but unacceptable volatility.

- **Sortino + MaxDD:**
  - Returns improved; volatility still higher than desired.

- **Sortino + MaxDD + Turnover:**
  - Best balance of returns and trading stability, but minor concentration risk persisted.

### Final Performance Metrics

| Metric | Value |
|:---|:---|
| Sharpe Ratio | 0.0363 |
| Max Drawdown | 60.6% |
| Annualized Excess Return | 7.44% |

**Interpretation:** While the strategy achieves reasonable returns, the severe drawdown highlights the difficulty of fully mitigating tail risk in highly concentrated portfolios.

---

## 9. Reflections

### Primary Challenges
- Choosing and tuning objective function weights without overfitting.
- Managing concentration without overly diluting returns.
- Maintaining computational efficiency while expanding stock universe.

### Lessons Learned
- **Data Hygiene:** Proper data cleaning and synchronization is crucial.
- **PyTorch:** Migration was painful but necessary for scalability and differentiability.
- **Diversification:** Essential for practical portfolio management.

### Evolution of the Project
- **Phase 1:** Simple 7-stock Sortino optimization (Numpy).
- **Phase 2:** PyTorch OGD implementation.
- **Phase 3:** Expanded universe (109 stocks), added Turnover and ENP penalties.

### AI Assistance
- **Claude:** Helped kickstart early PyTorch setup.
- **ChatGPT:** Helped structure project milestones and refine final documentation.

---

# Conclusion
We successfully built a dynamic, multi-objective portfolio optimizer capable of adapting daily to market shifts. Although the final model is not perfect — particularly around concentration and extreme drawdowns—it reflects a significant leap from traditional static portfolio approaches. With further improvements around risk management and transaction cost modeling, PRISM could serve as a practical backbone for real-world asset management strategies.
