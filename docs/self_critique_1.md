# Self-Critique (One-Page Version)

## 1. OBSERVE
- The project successfully implements a multi-objective optimization (blending returns, volatility, and drawdown) on a small universe of seven technology-focused stocks (magnificent 7).  
- The code runs without major issues, backtests strategies, and compares them to baselines such as equal-weight and single-stock holdings.

## 2. ORIENT
- **Strengths**  
  1. Demonstrates multi-objective PyTorch optimization, including drawdown in the objective.  
  2. Provides clear performance benchmarks against simple strategies.

- **Areas for Improvement**  
  1. **Objective Function Weights**: Current ratios (e.g., 0.4–0.3–0.3) are arbitrarily chosen and need data-driven justification or tuning.  
  2. **Narrow Asset Pool**: Focusing on just seven stocks reduces diversification and may not generalize well.
  3. **Window Size**: We should test on how different window size inputs affect the performance of the algorithm

- **Critical Risks/Assumptions**  
  - Heavy reliance on historical relationships and a tech-centric set of tickers could lead to overfitting.  
  - Using the Sharpe Ratio alone overlooks asymmetrical risk (downside vs. upside). Using sortino ratio could be better.

## 3. DECIDE
1. **Adopt Sortino Ratio**: Replace or supplement the Sharpe Ratio to focus on downside volatility.  
2. **Optimize Weights**: Run experiments to justify or refine the 0.4–0.3–0.3 blend.  
3. **Broaden Universe**: Add more diverse assets (e.g., 20–30 stocks across sectors) to strengthen robustness and mitigate concentration risk. Ultimate goal would be to diversify into S&P 500.
4. **Window Size**: Experiment with different window strategies and observe its effects

## 4. ACT
- **Resource Needs**: A larger, more diverse dataset (covering multiple sectors; ≥3 years) is crucial. Additional computing resources might be needed for more comprehensive backtesting. A methodical approach to hyperparameter tuning (e.g., Bayesian optimization) will help validate revised objective weights.
