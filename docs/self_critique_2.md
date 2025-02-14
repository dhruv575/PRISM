# Self-Critique (One-Page Version)

## 1. OBSERVE
- The project successfully implements a multi-objective optimization (blending returns, volatility, and drawdown) on a small universe of seven technology-focused stocks (magnificent 7)
- Our optimizer successfully beats returns for all strategies except for full investment in Nvidia

## 2. ORIENT
- **Strengths**  
  1. Demonstrates multi-objective NumPy optimization, including drawdown in the objective. 
  2. Provides clear performance benchmarks against simple strategies
  3. Performs significantly better than equal weight strategy

- **Areas for Improvement**  
  1. **Objective Function Weights**: While we have tweaked other hyper parameters, we are yet to tweak the weights assigned to each component of the objective function 
  2. **More Diverse Assets**: Our chosen equities are all in the technology sector; we should look to branch out to ETFs representing different sectors/commodities for a more resilient portfolio
  3. **Active Trading**: We should look at APIs that would allow us to actively trade using this algorithm so we can test it out in the real world

- **Critical Risks/Assumptions**  
  - Using only technical stocks (even if we choose to focus on the S&P 500)
  - Not penalizing sparseness enough/not rewarding entropy enough

## 3. DECIDE
1. **Involve Diverse ETFs**: Replace equities with sector based ETFs and invest in those instead
2. **Optimize Weights**: Once other hyperparameters are chosen, tune the weights for our fitness function too
3. **Trade Actively**: Find an API that would allow us to trade a small portfolio (1-2k) actively to real time test

## 4. ACT
- **ETF Dataset**: Figure out ways to find more diverse dataset to allow us to access historical ETF data, allow us to diversify by sectors, and implement some way of incentivizing more diversity within our portfolio.
