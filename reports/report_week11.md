# Portfolio Refinement Through Iterative Sequential Modeling (PRISM)

## 1. Problem Statement

Our goal remains to optimize a daily portfolio of assets to achieve high risk-adjusted returns, managing constraints related to leverage, drawdown, and volatility. This week, we attempted to implement hyperparameter tuning specifically through gradient descent, although we were unable to successfully get it running.

### Success Metric
No change.

### Constraints
No change.

### Data Requirements
No change.

### Potential Pitfalls
No change.

---

## 2. Technical Approach

### Mathematical Formulation
No changes were made to the underlying mathematical formulation from Week 9.

### Algorithm & PyTorch Strategy
We attempted to extend our approach by implementing gradient descent to optimize hyperparameters dynamically. Specifically, we explored gradient updates for:
- Learning rate adjustments.
- Momentum parameters.

However, due to implementation challenges, this approach was not successful in running effectively.

### Validation Methods
No change.

### Resource Requirements
No change.

---

## 3. Initial Results

### Evidence of Working Implementation
- **Hyperparameter Optimization**: Attempted implementation of hyperparameter gradient descent on:
  - Learning rate (initially ranging between 0.005 and 0.01).
  - Momentum (initially between 0.8 and 0.85).

### Performance Metrics
- **Implementation Issues**: Gradient descent on hyperparameters did not run successfully, preventing meaningful performance evaluation.

### Test Case Results
- Due to unsuccessful implementation, no concrete results were obtained from the gradient descent on hyperparameters.

### Current Limitations
- Difficulty in operationalizing gradient descent on hyperparameters.
- Implementation hurdles need further troubleshooting and exploration.

### Resource Usage Measurements
- No measurable increase due to unsuccessful implementation.

### Unexpected Challenges
- Encountered significant technical barriers in setting up a stable gradient descent optimization for hyperparameters.

---

## 4. Intermediate Results
We maintained the universe of 109 stocks and continued previous strategies. The gradient descent approach for hyperparameter tuning was explored but not successfully executed.

### Test Case Results
- No significant updates due to implementation challenges with gradient descent on hyperparameters.

### Portfolio Concentration Analysis
No changes; portfolio maintained based on previous implementations.

### Unexpected Challenges
- Difficulty in implementing stable gradient descent optimization for hyperparameters required extensive troubleshooting, which did not yield a solution.

---

## 5. Next Steps

1. **Troubleshoot Gradient Descent Implementation**
   - Further investigate and resolve technical issues preventing successful execution.

2. **Alternative Hyperparameter Tuning Approaches**
   - Explore alternative methods for adaptive hyperparameter tuning if gradient descent proves unfeasible.

3. **Enhanced Validation Techniques**
   - Continue evaluating alternative validation methods to effectively assess other adaptive hyperparameter tuning strategies.

4. **Computational Efficiency Improvements**
   - Optimize existing algorithms for computational efficiency to support future successful hyperparameter tuning implementations.

5. **Robustness to Extreme Events**
   - Continue assessing portfolio performance against extreme historical market events without dynamic hyperparameter adjustments for now.

---

# Literature Review

No change from Week 9.

