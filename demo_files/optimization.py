import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import statsmodels.api as sm
from tqdm import tqdm

# Small epsilon to avoid division by zero
eps = 1e-6

# --- Objective function components ---
def calculate_sortino(
        returns: torch.Tensor,
        min_acceptable_return: torch.Tensor
    ):
    """Calculates the Sortino ratio."""
    if min_acceptable_return is not None:
        excess_returns = returns - min_acceptable_return
    else:
        # If no MAR provided, treat 0 as the target
        excess_returns = returns

    # Calculate downside deviation only on returns below the target
    downside_returns = torch.where(excess_returns < 0, excess_returns, torch.tensor(0.0, device=returns.device))
    downside_deviation = torch.std(downside_returns, dim=0)
    
    # More robust division - avoid division by very small numbers
    downside_deviation = torch.clamp(downside_deviation, min=eps)
    
    # Calculate Sortino ratio with better stability
    sortino = torch.mean(excess_returns, dim=0) / downside_deviation
    
    # Clip extreme values to prevent propagation of extreme gradients
    sortino = torch.clamp(sortino, min=-100.0, max=100.0)
    
    return sortino

def calculate_max_drawdown(
        returns: torch.Tensor
    ):
    """Calculates max drawdown for the duration of the returns passed.
       Max drawdown is defined to be positive, takes the range [0, \\infty).
    """
    if returns.numel() == 0:
        return torch.tensor(0.0, device=returns.device) # Handle empty tensor
    
    # Handle NaN values in returns if any
    clean_returns = torch.nan_to_num(returns, nan=0.0)
    
    cum_returns = (clean_returns + 1).cumprod(dim=0)
    peak = torch.cummax(cum_returns, dim=0).values # Use torch.cummax
    
    # Prevent division by zero or very small peaks
    safe_peak = torch.clamp(peak, min=eps)
    
    drawdown = (peak - cum_returns) / safe_peak # Calculate drawdown relative to peak
    max_drawdown = torch.max(drawdown)
    
    # Clip extreme values
    max_drawdown = torch.clamp(max_drawdown, min=0.0, max=1.0)
    
    return max_drawdown

def calculate_turnover(
        new_weights: torch.Tensor,
        prev_weights: torch.Tensor
    ):
    """Turnover is defined as the sum of absolute differences
       between new and previous weights, divided by 2.
       Takes the range [0, \\infty).
    """
    # Safe handling of NaN weights
    new_weights_safe = torch.nan_to_num(new_weights, nan=1.0/new_weights.size(0))
    prev_weights_safe = torch.nan_to_num(prev_weights, nan=1.0/prev_weights.size(0))
    
    turnover = torch.sum(torch.abs(new_weights_safe - prev_weights_safe)) / 2.0
    
    # Clip to reasonable values
    turnover = torch.clamp(turnover, min=0.0, max=1.0)
    
    return turnover

def calculate_hhi(weights: torch.Tensor):
    """Calculate Herfindahl-Hirschman Index, a measure of concentration.
    Higher values indicate more concentration (less diversification).
    """
    return torch.sum(weights ** 2)

def concentration_penalty(
        weights: torch.Tensor,
        enp_min: float = 5.0,
        enp_max: float = 20.0
    ):
    """Calculate concentration penalty based on effective number of positions (ENP).
    ENP is the inverse of HHI. This encourages having between enp_min and enp_max
    effective positions.
    """
    hhi = calculate_hhi(weights)
    enp = 1.0 / (hhi + eps)
    penalty = torch.relu(enp_min - enp) + torch.relu(enp - enp_max)
    return penalty

def calculate_objective_func(
        returns: torch.Tensor,
        risk_free_rate: torch.Tensor,
        new_weights: torch.Tensor,
        prev_weights: torch.Tensor,
        alphas = [1.0, 1.0, 0.1, 0.25],  # Default alpha values [Sortino, MaxDrawdown, Turnover, Concentration]
        enp_min: float = 5.0,
        enp_max: float = 20.0
    ):
    """Calculates the weighted objective function to be MINIMIZED.
       Note: Sortino is maximized, drawdown, turnover, and concentration are minimized.
    """
    sortino = calculate_sortino(returns, risk_free_rate)
    max_drawdown = calculate_max_drawdown(returns)
    turnover = calculate_turnover(new_weights, prev_weights)
    conc_penalty = concentration_penalty(new_weights, enp_min, enp_max)

    # Apply scaling to individual components
    sortino_scaled = torch.clamp(sortino, min=-10.0, max=10.0)
    max_drawdown_scaled = torch.clamp(max_drawdown, min=0.0, max=1.0)
    turnover_scaled = torch.clamp(turnover, min=0.0, max=1.0)
    conc_penalty_scaled = torch.clamp(conc_penalty, min=0.0, max=10.0)

    # Objective: Maximize Sortino, Minimize MaxDrawdown, Minimize Turnover, Control Concentration
    # We negate Sortino because the optimizer minimizes the objective.
    objective = (
        -alphas[0] * sortino_scaled +
        alphas[1] * max_drawdown_scaled +
        alphas[2] * turnover_scaled +
        alphas[3] * conc_penalty_scaled
    )
                
    # Ensure objective is not NaN
    if torch.isnan(objective):
        print("Warning: NaN objective detected, using default value")
        objective = torch.tensor(0.0, requires_grad=True)
        
    return objective

# --- Main OGD Optimization Function ---
def run_ogd(
        data_df: pd.DataFrame,
        window_size: int = 20,
        learning_rate: float = 0.01,
        alphas: list[float] = [1.0, 1.0, 0.1, 0.25],  # Added concentration weight
        enp_min: float = 5.0,
        enp_max: float = 20.0,
        use_tqdm: bool = True,
        factor_data: pd.DataFrame = None
    ):
    """Runs the Online Gradient Descent (OGD) portfolio optimization.

    Args:
        data_df (pd.DataFrame): DataFrame with dates as index, ticker returns as columns,
                                and a final column named 'rf' for the risk-free rate.
        window_size (int): Lookback window for objective calculation.
        learning_rate (float): Learning rate for the SGD optimizer.
        alphas (list[float]): Weights for [Sortino, MaxDrawdown, Turnover, Concentration] in the objective.
        enp_min (float): Minimum effective number of positions target.
        enp_max (float): Maximum effective number of positions target.
        use_tqdm (bool): Whether to use tqdm progress bar.
        factor_data (pd.DataFrame, optional): DataFrame with factors for CAPM/FF3 analysis.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            - weights_df: DataFrame of daily portfolio weights (dates index, tickers columns).
            - returns_series: Series of daily portfolio returns (dates index).
    """
    if data_df.empty or len(data_df) <= window_size:
        print("Warning: Dataframe too small for OGD with the given window size.")
        return pd.DataFrame(), pd.Series(dtype=float)

    # --- Add data validation ---
    # Check for NaN values in the input data
    num_nan_values = data_df.isna().sum().sum()
    if num_nan_values > 0:
        print(f"WARNING: Input data contains {num_nan_values} NaN values. Filling with 0.")
        data_df = data_df.fillna(0)
    
    # --- Print diagnostic info ---
    print(f"Data shape: {data_df.shape}")
    print(f"Sample data (first few rows):")
    print(data_df.iloc[:3, :5])  # Show first 3 rows, first 5 columns
    
    # Check for any columns with all zeros or NaNs
    zero_cols = (data_df == 0).all()
    if zero_cols.any():
        zero_count = zero_cols.sum()
        print(f"WARNING: {zero_count} columns contain all zeros.")

    # Separate stock returns and risk-free rate
    returns = data_df.drop(columns=['rf'])
    rf = data_df['rf']
    tickers = returns.columns.tolist()
    num_assets = len(tickers)
    num_days = len(data_df)

    # Convert to PyTorch tensors with explicit handling of NaN values
    # Replace NaN values with 0 during tensor conversion
    returns_tensor = torch.tensor(returns.fillna(0).values, dtype=torch.float32)
    rf_tensor = torch.tensor(rf.fillna(0).values, dtype=torch.float32)
    
    # Check if returns_tensor contains any NaN values (after conversion)
    if torch.isnan(returns_tensor).any():
        print("WARNING: returns_tensor contains NaN values after conversion. Replacing with zeros.")
        returns_tensor = torch.nan_to_num(returns_tensor, nan=0.0)

    # Initialize weights as logits (will be converted to probabilities via softmax)
    # Starting with zeros gives equal weights after softmax
    weights = torch.zeros((num_assets,), requires_grad=True)

    # Use Adam optimizer with reduced learning rate
    optimizer = torch.optim.Adam([weights], lr=learning_rate)

    # Logging structures
    weights_log = torch.zeros((num_days, num_assets), dtype=torch.float32)
    portfolio_returns_log = torch.zeros((num_days,), dtype=torch.float32)
    rolling_portfolio_returns = [] # Store recent portfolio returns for objective calc

    print(f"Starting OGD optimization for {num_days} days, {num_assets} assets...")
    
    # Initial weights distribution - equal weights
    initial_weights = torch.full((num_assets,), 1.0/num_assets)
    
    # Use tqdm for progress tracking if requested
    day_iterator = tqdm(range(num_days)) if use_tqdm else range(num_days)
    
    for i in day_iterator:
        # Check for NaN in weights and reset if needed
        if torch.isnan(weights).any():
            print(f"WARNING: NaN detected in weights at day {i}, resetting to uniform weights")
            with torch.no_grad():
                weights.copy_(torch.zeros((num_assets,)))
                
        # More restrictive clamping for numerical stability
        clamped_weights = torch.clamp(weights, min=-5, max=5)
        normalized_weights = torch.nn.functional.softmax(clamped_weights, dim=0)
        
        # Verify normalized weights are valid probabilities
        if torch.isnan(normalized_weights).any() or torch.sum(normalized_weights) < 0.99:
            print(f"WARNING: Invalid normalized weights at day {i}, using uniform weights")
            normalized_weights = initial_weights.clone()

        # Get daily asset returns and check for NaN values
        daily_asset_returns = returns_tensor[i, :]
        if torch.isnan(daily_asset_returns).any():
            print(f"WARNING: NaN detected in asset returns at day {i}, replacing with zeros")
            daily_asset_returns = torch.nan_to_num(daily_asset_returns, nan=0.0)
        
        # Calculate portfolio return for the current day
        daily_portfolio_return = torch.dot(normalized_weights, daily_asset_returns)

        # Check for NaN in portfolio return
        if torch.isnan(daily_portfolio_return):
            print(f"WARNING: NaN detected in portfolio return at day {i}, using zero")
            daily_portfolio_return = torch.tensor(0.0)
            
            # Debug information - print sample weights and returns to diagnose the issue
            if i < 5 or i % 50 == 0:  # Print for first few days and then occasionally
                print(f"  Debug info for day {i}:")
                print(f"  Sample weights: {normalized_weights[:5].tolist()}")
                print(f"  Sample returns: {daily_asset_returns[:5].tolist()}")
                print(f"  Sum of weights: {torch.sum(normalized_weights).item()}")
                nan_count = torch.isnan(daily_asset_returns).sum().item()
                print(f"  NaN count in returns: {nan_count}/{len(daily_asset_returns)}")

        # Log weights and returns (use detach() to prevent tracking history)
        weights_log[i, :] = normalized_weights.detach()
        portfolio_returns_log[i] = daily_portfolio_return.detach()

        # Add current return to rolling list for objective calculation
        # Detach returns when storing to break gradient history
        rolling_portfolio_returns.append(daily_portfolio_return.detach())

        # --- Objective Calculation and Optimization Step ---
        # Wait until we have enough data for the lookback window
        if len(rolling_portfolio_returns) > window_size:
            rolling_portfolio_returns.pop(0) # Remove oldest return

            # Verify we don't have all zeros in our portfolio returns
            all_zeros = all(r.item() == 0 for r in rolling_portfolio_returns)
            if all_zeros:
                print(f"WARNING: All portfolio returns are zero at day {i}, skipping optimization")
                continue

            # Prepare tensors for objective function
            past_portfolio_returns = torch.stack(rolling_portfolio_returns[:-1] + [daily_portfolio_return])
            
            # Get corresponding risk-free rates for the window
            start_idx = max(0, i - window_size + 1)
            past_rf = rf_tensor[start_idx : i + 1]

            # Get previous day's weights for turnover calculation
            prev_weights = weights_log[i-1, :] if i > 0 else normalized_weights.detach()

            # Zero out gradients before computation
            optimizer.zero_grad()
            
            try:
                # Recompute normalized weights for fresh gradient computation
                clamped_weights = torch.clamp(weights, min=-5, max=5)
                current_norm_weights = torch.nn.functional.softmax(clamped_weights, dim=0)
                
                # Recalculate today's return for gradient computation
                current_return = torch.dot(current_norm_weights, daily_asset_returns)
                
                # Create list with detached historical returns + current gradient-connected return
                historical_returns = rolling_portfolio_returns[:-1]
                new_returns_list = historical_returns + [current_return]
                past_portfolio_returns = torch.stack(new_returns_list)
                
                # Calculate objective with robust error handling
                objective = calculate_objective_func(
                    past_portfolio_returns,
                    past_rf,
                    current_norm_weights,
                    prev_weights,
                    alphas,
                    enp_min,
                    enp_max
                )
                
                # Check if objective computation produced valid result
                if not torch.isnan(objective):
                    # Check objective is not just a default zero
                    if objective.item() != 0.0 or i % 50 == 0:  # Allow some zeros through for logging
                        # Compute and apply gradients
                        objective.backward()
                        
                        # --- Enhanced Logging --- 
                        log_interval = 50
                        if (i + 1) % log_interval == 0 or num_days - (i + 1) < 5:
                            if not use_tqdm:  # Don't print logs if using tqdm to avoid cluttering
                                print(f"\n--- Step {i+1}/{num_days} Log ---")
                                print(f"  Objective: {objective.item():.6f}")
                                
                                # Log average gradient magnitude rather than all gradients
                                if weights.grad is not None:
                                    avg_grad = torch.mean(torch.abs(weights.grad)).item()
                                    print(f"  Average Gradient Magnitude: {avg_grad:.6f}")
                                
                                # Record some sample weights before update
                                weights_before = weights.detach().clone()
                                
                                # Apply gradient update
                                optimizer.step()
                                
                                # Record weights after update
                                weights_after = weights.detach().clone()
                                weight_change = torch.sum(torch.abs(weights_after - weights_before)).item()
                                print(f"  Weight Change (Sum Abs): {weight_change:.6f}")
                                
                                # Display a few normalized weights as a sample
                                print(f"  Sample Normalized Weights: {[f'{w:.4f}' for w in normalized_weights[:5].tolist()]}")
                        else:
                            # Update weights without detailed logging
                            optimizer.step()
                        
                        # Apply gradient clipping after optimizer step
                        with torch.no_grad():
                            if weights.grad is not None and torch.isnan(weights.grad).any():
                                print(f"  WARNING: NaN gradient detected at day {i}, zeroing gradients")
                                weights.grad.zero_()
                    else:
                        if not use_tqdm:
                            print(f"  WARNING: Zero objective at day {i}, skipping gradient update")
                else:
                    if not use_tqdm:
                        print(f"  WARNING: NaN objective at day {i}, skipping gradient update")
                    
            except Exception as e:
                print(f"  Optimization error at day {i}: {e}")
                # Skip this day rather than propagating errors

    print("OGD optimization finished.")
    
    # Final check for validity of results
    if torch.isnan(weights_log).any():
        print("WARNING: Final weights contain NaN values")
        weights_log = torch.nan_to_num(weights_log, nan=1.0/num_assets)
    
    if torch.isnan(portfolio_returns_log).any():
        print("WARNING: Final portfolio returns contain NaN values")
        portfolio_returns_log = torch.nan_to_num(portfolio_returns_log, nan=0.0)

    # Convert logs back to pandas DataFrames/Series with original index
    weights_df = pd.DataFrame(weights_log.numpy(), index=data_df.index, columns=tickers)
    returns_series = pd.Series(portfolio_returns_log.numpy(), index=data_df.index, name="PortfolioReturn")

    return weights_df, returns_series

# --- Analysis Functions ---
def compute_sharpe(returns_series, rf_series, annualization_factor=252):
    """Compute annualized Sharpe ratio."""
    excess = returns_series - rf_series
    annual_excess_return = np.mean(excess) * annualization_factor
    annual_volatility = np.std(excess) * np.sqrt(annualization_factor)
    return annual_excess_return / (annual_volatility + eps)

def compute_max_drawdown(returns_series):
    """Compute maximum drawdown."""
    cr = np.cumprod(returns_series + 1)
    peak = np.maximum.accumulate(cr)
    return np.max((peak - cr) / (peak + eps))

def compute_alpha(returns_series, rf_series, factor_data, model="CAPM"):
    """Compute alpha using either CAPM or Fama-French 3-factor model.
    
    Args:
        returns_series: Portfolio returns series
        rf_series: Risk-free rate series
        factor_data: DataFrame with factor returns (must include 'mktrf' for CAPM, 
                    and 'smb', 'hml' for FF3)
        model: 'CAPM' or 'FF3'
        
    Returns:
        tuple: (alpha, regression_result)
    """
    y = np.asarray(returns_series - rf_series)
    
    if model == "CAPM":
        X = np.asarray(factor_data[["mktrf"]])
    elif model == "FF3":
        X = np.asarray(factor_data[["mktrf", "smb", "hml"]])
    else:
        raise ValueError("Model must be 'CAPM' or 'FF3'")

    X = sm.add_constant(X)
    result = sm.OLS(y, X).fit()
    return result.params[0], result

# --- Visualization Functions ---
def plot_optimization_results(
        opt_returns_series, 
        weights_df, 
        benchmark_returns=None, 
        top_n=5, 
        title_suffix=""
    ):
    """Plot optimization results with comparison to benchmarks.
    
    Args:
        opt_returns_series: Series of optimized portfolio returns
        weights_df: DataFrame of weights over time
        benchmark_returns: Dict of benchmark return series {name: series}
        top_n: Number of top assets to highlight in weights plot
        title_suffix: Additional text to add to plot titles
    """
    # Convert to numpy for plotting
    dates = opt_returns_series.index
    opt_returns = opt_returns_series.values
    weights_np = weights_df.values
    
    # Create plot with return distribution and cumulative returns
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Return distribution
    axes[0].hist(opt_returns, bins=50, alpha=0.5, label='Optimized', color='red')
    
    # Cumulative returns
    axes[1].plot(dates, np.cumprod(opt_returns + 1), label='Optimized', color='red')

    # Add benchmarks if provided
    if benchmark_returns:
        for name, b_returns in benchmark_returns.items():
            axes[0].hist(b_returns, bins=50, alpha=0.5, label=name)
            axes[1].plot(dates, np.cumprod(b_returns + 1), label=name)

    axes[0].set_title('Return Distribution')
    axes[0].legend()
    
    axes[1].set_title('Cumulative Returns')
    axes[1].legend()
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    fig.suptitle(f"Performance Comparison {title_suffix}", fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Create plot with weights evolution and distribution
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Weight evolution
    top_assets_idx = np.argsort(weights_np[-1])[-top_n:]
    for i in range(weights_np.shape[1]):
        label = weights_df.columns[i] if i in top_assets_idx else None
        lw = 2 if i in top_assets_idx else 0.3
        alpha = 0.8 if i in top_assets_idx else 0.3
        axes[0].plot(dates, weights_np[:, i], label=label, linewidth=lw, alpha=alpha)

    axes[0].xaxis.set_major_locator(mdates.YearLocator())
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axes[0].set_title("Weights Over Time")
    axes[0].legend()

    # Weight distribution
    axes[1].hist(weights_np[-1], bins=100, log=True, color='blue', alpha=0.7)
    axes[1].set_title("Final Day Weight Distribution")
    
    plt.tight_layout()
    plt.show()
    
    # Return effective number of positions over time
    enp_series = 1.0 / np.sum(weights_np ** 2, axis=1)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates, enp_series)
    ax.set_title("Effective Number of Positions Over Time")
    ax.set_ylabel("ENP")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.tight_layout()
    plt.show()
    
    return None 