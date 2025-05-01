import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-interactive plotting
import json
import os
from fastapi import FastAPI, Request, Response, Body
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import numpy as np
import datetime
import math

from utils import load_data, filter_data, load_factor_data, filter_factors
from optimization import run_ogd, plot_optimization_results, compute_sharpe, compute_max_drawdown, compute_alpha
# Import benchmark functions and metrics
from benchmarks import (
    run_equal_weight,
    run_random_portfolio,
    calculate_cumulative_returns,
    calculate_performance_metrics
)

# --- Custom JSON Encoder to handle inf and NaN values ---
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, float):
            if math.isnan(obj):
                return None  # Convert NaN to null
            if math.isinf(obj):
                if obj > 0:
                    return 1.0e308  # Very large number instead of infinity
                else:
                    return -1.0e308  # Very negative number instead of -infinity
        return super().default(obj)

# --- Function to safely convert values for JSON ---
def safe_json_value(value):
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
    return value

# --- Load Data Globally ---
stock_data_df, rf_data_df = load_data()
factor_data_df = load_factor_data()

if stock_data_df is None or rf_data_df is None:
    print("Exiting application due to data loading error.")
    exit()

# --- Create FastAPI app ---
app = FastAPI()

# --- Enable CORS for all origins ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# --- Setup Static Files ---
static_dir = Path(__file__).parent / "assets" / "static"
if not static_dir.exists():
    static_dir.mkdir(parents=True, exist_ok=True)
    
# Mount static files with a name that's more unique to avoid conflicts
app.mount("/portfolio_static", StaticFiles(directory=static_dir), name="portfolio_static")

# For backward compatibility, also mount at /static
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# --- Main Optimization Pipeline Function ---
def run_optimization_pipeline(
    start_date, end_date, tickers_str,
    window_size, learning_rate,
    alpha_sortino, alpha_max_drawdown, alpha_turnover, alpha_concentration=0.25,
    enp_min=5.0, enp_max=20.0
):
    """Runs the full pipeline: filter data -> run OGD & benchmarks -> calculate metrics & plots -> return results."""
    if not tickers_str:
        tickers = None
    else:
        tickers = [t.strip().upper() for t in tickers_str.split(',')]
    try:
        window_size = int(window_size)
        learning_rate = float(learning_rate)
        alphas = [
            float(alpha_sortino),
            float(alpha_max_drawdown),
            float(alpha_turnover),
            float(alpha_concentration)
        ]
        enp_min = float(enp_min)
        enp_max = float(enp_max)
    except ValueError as e:
        return {"error": f"Invalid hyperparameter input. Details: {e}"}

    print(f"Filtering data: Start={start_date}, End={end_date}, Tickers={tickers}")
    # 2. Filter Data
    filtered_df = filter_data(stock_data_df, rf_data_df, start_date, end_date, tickers)
    filtered_factors = filter_factors(factor_data_df, start_date, end_date)

    if filtered_df is None or filtered_df.empty:
        return {"error": "Filtering resulted in empty data. Cannot run."}
    
    # Extract risk-free series for metric calculation
    rf_series = filtered_df['rf']

    print(f"Running OGD: Window={window_size}, LR={learning_rate}, Alphas={alphas}, ENP Range={enp_min}-{enp_max}")
    # 3. Run OGD with enhanced parameters
    try:
        ogd_weights_df, ogd_returns_series = run_ogd(
            filtered_df, window_size=window_size, learning_rate=learning_rate, 
            alphas=alphas, enp_min=enp_min, enp_max=enp_max, use_tqdm=True,
            factor_data=filtered_factors
        )
        if ogd_weights_df.empty or ogd_returns_series.empty:
             return {"error": "OGD failed or returned empty results."}
    except Exception as e:
        return {"error": f"OGD optimization failed: {str(e)}"}

    print("Running Benchmarks...")
    # 4. Run Benchmarks
    try:
        equal_weight_returns = run_equal_weight(filtered_df)
        random_portfolio_returns = run_random_portfolio(filtered_df)
    except Exception as e:
        return {"error": f"Benchmark calculation failed: {str(e)}"}

    # 5. Calculate Metrics & Cumulative Returns
    try:
        ogd_metrics = calculate_performance_metrics(ogd_returns_series, rf_series)
        ew_metrics = calculate_performance_metrics(equal_weight_returns, rf_series)
        rp_metrics = calculate_performance_metrics(random_portfolio_returns, rf_series)
    except Exception as e:
        return {"error": f"Performance metrics calculation failed: {str(e)}"}

    # Calculate factor model alphas if factor data is available
    factor_metrics = {}
    if filtered_factors is not None and not filtered_factors.empty:
        try:
            # Align dates between returns and factors
            common_dates = filtered_factors.index.intersection(ogd_returns_series.index)
            if len(common_dates) > 30:  # Ensure we have enough data points for regression
                # Calculate CAPM and FF3 alphas for each strategy
                ogd_capm_alpha, _ = compute_alpha(
                    ogd_returns_series.loc[common_dates], 
                    rf_series.loc[common_dates], 
                    filtered_factors.loc[common_dates], 
                    model="CAPM"
                )
                ogd_ff3_alpha, _ = compute_alpha(
                    ogd_returns_series.loc[common_dates], 
                    rf_series.loc[common_dates], 
                    filtered_factors.loc[common_dates], 
                    model="FF3"
                )
                
                ew_capm_alpha, _ = compute_alpha(
                    equal_weight_returns.loc[common_dates], 
                    rf_series.loc[common_dates], 
                    filtered_factors.loc[common_dates], 
                    model="CAPM"
                )
                ew_ff3_alpha, _ = compute_alpha(
                    equal_weight_returns.loc[common_dates], 
                    rf_series.loc[common_dates], 
                    filtered_factors.loc[common_dates], 
                    model="FF3"
                )
                
                rp_capm_alpha, _ = compute_alpha(
                    random_portfolio_returns.loc[common_dates], 
                    rf_series.loc[common_dates], 
                    filtered_factors.loc[common_dates], 
                    model="CAPM"
                )
                rp_ff3_alpha, _ = compute_alpha(
                    random_portfolio_returns.loc[common_dates], 
                    rf_series.loc[common_dates], 
                    filtered_factors.loc[common_dates], 
                    model="FF3"
                )
                
                # Store alphas in dictionary (annualize them)
                factor_metrics = {
                    "ogd": {
                        "capm_alpha": safe_json_value(float(ogd_capm_alpha * 252)),  # Annualize alpha
                        "ff3_alpha": safe_json_value(float(ogd_ff3_alpha * 252))
                    },
                    "equal_weight": {
                        "capm_alpha": safe_json_value(float(ew_capm_alpha * 252)),
                        "ff3_alpha": safe_json_value(float(ew_ff3_alpha * 252))
                    },
                    "random": {
                        "capm_alpha": safe_json_value(float(rp_capm_alpha * 252)),
                        "ff3_alpha": safe_json_value(float(rp_ff3_alpha * 252))
                    }
                }
            else:
                print(f"Warning: Not enough common dates between returns and factors. Factor analysis skipped.")
                factor_metrics = None
        except Exception as e:
            print(f"Factor metrics calculation failed: {str(e)}")
            factor_metrics = None
    else:
        print("Factor data not available. Factor analysis skipped.")
        factor_metrics = None

    # Calculate cumulative returns for charts
    try:
        ogd_cumulative = calculate_cumulative_returns(ogd_returns_series)
        ew_cumulative = calculate_cumulative_returns(equal_weight_returns)
        rp_cumulative = calculate_cumulative_returns(random_portfolio_returns)
    except Exception as e:
        return {"error": f"Cumulative returns calculation failed: {str(e)}"}

    # Convert cumulative returns to chart-friendly format with safe values
    ogd_returns_data = []
    ew_returns_data = []
    rp_returns_data = []
    
    try:
        for date, value in ogd_cumulative.items():
            safe_value = safe_json_value(float(value))
            if safe_value is not None:  # Only include valid values
                ogd_returns_data.append({"date": date.strftime("%Y-%m-%d"), "value": safe_value})
                
        for date, value in ew_cumulative.items():
            safe_value = safe_json_value(float(value))
            if safe_value is not None:
                ew_returns_data.append({"date": date.strftime("%Y-%m-%d"), "value": safe_value})
                
        for date, value in rp_cumulative.items():
            safe_value = safe_json_value(float(value))
            if safe_value is not None:
                rp_returns_data.append({"date": date.strftime("%Y-%m-%d"), "value": safe_value})
    except Exception as e:
        return {"error": f"Error formatting return data: {str(e)}"}

    # Convert weights to chart-friendly format with safe values
    weights_data = []
    try:
        for date, row in ogd_weights_df.iterrows():
            # Filter out very small weights and ensure all values are JSON safe
            significant_weights = {}
            for ticker, weight in row.items():
                if weight > 0.01:  # Only include weights > 1%
                    safe_weight = safe_json_value(float(weight))
                    if safe_weight is not None:
                        significant_weights[ticker] = safe_weight
                        
            weights_data.append({
                "date": date.strftime("%Y-%m-%d"),
                "weights": significant_weights
            })
    except Exception as e:
        return {"error": f"Error formatting weight data: {str(e)}"}

    # Calculate HHI (concentration) over time with safe values
    hhi_data = []
    enp_data = []
    try:
        hhi_values = [(ogd_weights_df.loc[date] ** 2).sum() for date in ogd_weights_df.index]
        enp_values = [1.0 / hhi if hhi > 0 else None for hhi in hhi_values]
        
        for date, value in zip(ogd_weights_df.index, hhi_values):
            safe_value = safe_json_value(float(value))
            if safe_value is not None:
                hhi_data.append({"date": date.strftime("%Y-%m-%d"), "value": safe_value})
                
        for date, value in zip(ogd_weights_df.index, enp_values):
            safe_value = safe_json_value(float(value) if value is not None else None)
            if safe_value is not None:
                enp_data.append({"date": date.strftime("%Y-%m-%d"), "value": safe_value})
    except Exception as e:
        return {"error": f"Error calculating concentration metrics: {str(e)}"}

    # Create the result dictionary with safe values for all metrics
    try:
        result = {
            "success": True,
            "cumulative_returns": {
                "ogd": ogd_returns_data,
                "equal_weight": ew_returns_data,
                "random": rp_returns_data
            },
            "weights": weights_data,
            "concentration": {
                "hhi": hhi_data,
                "enp": enp_data
            },
            "metrics": {
                "ogd": {
                    "sharpe": safe_json_value(float(ogd_metrics["Annualized Sharpe Ratio"])),
                    "max_drawdown": safe_json_value(float(ogd_metrics["Max Drawdown"])),
                    "cumulative_return": safe_json_value(float(ogd_metrics["Cumulative Return"]))
                },
                "equal_weight": {
                    "sharpe": safe_json_value(float(ew_metrics["Annualized Sharpe Ratio"])),
                    "max_drawdown": safe_json_value(float(ew_metrics["Max Drawdown"])),
                    "cumulative_return": safe_json_value(float(ew_metrics["Cumulative Return"]))
                },
                "random": {
                    "sharpe": safe_json_value(float(rp_metrics["Annualized Sharpe Ratio"])),
                    "max_drawdown": safe_json_value(float(rp_metrics["Max Drawdown"])),
                    "cumulative_return": safe_json_value(float(rp_metrics["Cumulative Return"]))
                }
            }
        }
    except Exception as e:
        return {"error": f"Error creating result data structure: {str(e)}"}
    
    # Add factor metrics if available
    if factor_metrics:
        result["factor_metrics"] = factor_metrics
        
    return result

# --- API Endpoints ---
@app.get("/")
async def serve_frontend():
    """Serve the custom frontend HTML."""
    html_path = Path(__file__).parent / "assets" / "static" / "index.html"
    if html_path.exists():
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Replace static paths to use the new mount point
            content = content.replace('href="/static/', 'href="/portfolio_static/')
            content = content.replace('src="/static/', 'src="/portfolio_static/')
        return HTMLResponse(content=content)
    else:
        return {"error": "Frontend HTML file not found"}

@app.get("/index.html")
async def serve_index():
    """Serve the index page HTML."""
    return await serve_frontend()

@app.get("/education.html")
async def serve_education():
    """Serve the education page HTML."""
    html_path = Path(__file__).parent / "assets" / "static" / "education.html"
    if html_path.exists():
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Replace static paths to use the new mount point
            content = content.replace('href="/static/', 'href="/portfolio_static/')
            content = content.replace('src="/static/', 'src="/portfolio_static/')
        return HTMLResponse(content=content)
    else:
        return {"error": "Education HTML file not found"}

# Add a direct full-page access endpoint for HuggingFace Spaces
@app.get("/fullpage")
async def serve_fullpage():
    """Serve the fullpage HTML for standalone view."""
    html_path = Path(__file__).parent / "assets" / "static" / "fullpage.html"
    if html_path.exists():
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return HTMLResponse(content=content)
    else:
        # Fallback to redirecting to the main app
        return RedirectResponse(url="/")

@app.get("/fullpage/education")
async def serve_fullpage_education():
    """Serve the fullpage HTML for education view."""
    html_path = Path(__file__).parent / "assets" / "static" / "fullpage.html"
    if html_path.exists():
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return HTMLResponse(content=content)
    else:
        # Fallback to redirecting to the education page
        return RedirectResponse(url="/education.html")

@app.get("/api/tickers_by_sector")
async def get_tickers_by_sector():
    """Return the tickers grouped by sector."""
    json_path = Path(__file__).parent / "data" / "tickers_by_sector.json"
    if json_path.exists():
        with open(json_path) as f:
            return json.load(f)
    else:
        # Fallback to generating sectors from available tickers
        tickers = stock_data_df.columns.tolist()
        if 'rf' in tickers:
            tickers.remove('rf')
        return [{"sector": "All Available Tickers", "tickers": tickers}]

@app.post("/api/run_optimization")
async def api_run_optimization(data: dict = Body(...)):
    """API endpoint for running the optimization pipeline."""
    try:
        result = run_optimization_pipeline(
            start_date=data.get('start_date'),
            end_date=data.get('end_date'),
            tickers_str=data.get('tickers', ''),
            window_size=data.get('window_size', 20),
            learning_rate=data.get('learning_rate', 0.1),
            alpha_sortino=data.get('alpha_sortino', 1.0),
            alpha_max_drawdown=data.get('alpha_max_drawdown', 1.0),
            alpha_turnover=data.get('alpha_turnover', 0.1),
            alpha_concentration=data.get('alpha_concentration', 0.25),
            enp_min=data.get('enp_min', 5.0),
            enp_max=data.get('enp_max', 20.0)
        )
        
        # Check if there was an error
        if "error" in result:
            return result
        
        # Handle NaN and Infinity values in metrics
        if 'metrics' in result:
            for strategy in result['metrics']:
                for metric, value in result['metrics'][strategy].items():
                    result['metrics'][strategy][metric] = safe_json_value(value)
        
        # Check if factor metrics exist and merge them into the metrics
        if 'factor_metrics' in result:
            # Add factor metrics to each strategy's metrics with safe values
            for strategy in result['factor_metrics']:
                for metric, value in result['factor_metrics'][strategy].items():
                    safe_value = safe_json_value(value)
                    result['metrics'][strategy][metric] = safe_value
            
            # Remove the separate factor_metrics key since we've merged it
            del result['factor_metrics']
        
        # Use custom JSON encoder by pre-encoding the content
        content = json.dumps(result, cls=CustomJSONEncoder)
        return Response(content=content, media_type="application/json")
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

# --- Gradio Interface ---
# Create a custom dark theme for Gradio
dark_theme = gr.themes.Monochrome(
    primary_hue="indigo",
    secondary_hue="slate",
    neutral_hue="slate",
    radius_size=gr.themes.sizes.radius_sm,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"]
)

with gr.Blocks(theme=dark_theme) as demo:
    gr.Markdown("""# Portfolio Optimization with OGD
    *Optimize your portfolio using Online Gradient Descent and compare against benchmarks.*""")

    # Add a link to the custom frontend
    gr.Markdown("""
    ## View Enhanced UI
    
    Try our enhanced, modern UI with interactive charts and stock selection:
    
    * [Open Modern Interface](/)
    
    Below is the basic Gradio interface for quick testing:
    """)

    with gr.Row():
        with gr.Column(scale=1): # Input Column
            gr.Markdown("### Configure Simulation")
            with gr.Accordion("Data Selection", open=True): # Group data inputs
                start_date_input = gr.Textbox(label="Start Date (YYYY-MM-DD)", placeholder="Default: Earliest", info="Leave blank for earliest available date.")
                end_date_input = gr.Textbox(label="End Date (YYYY-MM-DD)", placeholder="Default: Latest", info="Leave blank for latest available date.")
                tickers_input = gr.Textbox(
                    label="Tickers (comma-separated)",
                    placeholder="e.g., AAPL, MSFT, GOOGL",
                    info="Leave blank to use all available tickers in the date range."
                )
            
            with gr.Accordion("OGD Hyperparameters", open=True): # Group hyperparameters
                window_size_input = gr.Number(label="Lookback Window (days)", value=20, minimum=5, step=1, info="Days of past returns used for optimization.")
                learning_rate_input = gr.Number(label="Learning Rate", value=0.1, minimum=0.001, info="Step size for gradient updates.")
                
                gr.Markdown("##### Objective Function Weights (Alphas)")
                alpha_sortino_input = gr.Number(label="Sortino Ratio Weight", value=1.0, minimum=0, info="Emphasis on maximizing risk-adjusted returns (downside risk).")
                alpha_max_drawdown_input = gr.Number(label="Max Drawdown Weight", value=1.0, minimum=0, info="Emphasis on minimizing the largest peak-to-trough decline.")
                alpha_turnover_input = gr.Number(label="Turnover Weight", value=0.1, minimum=0, info="Emphasis on minimizing trading frequency/costs.")
                alpha_concentration_input = gr.Number(label="Concentration Weight", value=0.25, minimum=0, info="Emphasis on controlling portfolio concentration.")
                
                with gr.Accordion("Advanced Settings", open=False):
                    enp_min_input = gr.Number(label="Min Effective Positions", value=5.0, minimum=1.0, info="Minimum target for effective number of positions.")
                    enp_max_input = gr.Number(label="Max Effective Positions", value=20.0, minimum=1.0, info="Maximum target for effective number of positions.")

            run_button = gr.Button("Run Optimization", variant="primary", scale=1) # Made button full width within column

        with gr.Column(scale=3): # Output Column
            gr.Markdown("### Results")
            # Output components:
            run_status_text = gr.Textbox(label="Run Status", interactive=False, lines=1)
            metrics_output_df = gr.DataFrame(label="Performance Metrics Summary", interactive=False)
            plot_output = gr.Plot(label="Cumulative Returns Comparison")
            weights_output_df = gr.DataFrame(label="OGD Portfolio Weights (Daily)", interactive=False) # Removed height parameter


    # This is the same function as before but wrapped to match the Gradio interface
    def gradio_run_optimization(
        start_date, end_date, tickers_str,
        window_size, learning_rate,
        alpha_sortino, alpha_max_drawdown, alpha_turnover, alpha_concentration,
        enp_min, enp_max
    ):
        result = run_optimization_pipeline(
            start_date, end_date, tickers_str,
            window_size, learning_rate,
            alpha_sortino, alpha_max_drawdown, alpha_turnover, alpha_concentration,
            enp_min, enp_max
        )
        
        if "error" in result:
            return result["error"], None, None, None, None
        
        # Create metrics dataframe with standard metrics
        metrics_data = {
            'OGD Portfolio': {
                'Annualized Sharpe Ratio': result['metrics']['ogd']['sharpe'],
                'Max Drawdown': result['metrics']['ogd']['max_drawdown'],
                'Cumulative Return': result['metrics']['ogd']['cumulative_return']
            },
            'Equal Weight': {
                'Annualized Sharpe Ratio': result['metrics']['equal_weight']['sharpe'],
                'Max Drawdown': result['metrics']['equal_weight']['max_drawdown'],
                'Cumulative Return': result['metrics']['equal_weight']['cumulative_return']
            },
            'Random Portfolio': {
                'Annualized Sharpe Ratio': result['metrics']['random']['sharpe'],
                'Max Drawdown': result['metrics']['random']['max_drawdown'],
                'Cumulative Return': result['metrics']['random']['cumulative_return']
            }
        }
        
        # Add factor metrics if available
        if "factor_metrics" in result:
            # Add CAPM and FF3 alphas for each strategy
            metrics_data['OGD Portfolio']['CAPM Alpha (Ann.)'] = result['factor_metrics']['ogd']['capm_alpha']
            metrics_data['OGD Portfolio']['FF3 Alpha (Ann.)'] = result['factor_metrics']['ogd']['ff3_alpha']
            metrics_data['Equal Weight']['CAPM Alpha (Ann.)'] = result['factor_metrics']['equal_weight']['capm_alpha']
            metrics_data['Equal Weight']['FF3 Alpha (Ann.)'] = result['factor_metrics']['equal_weight']['ff3_alpha']
            metrics_data['Random Portfolio']['CAPM Alpha (Ann.)'] = result['factor_metrics']['random']['capm_alpha']
            metrics_data['Random Portfolio']['FF3 Alpha (Ann.)'] = result['factor_metrics']['random']['ff3_alpha']
        
        # Create the metrics dataframe
        metrics_df = pd.DataFrame(metrics_data).T
        
        # Create the matplotlib plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Convert API-friendly format back to series for plotting
        ogd_data = pd.Series({datetime.datetime.strptime(d['date'], '%Y-%m-%d').date(): d['value'] 
                              for d in result['cumulative_returns']['ogd']})
        ew_data = pd.Series({datetime.datetime.strptime(d['date'], '%Y-%m-%d').date(): d['value'] 
                            for d in result['cumulative_returns']['equal_weight']})
        rp_data = pd.Series({datetime.datetime.strptime(d['date'], '%Y-%m-%d').date(): d['value'] 
                            for d in result['cumulative_returns']['random']})
        
        ogd_data.plot(ax=ax, label='OGD Portfolio')
        ew_data.plot(ax=ax, label='Equal Weight')
        rp_data.plot(ax=ax, label='Random Portfolio')
        ax.set_title('Cumulative Portfolio Returns')
        ax.set_ylabel('Cumulative Return')
        ax.set_xlabel('Date')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        
        # Convert weights data back to DataFrame for Gradio
        weights_df = pd.DataFrame()
        for day_data in result['weights']:
            date = datetime.datetime.strptime(day_data['date'], '%Y-%m-%d').date()
            weights_df = pd.concat([weights_df, pd.Series(day_data['weights'], name=date)])
        
        # Add ENP data
        if 'concentration' in result:
            enp_series = pd.Series({datetime.datetime.strptime(d['date'], '%Y-%m-%d').date(): d['value'] 
                                for d in result['concentration']['enp']})
            fig2, ax2 = plt.subplots(figsize=(10, 3))
            enp_series.plot(ax=ax2)
            ax2.set_title('Effective Number of Positions')
            ax2.set_ylabel('ENP')
            ax2.grid(True)
            ax2.axhline(y=enp_min, color='r', linestyle='--', alpha=0.5, label=f'Min Target ({enp_min})')
            ax2.axhline(y=enp_max, color='g', linestyle='--', alpha=0.5, label=f'Max Target ({enp_max})')
            ax2.legend()
            plt.tight_layout()
            
            # Add ENP plot to the output
            return "Run successful!", metrics_df, fig, weights_df, fig2
        
        return "Run successful!", metrics_df, fig, weights_df

    run_button.click(
        gradio_run_optimization,
        inputs=[
            start_date_input, end_date_input, tickers_input,
            window_size_input, learning_rate_input,
            alpha_sortino_input, alpha_max_drawdown_input, alpha_turnover_input, alpha_concentration_input,
            enp_min_input, enp_max_input
        ],
        outputs=[
            run_status_text,
            metrics_output_df,
            plot_output, 
            weights_output_df,
            gr.Plot(label="Effective Number of Positions")  # Added output for ENP plot
        ]
    )

# --- Mount Gradio app to FastAPI ---
app = gr.mount_gradio_app(app, demo, path="/gradio")

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 