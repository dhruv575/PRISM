---
title: Portfolio Optimization with OGD
emoji: 📈
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 8000
---

# Portfolio Optimizer

An interactive portfolio optimization tool that uses Online Gradient Descent (OGD) to optimize asset allocation based on multiple objectives.

## Access Options in HuggingFace Spaces

You can access this application in three different ways:

1. **Via Gradio Interface**: The default access method when you open the space.

2. **Via Direct Application**: For a better experience with the full UI, use one of these links:
   - **Main Dashboard**: `/fullpage`
   - **Education Page**: `/fullpage/education`

   You can append these paths to your HuggingFace space URL.

3. **Via API**: The application also exposes several API endpoints for programmatic access.

## Features

- **Multi-objective Optimization**: Balance risk-adjusted returns, maximum drawdown, turnover, and portfolio concentration
- **Interactive UI**: Visualize portfolio performance through interactive charts
- **Stock Selection**: Select individual stocks or entire sectors
- **Educational Resources**: Learn about portfolio optimization concepts

## How It Works

This application implements a robust portfolio optimization strategy using Online Gradient Descent (OGD). The optimization aims to maximize risk-adjusted returns while minimizing drawdowns, turnover, and over-concentration.

- **Data Source**: Historical stock price data for S&P 500 constituents
- **Optimization Method**: Online Gradient Descent
- **Benchmarks**: Equal-weight and random portfolios for comparison

## Technical Details

The application is built with:

- **Backend**: Python with FastAPI
- **Data Processing**: pandas, NumPy
- **Visualization**: Chart.js
- **Frontend**: HTML, CSS, JavaScript

## License

This project is available for educational and research purposes.

## How to Use

1. Select your desired date range and optimization parameters
2. Choose stocks from the sector lists on the right
3. Click "Run Allocation" to run the optimization
4. View results in the interactive charts and metrics panels

## Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

The application will be available at http://localhost:8000
