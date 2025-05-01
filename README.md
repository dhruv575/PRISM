# PRISM: Portfolio Refinement Through Iterative Sequential Modeling

**Team Members:** Dhruv Gupta, Aiden Lee, Frank Ma, Kelly Wang, Didrik Wiig-Andersen

## High-Level Summary

**Problem Statement:** Traditional portfolio strategies often fail under real-world market stress. The challenge is to develop a trading algorithm that is robust to market shocks and adapts to changing conditions, aiming for strong risk-adjusted returns while managing drawdown, turnover, and concentration risk.

**Approach:** We developed PRISM, a dynamic portfolio optimizer using Online Gradient Descent (OGD) implemented in PyTorch. It optimizes a multi-objective function balancing Sortino ratio, maximum drawdown, portfolio turnover, and concentration penalties (measured via Effective Number of Positions). The model processes market data sequentially, updating portfolio weights daily based on rolling window calculations of these metrics.

**Key Findings/Contributions:**
- Developed a dynamic, multi-objective portfolio optimizer capable of adapting daily to market shifts.
- The optimizer, balancing Sortino, Max Drawdown, Turnover, and Concentration, consistently outperformed equal-weighted benchmarks in backtests.
- Identified and quantified the trade-offs between maximizing returns and controlling risk factors like drawdown and concentration, particularly in tech-heavy portfolios.
- Implemented the solution using PyTorch, leveraging automatic differentiation for gradient-based optimization.
- Provided a reusable framework (`src/`) for implementing and testing similar dynamic portfolio optimization strategies.
- Created an interactive demo showcasing portfolio evolution over time ([Demo Link](https://droov-opt.hf.space/)).

## Repository Structure Overview

```
.
├── README.md                    # You are here!
├── final_report.md              # Final project report (detailed).
├── report.md                    # Potentially an earlier or summary report.
├── requirements.txt             # Python package dependencies.
├── notebooks/                   # Jupyter notebooks for experiments, visualization, and demos.
│   └── demo.ipynb               # (Expected) Runnable demo notebook showcasing key functionalities.
├── src/                         # Source code for the PRISM optimization library and scripts.
│   ├── __init__.py
│   ├── optimizers/              # Core optimization logic (e.g., OGD).
│   ├── metrics/                 # Calculation of Sortino, MDD, Turnover, ENP, etc.
│   └── utils/                   # Utility functions (data loading, plotting).
├── data/                        # Placeholder for data files (not included, specify download instructions).
├── slides/                      # Presentation slides.
└── _development_history/        # Archive of previous report drafts, critiques, etc. (Not needed for final run).
```
*Note: The exact structure within `src/` and the presence of `demo.ipynb` are inferred; please adjust if needed.*

- **`src/`**: Contains the core Python modules for the optimization logic and metric calculations.
- **`notebooks/`**: Holds Jupyter notebooks, ideally including a `demo.ipynb`.
- **`final_report.md` / `report.md`**: The detailed written report(s).
- **`requirements.txt`**: Required Python packages.
- **`slides/`**: Project presentation slides.
- **`_development_history/`**: Archival folder showing project evolution, not required for execution.

## Setup Instructions

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/dhruv575/PRISM
    cd PRISM
    ```

2.  **Python Version:**
    Ensure you have Python 3.10 or newer installed (as per the original template, confirm if different).

3.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # Activate the environment:
    # Windows:
    .\\venv\\Scripts\\activate
    # macOS/Linux:
    source venv/bin/activate
    ```

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Data:**
    - The necessary stock data is included in demo_files/data. You can choose to obtain historical stock price data (e.g., from Yahoo Finance, Alpha Vantagem, WRDS) and place it in a location accessible by the scripts/notebooks. Ensure the data format matches what the code expects.

## Running the Code

1.  **Activate Virtual Environment:** (If created)
    ```bash
    # Windows: .\\venv\\Scripts\\activate
    # macOS/Linux: source venv/bin/activate
    ```

2.  **Run the Notebook:**
    Look at the `final_notebook.ipynb` exists in the `notebooks/` folder:
    ```bash
    jupyter notebook notebooks/final_notebook.ipynb
    # Or: jupyter lab notebooks/final_notebook.ipynb
    ```
    Follow the instructions within the notebook.

## Executable Demo Link

Explore an interactive demo of our project hosted on Hugging Face Spaces:

**➡️ [PRISM Interactive Demo](https://droov-opt.hf.space/)**






