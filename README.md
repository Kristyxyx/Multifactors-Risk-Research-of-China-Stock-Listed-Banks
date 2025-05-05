# Multifactors Risk Research of China Stock Listed Banks

This project conducts a multifactor risk analysis of China's stock-listed banks. It involves cleaning and processing various financial factors, decomposing risk premiums into systematic (common) and idiosyncratic components, as well as specific risk types (default, liquidity, market). Causal inference techniques (Double Machine Learning - DML) are employed to estimate the causal effects of different risk factors on risk premiums.

## Project Structure

```
.
├── data/                     # Directory for input data (implied, needs specific .pkl files)
│   └── exported_data/        # Processed data files used by scripts
│       ├── factors_data_train.pkl
│       ├── fundamentals_data_train.pkl
│       ├── ... (other factor/data pkl files for train/test)
│       └── factors_risks_dicts.csv # Factor to risk mapping
├── image/                    # Output directory for generated plots and images
│   ├── risk_factors_clusters.png
│   ├── causal_effect/
│   │   ├── ate_by_treatment_*.png
│   │   ├── cate_over_time_*.png
│   │   └── ... (various causal effect plots)
│   └── ... (other visualization outputs)
├── script/                   # Contains the core logic and analysis scripts
│   ├── factors_risks_dicts_generator.py # Generates the factor-risk mapping
│   ├── main.py               # Main script for running the analysis pipeline
│   ├── main.ipynb            # Jupyter notebook version of the main script
│   ├── DataGetter.py         # (Likely) Data fetching/preprocessing scripts
│   ├── DataGetter_v2.py      # (Likely) Data fetching/preprocessing scripts
│   └── ... (other helper scripts)
├── presentation/             # Presentation materials (e.g., milestones)
├── reference/                # Reference documents or papers
├── README.md                 # This file
└── requirements.txt          # Project dependencies
```

## Data Source

The project requires several pre-processed data files in Pickle (`.pkl`) format, expected to be located in the `./data/exported_data/` directory. These files include:

*   Factor data (e.g., `factors_data`, `fundamentals_data`, `macros_data`, etc.)
*   Bank stock information (`bank_stocks_info`)
*   Returns data (`returns_data`)
*   FCF discounted model parameters (`FCF_discounted_model_params_data`)

These files are typically generated from raw financial data sources like Wind Financial Terminal or CSMAR Database through preliminary data processing steps (potentially using scripts like `DataGetter.py` or `export_data.ipynb`). The `factors_risks_dicts.csv` file maps financial factors to risk categories (Default, Liquidity, Market).

## Environment

See `requirements.txt` for a list of necessary Python packages.

## Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/Multifactors-Risk-Research-of-China-Stock-Listed-Banks.git
cd Multifactors-Risk-Research-of-China-Stock-Listed-Banks
```
*(Replace `yourusername` with the actual GitHub username or organization)*

### 2. Set up the environment and activate

Choose either `venv` or `conda`:

*   **venv**:
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

*   **conda**:
    ```bash
    conda create --name risk_env python=3.9 # Or desired Python version
    conda activate risk_env
    ```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

The primary analysis is performed by the `script/main.py` script (or its notebook equivalent `script/main.ipynb`).

1.  **Prepare Data**: Ensure all required `.pkl` data files are present in the `./data/exported_data/` directory. The script can run in 'train' or 'test' mode, requiring corresponding `_train.pkl` or `_test.pkl` files. If `factors_risks_dicts.csv` is missing, `script/factors_risks_dicts_generator.py` might need to be run or adapted.
2.  **Execute**: Run the main script from the root directory:
    ```bash
    python ./script/main.py
    ```
    Alternatively, run the cells within `script/main.ipynb`.

The script performs the following steps:
*   Loads and merges various factor datasets.
*   Cleans data, handles missing values using IterativeImputer (with XGBoost).
*   Calculates industry-level factors weighted by enterprise value.
*   Standardizes factors and clusters them into risk categories (Default, Liquidity, Market) using KShape.
*   Decomposes the industry risk premium (based on WACC) into common and idiosyncratic components using a custom AdaBoost-based regressor (`CustomRegressor`).
*   Estimates the contribution of each risk category (Default, Liquidity, Market) to the total risk premium.
*   Uses CausalForestDML to estimate the causal effect of each risk component premium on the common and idiosyncratic risk premiums, controlling for time trends, seasonality, and other risk components as confounders.
*   Generates visualizations for factor clusters and causal effects.

## Results

The execution of the script produces several outputs:

*   **Processed Data**: Intermediate dataframes like `industry_factors_df` might be implicitly available within the `Modeler` class instance.
*   **Risk Premium Estimates**:
    *   Total Industry Risk Premium (`industry_risk_premium`)
    *   Common Risk Premium (`industry_risks_premiums_totals['common_risk']`)
    *   Idiosyncratic Risk Premium (`industry_risks_premiums_totals['idiosyncratic_risk']`)
    *   Risk Component Premiums (`industry_risks_premiums_components['default_risk']`, etc.)
*   **Causal Models**: Fitted `CausalForestDML` models stored in `modeler.causal_forest_forests`.
*   **Causal Effect Estimates**: Average Treatment Effects (ATE) and Conditional Average Treatment Effects (CATE) for the impact of each risk component on common/idiosyncratic risk.
*   **Visualizations**: Plots saved in the `image/` directory, including:
    *   `risk_factors_clusters.png`: Visualization of clustered risk factors over time.
    *   Various plots in `image/causal_effect/` showing CATE distributions, CATE over time, feature importances, etc., for each treatment-outcome pair (e.g., `ate_distribution_default_risk_vs_common_risk.png`).

## License

MIT License

Copyright (c) 2025 Multifactors Risk Research of China Stock Listed Banks Project
