# Multifactors Risk Research of China Stock Listed Banks

## Project Description
This project conducts a multifactor risk analysis of China's stock-listed banks. The research framework includes:
1. Accounting premium decomposition methodology
2. Exploration of financial factors' causal effects on various risk components
3. Mapping risk components to probability of loss
4. Determining loss rate distribution characteristics

## Data Sources
- Financial data of China's listed banks (from Wind Financial Terminal and CSMAR Database)
- China's banking regulatory indicator data
- Macroeconomic data

## Required Packages
```
pandas
numpy
scipy
statsmodels
matplotlib
seaborn
scikit-learn
PyMC3 (for Bayesian inference)
networkx (for causal network analysis)
```

## Running Instructions
1. Clone the repository
    ```
    git clone https://github.com/username/Multifactors-Risk-Research-of-China-Stock-Listed-Banks.git
    cd Multifactors-Risk-Research-of-China-Stock-Listed-Banks
    ```

2. Install dependencies
    ```
    pip install -r requirements.txt
    ```

3. Data preparation
    - Place data files in the `data/` directory

4. Run analysis
    ```
    python src/main.py
    ```

5. View results
    - Analysis results will be saved in the `results/` directory
    - Visualization charts will be saved in the `figures/` directory
