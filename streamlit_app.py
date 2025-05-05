import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import graphviz # Required for causal forest tree visualization
import io
from PIL import Image
import os
import sys
import random # Required for visualize_risk_factors_clusters sampling

# --- Dynamic Import of Modeler ---
# Add the script directory to the Python path to import Modeler
# Assumes streamlit_app.py is in the root directory and main.py is in ./script/
try:
    script_dir = os.path.join(os.path.dirname(__file__), 'script')
    if script_dir not in sys.path:
        sys.path.append(script_dir)
    # Import necessary classes from main.py
    from main import Modeler, CustomRegressor, ComplementRegressor
except ImportError as e:
    st.error(f"Fatal Error: Could not import 'Modeler' from 'script/main.py'. "
             f"Ensure 'script/main.py' exists and is in the correct location relative to this script. "
             f"Error details: {e}")
    st.stop() # Stop execution if core import fails
except Exception as e:
    st.error(f"Fatal Error: An unexpected error occurred during import from 'script/main.py'. "
             f"Error details: {e}")
    st.stop()

# --- Configuration ---
# Define base directories relative to this script's location
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
IMAGE_DIR = os.path.join(BASE_DIR, 'image') # Although not directly used for loading, good practice
MODELED_DATA_DIR = os.path.join(DATA_DIR, 'modeled_data')
DICT_DATA_DIR = os.path.join(DATA_DIR, 'dict')

# Define expected file paths
PATHS = {
    "factors_df": os.path.join(MODELED_DATA_DIR, 'bank_factors_df_train.csv'),
    "factors_risks_standardized": os.path.join(DICT_DATA_DIR, 'factors_risks_dicts_standardized.csv'),
    "premiums_totals": os.path.join(MODELED_DATA_DIR, 'bank_risks_premiums_totals.pkl'),
    "premiums_components": os.path.join(MODELED_DATA_DIR, 'bank_risks_premiums_components.pkl'),
    "causal_forests": os.path.join(MODEL_DIR, 'bank_causal_forest_forests.pkl'),
    # Add other paths if needed by Modeler initialization or other functions
}

# --- Caching Functions for Data/Model Loading ---

@st.cache_data
def load_csv(path, index_col=None, parse_dates=None):
    """Loads a CSV file into a pandas DataFrame with error handling."""
    if not os.path.exists(path):
        st.error(f"File not found: {path}. Please run `python script/main.py` first.")
        return None
    try:
        df = pd.read_csv(path, index_col=index_col, parse_dates=parse_dates)
        # Ensure date index is DatetimeIndex if parsed
        if parse_dates and index_col and isinstance(df.index, pd.Index) and not isinstance(df.index, pd.DatetimeIndex):
             try:
                 df.index = pd.to_datetime(df.index)
             except Exception as date_err:
                 st.warning(f"Could not convert index to DatetimeIndex for {path}: {date_err}")
        return df
    except Exception as e:
        st.error(f"Error loading CSV {path}: {e}")
        return None

@st.cache_resource
def load_pickle(path):
    """Loads a pickle file with error handling."""
    if not os.path.exists(path):
        st.error(f"File not found: {path}. Please run `python script/main.py` first.")
        return None
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading pickle {path}: {e}")
        return None

@st.cache_resource
def get_modeler_instance():
    """Initializes the Modeler instance and loads necessary data/models."""
    st.write("Initializing Modeler and loading data...") # Progress message

    # Initialize Modeler (assuming 'train' mode data is used for visualization)
    # Pass dummy data during init if needed, actual data loaded below
    try:
        # Modeler init might require specific arguments or load data itself.
        # Adjust initialization based on the actual Modeler.__init__ signature if needed.
        # Assuming default init is sufficient and we load attributes manually.
        mol = Modeler(mode='train') # Or determine mode differently if needed
    except Exception as e:
        st.error(f"Failed to initialize Modeler: {e}")
        return None

    # Load necessary data attributes into the modeler instance
    # These paths correspond to the files saved in the main.py script
    mol.industry_factors_df = load_csv(PATHS["factors_df"], index_col='date', parse_dates=['date'])
    mol.factors_risks_data_standardized = load_csv(PATHS["factors_risks_standardized"], index_col='factor_code')
    mol.industry_risks_premiums_totals = load_pickle(PATHS["premiums_totals"])
    mol.industry_risks_premiums_components = load_pickle(PATHS["premiums_components"])
    mol.causal_forest_forests = load_pickle(PATHS["causal_forests"])

    # --- Crucial Check and Setup ---
    # Check if essential data loaded successfully
    essential_data = {
        "industry_factors_df": mol.industry_factors_df,
        "factors_risks_data_standardized": mol.factors_risks_data_standardized,
        "industry_risks_premiums_totals": mol.industry_risks_premiums_totals,
        "industry_risks_premiums_components": mol.industry_risks_premiums_components,
        "causal_forest_forests": mol.causal_forest_forests,
    }

    loaded_successfully = True
    for name, data in essential_data.items():
        if data is None:
            st.warning(f"Essential data '{name}' failed to load. Some visualizations may not work.")
            loaded_successfully = False

    # Set index attribute (required by some methods like _construct_causal_data)
    if mol.industry_factors_df is not None:
        # Ensure the index is a DatetimeIndex if it's supposed to be
        if not isinstance(mol.industry_factors_df.index, pd.DatetimeIndex):
             try:
                 mol.industry_factors_df.index = pd.to_datetime(mol.industry_factors_df.index)
                 mol.index = mol.industry_factors_df.index # Set the index attribute
             except Exception as e:
                 st.error(f"Failed to convert industry_factors_df index to DatetimeIndex: {e}")
                 loaded_successfully = False # Mark as failed if index conversion fails
        else:
             mol.index = mol.industry_factors_df.index # Set the index attribute
    else:
         st.warning("Cannot set modeler index as 'industry_factors_df' failed to load.")
         # Depending on requirements, you might want to set loaded_successfully = False here too

    if not loaded_successfully:
        st.error("Modeler initialization failed due to missing essential data. Cannot proceed.")
        return None # Return None if essential data is missing

    st.write("Modeler initialized successfully.")
    return mol

# --- Streamlit App UI ---
st.set_page_config(layout="wide") # Use wide layout for better plot display
st.title("Multifactor Risk Research Visualization")

# Load the Modeler instance (cached)
# This will only run the function once and cache the result
modeler_instance = get_modeler_instance()

# Check if Modeler instance is valid before proceeding
if modeler_instance is None:
    st.error("Failed to load the Modeler and its data. Please check the error messages above and ensure `script/main.py` has been run successfully.")
    st.stop() # Stop the app if Modeler couldn't be loaded

# --- Sidebar Options ---
st.sidebar.title("Visualization Options")
viz_choice = st.sidebar.selectbox(
    "Choose Visualization",
    [
        "Risk Factors Clusters",
        "Causal Forest Trees",
        "Causal Effect Analysis"
    ]
)
st.sidebar.info("Ensure you have run `python script/main.py` first to generate the necessary data and model files.")

# --- Visualization Logic ---

if viz_choice == "Risk Factors Clusters":
    st.header("Risk Factors Clusters Visualization")
    st.write("Displays moving averages of risk factors grouped by clustered risk type.")

    # Check required data for this visualization
    if modeler_instance.industry_factors_df is not None and modeler_instance.factors_risks_data_standardized is not None:
        try:
            # Use a fixed random seed for reproducibility within the app session
            # The Modeler instance itself might have a random_state, use that if appropriate
            # Or define a specific seed for the visualization call
            fig = modeler_instance.visualize_risk_factors_clusters(random_seed=42) # Example seed
            st.pyplot(fig)
            # Optional: Add download button
            # buf = io.BytesIO()
            # fig.savefig(buf, format="png", dpi=300)
            # st.download_button("Download Plot", buf.getvalue(), "risk_factors_clusters.png", "image/png")
        except AttributeError as e:
            st.error(f"Error calling visualize_risk_factors_clusters: {e}. "
                     f"This might indicate an issue with the Modeler class implementation or missing data.")
        except Exception as e:
            st.error(f"An unexpected error occurred during Risk Factors Clusters visualization: {e}")
            st.exception(e) # Show full traceback for debugging
    else:
        st.warning("Required data ('industry_factors_df' or 'factors_risks_data_standardized') is not loaded. Cannot display this visualization.")

elif viz_choice == "Causal Forest Trees":
    st.header("Causal Forest Trees Visualization")
    st.write("Displays the last decision tree from the Causal Forest model for each treatment-outcome pair.")

    # Check required data
    if modeler_instance.causal_forest_forests is not None:
        try:
            with st.spinner("Generating Causal Forest Tree visualization..."):
                fig = modeler_instance.visualize_causal_forest_trees()
            st.pyplot(fig)
            # Optional: Add download button
            # buf = io.BytesIO()
            # fig.savefig(buf, format="png", dpi=300)
            # st.download_button("Download Plot", buf.getvalue(), "causal_forest_trees.png", "image/png")
        except ImportError as e:
             st.error(f"Import error during Causal Forest Trees visualization: {e}. "
                      f"Make sure 'graphviz' Python package is installed (`pip install graphviz`). "
                      f"You might also need to install the Graphviz system library (see graphviz.org/download/).")
        except AttributeError as e:
            st.error(f"Error calling visualize_causal_forest_trees: {e}. "
                     f"Check if the 'causal_forest_forests' attribute exists and is loaded correctly in the Modeler.")
        except Exception as e:
            st.error(f"An unexpected error occurred during Causal Forest Trees visualization: {e}")
            st.exception(e) # Show traceback for debugging
    else:
        st.warning("Required model ('causal_forest_forests') is not loaded. Cannot display this visualization.")


elif viz_choice == "Causal Effect Analysis":
    st.header("Causal Effect Analysis Visualization")
    st.write("Displays various plots analyzing the Causal Average Treatment Effect (CATE).")

    # Check required data
    required_attributes = [
        modeler_instance.causal_forest_forests,
        modeler_instance.industry_risks_premiums_totals,
        modeler_instance.industry_risks_premiums_components,
        hasattr(modeler_instance, 'index') and modeler_instance.index is not None
    ]

    if all(attr is not None for attr in required_attributes):
        # Define potential treatments and outcomes based on your main.py
        treatments = ['default_risk', 'liquidity_risk', 'market_risk']
        outcomes = ['common_risk', 'idiosyncratic_risk']

        # Sidebar selectors for treatment and outcome
        selected_treatment = st.sidebar.selectbox(
            "Select Treatment Variable (Risk Component)",
            options=treatments,
            index=0 # Default selection
        )
        selected_outcome = st.sidebar.selectbox(
            "Select Outcome Variable (Total Risk)",
            options=outcomes,
            index=0 # Default selection
        )

        st.subheader(f"Analysis for Treatment: `{selected_treatment}` -> Outcome: `{selected_outcome}`")

        try:
            with st.spinner(f"Generating Causal Effect plots for {selected_treatment} -> {selected_outcome}..."):
                # This function should return a list of figures
                figures = modeler_instance.visualize_causal_effect(selected_treatment, selected_outcome)

            # Define titles for each figure based on the order in visualize_causal_effect (from main.py)
            fig_titles = [
                "1. CATE Over Time",
                "2. CATE by Month (Seasonality)",
                "3. CATE vs. Treatment Value",
                "4. CATE Distribution",
                "5. Treatment vs. Outcome (by Season)",
                "6. Feature Importance for CATE Model"
            ]

            if not isinstance(figures, list):
                 st.error("The 'visualize_causal_effect' function did not return a list of figures as expected.")
            elif len(figures) != len(fig_titles):
                 st.warning(f"Expected {len(fig_titles)} figures from visualize_causal_effect, but received {len(figures)}. Displaying available figures.")

            # Display each figure
            for i, fig in enumerate(figures):
                title = fig_titles[i] if i < len(fig_titles) else f"Figure {i+1}"
                st.markdown(f"--- \n**{title}**") # Add separator and title
                if fig is not None and isinstance(fig, plt.Figure):
                    st.pyplot(fig)
                    # Optional download button per figure
                    # buf = io.BytesIO()
                    # fig_filename = f"causal_effect_{selected_treatment}_{selected_outcome}_{title.lower().replace(' ', '_').replace('.', '').replace(':', '')}.png"
                    # fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
                    # st.download_button(f"Download Plot {i+1}", buf.getvalue(), fig_filename, "image/png", key=f"download_{i}")
                else:
                    st.warning(f"Plot '{title}' could not be generated or is not a valid figure.")
                # st.divider() # Use markdown separator instead

        except KeyError as e:
             st.error(f"KeyError: {e}. This usually means the selected treatment '{selected_treatment}' or outcome '{selected_outcome}' "
                      f"was not found in the loaded data structures (e.g., premiums components/totals, causal forests). "
                      f"Check the keys in the corresponding .pkl files.")
        except AttributeError as e:
            st.error(f"AttributeError: {e}. Check if the Modeler instance has all required attributes and methods "
                     f"needed for 'visualize_causal_effect'.")
        except Exception as e:
            st.error(f"An unexpected error occurred during Causal Effect Analysis visualization for "
                     f"{selected_treatment} -> {selected_outcome}: {e}")
            st.exception(e) # Show traceback for debugging

    else:
        st.warning("One or more required data/models ('causal_forest_forests', 'premiums_totals', 'premiums_components') "
                   "or the modeler 'index' attribute are not loaded/set. Cannot display this visualization.")

# --- End of App ---