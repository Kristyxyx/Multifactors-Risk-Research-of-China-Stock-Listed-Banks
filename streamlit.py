import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import io
from PIL import Image
import graphviz
from sklearn.tree import export_graphviz
import matplotlib.image as mpimg
import random
from matplotlib.figure import Figure

# Import your Modeler class
# Uncomment this if you're running the app in the same directory as main.py
# from main import Modeler

# For demonstration, I'll include a function to load the model
@st.cache_resource
def load_model(mode='train'):
    """Load the Modeler instance or create a new one"""
    try:
        return pickle.load(open('model/modeler_instance.pkl', 'rb'))
    except:
        # If you're running this in a different location, you'll need to adjust paths
        mol = Modeler(mode=mode)
        
        # Load necessary data
        try:
            mol.industry_factors_datas = pickle.load(open('data/modeled_data/bank_factors_datas_train.pkl', 'rb'))
        except:
            mol.industry_factors_datas = mol.clean_and_average_factors_datas()
            
        try:
            mol.industry_r_waac_data = pickle.load(open('data/modeled_data/bank_r_waac_data_train.pkl', 'rb'))
        except:
            mol.industry_r_waac_data = mol.clean_and_average_r_waac_data()
            
        try:
            mol.industry_factors_df = pd.read_csv('data/modeled_data/bank_factors_df_train.csv', index_col='date', parse_dates=['date'])
        except:
            mol.industry_factors_df = mol.transform_factors_datas_from_dict_to_df()
            
        try:
            mol.factors_risks_data_standardized = pd.read_csv('data/dict/factors_risks_data_standardized.csv', index_col='factor_code')
        except:
            factors_codes_missing, factors_codes_excessive, factors_codes_duplicated, mol.factors_risks_data_standardized = mol.standardize_factors_risks_data()
            
        try:
            mol.industry_risks_premiums_totals = pickle.load(open('data/modeled_data/bank_risks_premiums_totals.pkl', 'rb'))
            mol.risks_premiums_totals_curs = pickle.load(open('model/risks_premiums_totals_curs.pkl', 'rb'))
        except:
            mol.industry_risks_premiums_totals, mol.risks_premiums_totals_curs = mol.get_industry_risks_premiums_totals()
            
        try:
            mol.industry_risks_premiums_components = pickle.load(open('data/modeled_data/bank_risks_premiums_components.pkl', 'rb'))
            mol.risks_premiums_components_curs = pickle.load(open('model/risks_premiums_components_curs.pkl', 'rb'))
        except:
            mol.industry_risks_premiums_components, mol.risks_premiums_components_curs = mol.get_industry_risks_premiums_components()
            
        try:
            mol.causal_forest_forests = pickle.load(open('model/bank_causal_forest_forests.pkl', 'rb'))
        except:
            mol.causal_forest_forests = mol.estimate_causal_effects_with_dml()
            
        return mol

# Define functions for the visualizations
def risk_factors_clusters_viz(model, windows=None, risk_types=None, num_factors=None, figsize=None):
    """Generate the risk factors clusters visualization"""
    if windows is None:
        windows = (1, 30, 120)
    if risk_types is None:
        risk_types = ['default_risk', 'liquidity_risk', 'market_risk']
    if num_factors is None:
        num_factors = 3
    if figsize is None:
        figsize = (5 * len(risk_types), 5 * len(windows))
        
    fig = model.visualize_risk_factors_clusters(
        windows=windows,
        risk_types=risk_types,
        num_factors=num_factors,
        figsize=figsize
    )
    return fig

def causal_forest_trees_viz(model):
    """Generate the causal forest trees visualization"""
    fig = model.visualize_causal_forest_trees()
    return fig

def causal_effect_viz(model, treatment, outcome):
    """Generate visualizations for causal effects analysis"""
    figs = model.visualize_causal_effect(treatment, outcome)
    return figs

# Streamlit App
st.set_page_config(page_title="Financial Risk Modeling Dashboard", 
                   page_icon="📊", 
                   layout="wide")

st.title("Financial Risk Modeling & Causal Analysis Dashboard")

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose a visualization", 
    ["Risk Factors Clusters", 
     "Causal Forest Trees", 
     "Causal Effect Analysis"])

# Load the model (cached)
try:
    model = load_model()
    st.sidebar.success("Model loaded successfully!")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    st.stop()

# Main content area
if app_mode == "Risk Factors Clusters":
    st.header("Risk Factors Clusters Visualization")
    
    # Settings for risk factors clusters
    st.subheader("Customize Visualization")
    col1, col2 = st.columns(2)
    
    with col1:
        window_size_1 = st.slider("Short-term Window Size", 1, 30, 1)
        window_size_2 = st.slider("Medium-term Window Size", 31, 90, 30)
        window_size_3 = st.slider("Long-term Window Size", 91, 365, 120)
        windows = (window_size_1, window_size_2, window_size_3)
    
    with col2:
        risk_types = st.multiselect(
            "Select Risk Types",
            options=['default_risk', 'liquidity_risk', 'market_risk'],
            default=['default_risk', 'liquidity_risk', 'market_risk']
        )
        num_factors = st.slider("Number of Factors per Risk Type", 1, 10, 3)
    
    # Generate and display visualization
    if st.button("Generate Risk Factors Clusters Visualization"):
        with st.spinner("Generating visualization..."):
            try:
                fig = risk_factors_clusters_viz(model, windows, risk_types, num_factors)
                st.pyplot(fig)
                
                # Option to download
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=300)
                buf.seek(0)
                st.download_button(
                    label="Download Visualization",
                    data=buf,
                    file_name="risk_factors_clusters.png",
                    mime="image/png"
                )
            except Exception as e:
                st.error(f"Error generating visualization: {e}")

elif app_mode == "Causal Forest Trees":
    st.header("Causal Forest Trees Visualization")
    
    # Generate and display visualization
    try:
        with st.spinner("Generating Causal Forest Trees visualization (this may take a while)..."):
            fig = causal_forest_trees_viz(model)
            st.pyplot(fig)
            
            # Option to download
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=300)
            buf.seek(0)
            st.download_button(
                label="Download Visualization",
                data=buf,
                file_name="causal_forest_trees.png",
                mime="image/png"
            )
    except Exception as e:
        st.error(f"Error generating visualization: {e}")

elif app_mode == "Causal Effect Analysis":
    st.header("Causal Effect Analysis")
    
    # Settings for causal effect analysis
    col1, col2 = st.columns(2)
    
    with col1:
        treatment = st.selectbox(
            "Select Treatment Variable",
            options=['default_risk', 'liquidity_risk', 'market_risk']
        )
    
    with col2:
        outcome = st.selectbox(
            "Select Outcome Variable",
            options=['common_risk', 'idiosyncratic_risk']
        )
    
    # Generate and display visualizations
    if st.button("Generate Causal Effect Visualizations"):
        with st.spinner("Generating causal effect visualizations..."):
            try:
                figs = causal_effect_viz(model, treatment, outcome)
                
                # Create tabs for different visualizations
                viz_names = [
                    "CATE Over Time", 
                    "CATE By Month", 
                    "ATE By Treatment Intensity",
                    "ATE Distribution",
                    "Treatment vs Outcome",
                    "Feature Importance"
                ]
                
                tabs = st.tabs(viz_names[:len(figs)])
                
                for i, (tab, fig) in enumerate(zip(tabs, figs)):
                    with tab:
                        st.pyplot(fig)
                        
                        # Option to download
                        buf = io.BytesIO()
                        fig.savefig(buf, format="png", dpi=300)
                        buf.seek(0)
                        st.download_button(
                            label=f"Download {viz_names[i]}",
                            data=buf,
                            file_name=f"causal_effect_{viz_names[i].lower().replace(' ', '_')}_{treatment}_vs_{outcome}.png",
                            mime="image/png"
                        )
            except Exception as e:
                st.error(f"Error generating visualizations: {e}")

# Add footer
st.sidebar.markdown("---")
st.sidebar.info(
    "This dashboard visualizes financial risk modeling and causal analysis "
    "for bank stocks. Use the navigation menu to explore different "
    "visualizations."
)