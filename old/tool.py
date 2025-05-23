import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from InfrontConnect import infront
import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import datetime

# A script that uses Streamlit to create a portfolio illustration using Infront API. (https://software.infrontservices.com/helpfiles/infront/v86/en/index.html?api_python.html)
# The user can select a list of assets, set their weights, and view the portfolio performance over a specified date range.
# To use the tool, you need to have an Infront account and API access. 
# Write the following in your terminal to install the required packages and run the script:

# python pip install streamlit infrontconnect pandas plotly
# python streamlit run tool.py

# Connect to Infront API
infront.InfrontConnect(user="David.Lundberg.ipt", password="Infront2022!")  # Replace with your credentials

# Map structure for assets and their associated indices
ASSETS_INDICES_MAP = {
    "NYS:BP": {"index": "NYS:BP", "type": "share"},
    "NYS:BRK.B": {"index": "NYS:BP", "type": "share"},
    "NYS:CAT": {"index": "NYS:BP", "type": "share"},
    "NYS:DIS": {"index": "NYS:BP", "type": "share"},
    "NYS:TLT": {"index": "NYS:BP", "type": "bond"},
    "NYS:GLD": {"index": "GSCI", "type": "alternative"}
}
# Fetch data function using Infront API
def fetch_data_infront(tickers, index_tickers, start_date, end_date):
    try:
        # Use Infront API to fetch historical data for assets
        history = infront.GetHistory(
            tickers=tickers,
            fields=["last"],  # Fetching only 'last' price for simplicity
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
        data_frames = []
        for ticker, df in history.items():
            df['Type'] = 'Asset'
            df['Name'] = ticker
            data_frames.append(df)
        
        # Use Infront API to fetch historical data for indices
        index_history = infront.GetHistory(
            tickers=index_tickers,
            fields=["last"],  # Fetching only 'last' price for simplicity
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
        index_data_frames = []
        for ticker, df in index_history.items():
            df['Type'] = 'Index'
            df['Name'] = ticker
            index_data_frames.append(df)
        
        # Combine asset and index data
        asset_data = pd.concat(data_frames)
        index_data = pd.concat(index_data_frames)
        combined_data = pd.concat([asset_data, index_data])
        return combined_data
    except Exception as e:
        st.error(f"Error fetching data: {e}")

# Calculate weighted portfolio performance
def calculate_weighted_portfolio(data, weights):
    st.write("Data before pivot:", data)
    data = data.pivot_table(index=data.index, columns='Name', values='last')
    st.write("Pivot table:", data)
    st.write("Weights:", weights)
    
     # Ensure weights are aligned with the pivot table columns
    aligned_weights = {}
    for col in data.columns:
        
        st.write("col :", col)
        base_col = col.split(':')[0]  # Strip additional information
        if base_col in weights:
            aligned_weights[col] = weights[base_col]
    st.write("Aligned Weights:", aligned_weights)
    
    # Check if any columns are missing weights
    missing_weights = [col for col in data.columns if col not in weights]
    if missing_weights:
        st.error(f"Missing weights for columns: {missing_weights}")
    
    # Multiply the data by the weights
    multiplied_data = data.mul(aligned_weights, axis=1)
    st.write("Multiplied data:", multiplied_data)
    
    # Sum the weighted data
    weighted_data = multiplied_data.sum(axis=1)
    st.write("Weighted data:", weighted_data)
    
    return weighted_data




# Streamlit app
st.title("Portfolio Illustration Tool")

# User input for selecting assets
ASSETS = list(ASSETS_INDICES_MAP.keys())
selected_assets = st.multiselect("Select assets for the portfolio:", ASSETS)

# Determine associated indices
selected_indices = list({ASSETS_INDICES_MAP[asset]["index"] for asset in selected_assets})

# Display selected asset types
if selected_assets:
    asset_types = {asset: ASSETS_INDICES_MAP[asset]["type"] for asset in selected_assets}
    st.write("Selected Asset Types:", asset_types)

# User input for selecting weights
weights = {}
for asset in selected_assets:
    weight = st.number_input(f"Weight for {asset}", min_value=0.0, max_value=1.0, value=0.1)
    weights[asset] = weight

# Check if weights add up to 1
if sum(weights.values()) != 1.0:
    st.error("The weights must add up to 1. Please adjust the weights.")

start_date = st.date_input("Start date", datetime(2020, 1, 1))
end_date = st.date_input("End date", datetime.today())

# Fetch data
if st.button("Fetch Data") and sum(weights.values()) == 1.0:
    combined_data = fetch_data_infront(selected_assets, selected_indices, start_date, end_date)
    
    # Debug: Check if data is fetched correctly
    st.write("Combined Data:", combined_data)
    
    # Calculate weighted portfolio performance
    weighted_portfolio = calculate_weighted_portfolio(combined_data[combined_data['Type'] == 'Asset'], weights)
    weighted_index = calculate_weighted_portfolio(combined_data[combined_data['Type'] == 'Index'], weights)
    # Plot data
    if not combined_data.empty:
        fig = px.line(combined_data, x=combined_data.index, y="last", color="Name", line_dash="Type", title="Asset and Index Prices")
        fig.add_scatter(x=weighted_portfolio.index, y=weighted_portfolio, mode='lines', name='Weighted Portfolio')
        fig.add_scatter(x=weighted_index.index, y=weighted_index, mode='lines', name='Weighted Index')
        st.plotly_chart(fig)
    else:
        st.error("No data available for the selected tickers and date range.")