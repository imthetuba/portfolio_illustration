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
    "NYS:BP": "NYS:BP",
    "NYS:BRK.B": "NYS:BP",
    "NYS:CAT": "NYS:BP",
    "NYS:DIS": "NYS:BP"
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
            df['Asset'] = ticker
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
            df['Index'] = ticker
            index_data_frames.append(df)
        
        # Combine asset and index data
        asset_data = pd.concat(data_frames)
        index_data = pd.concat(index_data_frames)
        return asset_data, index_data
    except Exception as e:
        st.error(f"Error fetching data: {e}")

# Streamlit app
st.title("Portfolio Illustration Tool")

# User input for selecting assets
ASSETS = list(ASSETS_INDICES_MAP.keys())
selected_assets = st.multiselect("Select assets for the portfolio:", ASSETS)

# Determine associated indices
selected_indices = list({ASSETS_INDICES_MAP[asset] for asset in selected_assets})

start_date = st.date_input("Start date", datetime(2020, 1, 1))
end_date = st.date_input("End date", datetime.today())

# Fetch data
if st.button("Fetch Data"):
    asset_data, index_data = fetch_data_infront(selected_assets, selected_indices, start_date, end_date)
    
    # Plot data
    if not asset_data.empty and not index_data.empty:
        fig = px.line(asset_data, x=asset_data.index, y="last", color="Asset", title="Asset and Index Prices")
        for index in selected_indices:
            index_df = index_data[index_data['Index'] == index]
            fig.add_scatter(x=index_df.index, y=index_df['last'], mode='lines', name=index)
        st.plotly_chart(fig)
    else:
        st.error("No data available for the selected tickers and date range.")