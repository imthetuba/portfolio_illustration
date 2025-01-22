import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from InfrontConnect import infront
import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import datetime

# A script that uses Streamlit to create a simple portfolio illustration tool using Infront API.
# The user can select a list of assets, set their weights, and view the portfolio performance over a specified date range.
# To use the tool, you need to have an Infront account and API access. 
# Write the following in your terminal to install the required packages and run the script:

# pip install streamlit infrontconnect pandas plotly
# python streamlit run tool.py

# Connect to Infront API
infront.InfrontConnect(user="David.Lundberg.ipt", password="Infront2022!")  # Replace with your credentials

# Fetch data function using Infront API
def fetch_data_infront(tickers, start_date, end_date):
    try:
        # Use Infront API to fetch historical data
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
        return pd.concat(data_frames)
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# Streamlit app
st.title("Portfolio Illustration Tool (Infront)")

# User input for selecting assets
asset_list = ["NYS:BP", "NYS:BRK.B", "NYS:CAT", "NYS:DIS"]  # Example tickers
selected_assets = st.multiselect("Select assets for the portfolio:", asset_list)

# User input for setting portfolio weights
weights = {}
if selected_assets:
    st.subheader("Set portfolio weights")
    for asset in selected_assets:
        weight = st.slider(f"Weight for {asset} (%):", min_value=0, max_value=100, value=0, step=1)
        weights[asset] = weight

# User input for global area allocation
regions = ["North America", "Europe", "Asia", "Other"]
region_weights = {}
st.subheader("Allocate portfolio weights by region")
for region in regions:
    region_weight = st.slider(f"Weight for {region} (%):", min_value=0, max_value=100, value=0, step=1)
    region_weights[region] = region_weight

# User input for date range
st.subheader("Select date range for analysis")
start_date = st.date_input("Start date", datetime(2020, 1, 1))
end_date = st.date_input("End date", datetime.today())

# Fetch and display data
if st.button("Fetch Data"):
    if sum(weights.values()) != 100:
        st.error("Total weights must sum to 100%.")
    elif sum(region_weights.values()) != 100:
        st.error("Total region weights must sum to 100%.")
    else:
        data = fetch_data_infront(selected_assets, start_date, end_date)
        if not data.empty:
            st.write("Portfolio Data:", data.head())

            # Create portfolio performance plot
            data['Weighted Price'] = data['last'] * data['Asset'].map(weights) / 100
            portfolio_performance = data.groupby(data.index)['Weighted Price'].sum()

            fig = px.line(portfolio_performance, x=portfolio_performance.index, y=portfolio_performance.values,
                          labels={"x": "Date", "y": "Portfolio Value"}, title="Portfolio Performance")
            st.plotly_chart(fig)
        else:
            st.error("No data fetched.")
