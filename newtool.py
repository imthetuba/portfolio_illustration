import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from InfrontConnect import infront
import pandas as pd
import numpy as np
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
# Index has no OGC ex. post value, so it is set to 0. It is PER YEAR so divided by period later
PERIOD = 252
ASSETS_INDICES_MAP = {
    "NYS:BP": {"index": "NYS:BP", "type": "share", "OGC ex. post": 0.001 },
    "NYS:BRK.B": {"index": "NYS:BP", "type": "share", "OGC ex. post": 0.002},
    "NYS:CAT": {"index": "NYS:BP", "type": "share", "OGC ex. post": 0.003},
    "NYS:DIS": {"index": "NYS:BP", "type": "share", "OGC ex. post": 0.004},
    "NYS:TLT": {"index": "NYS:BP", "type": "bond", "OGC ex. post": 0.005},
    "NYS:GLD": {"index": "GSCI", "type": "alternative", "OGC ex. post": 0.01}
}
ASSETS = list(ASSETS_INDICES_MAP.keys())


# Fetch data function using Infront API, concatenates asset and index data in pandas DataFrame
def fetch_data_infront(tickers, index_tickers, start_date, end_date):
    try:
        # Use Infront API to fetch historical data for assets
        history = infront.GetHistory(
            tickers=tickers,
            fields=["last"],  # Fetching only 'last' price for simplicity
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
        st.write("History Data:", history)  # Debug: Check values
        data_frames = []
        i = 0
        for ticker, df in history.items():
            df['Type'] = 'Asset'
            df['Name'] = tickers[i] #stupid solution to get name as the gethistory function does not return the name the same way
            i = i + 1
            data_frames.append(df)
        
        # Use Infront API to fetch historical data for indices
        index_history = infront.GetHistory(
            tickers=index_tickers,
            fields=["last"],  # Fetching only 'last' price for simplicity
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
        index_data_frames = []
        i=0
        for ticker, df in index_history.items():
            df['Type'] = 'Index'
            df['Name'] = index_tickers[i]
            i = i + 1
            index_data_frames.append(df)
        
        # Combine asset and index data
        asset_data = pd.concat(data_frames)
        index_data = pd.concat(index_data_frames)
        combined_data = pd.concat([asset_data, index_data])
        return combined_data
    except Exception as e:
        st.error(f"Error fetching data: {e}")


def indexed_net_to_100(combined_data):
    # Calculate indexed net return to 100 for each asset, returns in percentages

     # Calculate indexed net return to 100 for each asset
    combined_data['Period Net Return'] = combined_data.groupby('Name')['last'].transform(lambda x: (x / x.iloc[0]) - 1)  # calculate percentage change from start
    st.write("Period Net Return Data:", combined_data[['Name', 'last', 'Period Net Return']])  # Debug: Check values
    
    combined_data['Period Net Return'] = combined_data['Period Net Return'].fillna(0)  # Fill missing values with 0
    combined_data['Indexed Net Return'] = combined_data.groupby('Name')['Period Net Return'].transform(lambda x: (1 + x) * 100) 
    return combined_data

def period_change(combined_data):
    # Calculate period change fron Indexed Net Return for each asset, returns in percentages
    combined_data['Period Change'] = combined_data.groupby('Name')['Indexed Net Return'].transform(lambda x: x.pct_change() * 100)  # calculate percentage change from previous period
    combined_data['Period Change'] = combined_data['Period Change'].fillna(0)  # Fill missing values with 0
    return combined_data


def OCG_adjusted_Period_Change(combined_data):
    combined_data['OCG Adjusted Period Change'] = combined_data.apply(
        lambda row: row['Period Change'] - ASSETS_INDICES_MAP[row['Name']]["OGC ex. post"]/PERIOD, axis=1)
    combined_data['OCG Adjusted Period Change'] = combined_data['OCG Adjusted Period Change'].fillna(0)  # Fill missing values with 0
    return combined_data

def main():
    # Streamlit app
    st.title("Portfolio Illustration Tool")    
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
        # calculate inedexed net return to 100 for each asset    
        combined_data = indexed_net_to_100(combined_data)
        # Debug: Check if data is written correctly
        st.write("Indexed Data:", combined_data)

        combined_data = period_change(combined_data)
        # Debug: Check if data is written correctly
        st.write("Period Change Data:", combined_data)

        combined_data = OCG_adjusted_Period_Change(combined_data)
        st.write("OCG Adjusted Period Change Data:", combined_data)




if __name__=="__main__":
    main()