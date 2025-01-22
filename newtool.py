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
# Prompt for username and password
#username = st.text_input("Enter your Infront username:")
#password = st.text_input("Enter your Infront password:", type="password")


#if username and password:
#    infront.InfrontConnect(user=username, password=password)  # Use the provided credentials
#else:
#    st.warning("Please enter your Infront username and password to continue.")
# Connect to Infront API
infront.InfrontConnect(user="David.Lundberg.ipt", password="Infront2022!") 


# Map structure for assets and their associated indices
# Index has no OGC ex. post value, so it is set to 0. It is PER YEAR so divided by period later
PERIOD = 252
ASSETS_INDICES_MAP = {
    "NYS:BP": {"index": "NYS:BP", "type": "share", "OGC ex. post": 0.001 },
    "NYS:BRK.B": {"index": "NYS:BP", "type": "share", "OGC ex. post": 0.002},
    "NYS:CAT": {"index": "NYS:BP", "type": "share", "OGC ex. post": 0.01},
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
    # Calculate indexed net return to 100 for each asset

     # Calculate indexed net return to 100 for each asset
    combined_data['Period Net Return'] = combined_data.groupby('Name')['last'].transform(lambda x: (x / x.iloc[0]) - 1)  # calculate percentage change from start
    
    combined_data['Period Net Return'] = combined_data['Period Net Return'].fillna(0)  # Fill missing values with 0
    combined_data['Indexed Net Return'] = combined_data.groupby('Name')['Period Net Return'].transform(lambda x: (1 + x)) 
    return combined_data

def period_change(combined_data):
    # Calculate period change fron Indexed Net Return for each asset
    combined_data['Period Change'] = combined_data.groupby('Name')['Indexed Net Return'].transform(lambda x: x.pct_change())  # calculate percentage change from previous period
    combined_data['Period Change'] = combined_data['Period Change'].fillna(0)  # Fill missing values with 0
    return combined_data


def OCG_adjusted_Period_Change(combined_data):
    # Calculate OCG adjusted period change for each asset
    combined_data['OCG Adjusted Period Change'] = combined_data.apply(
        lambda row: row['Period Change'] - ASSETS_INDICES_MAP[row['Name']]["OGC ex. post"]/PERIOD, axis=1)
    
    combined_data['OCG Adjusted Period Change'] = combined_data['OCG Adjusted Period Change'].fillna(0)  # Fill missing values with 0
    return combined_data

def indexed_OCG_adjusted_to_100(combined_data):
    # Calculate indexed OCG adjusted net return to 100 for each asset
    combined_data['Indexed OCG Adjusted'] = combined_data.groupby('Name')['OCG Adjusted Period Change'].transform(lambda x: (1 + x).cumprod())
    return combined_data

def create_portfolio(combined_data, weights, start_investment, allocation_limit):
    combined_data['Weight'] = combined_data.apply(lambda row: weights[row['Name']], axis=1)
    combined_data['Holdnings'] = combined_data['Weight'] * start_investment
    
    # Initialize the total portfolio amount
    group_length = combined_data.groupby('Name').size().max()
    total_portfolio_amounts = [0] * group_length # over allocing space, but it is fine for now
    total_portfolio_amounts[0] = start_investment
    total_portfolio_amounts_index = [0] * group_length # over allocing space, but it is fine for now
    total_portfolio_amounts_index[0] = start_investment

    # Group by 'Name' and apply the Holdnings and weight calculations
    for name, group in combined_data.groupby('Name'):
        for i in range(1, len(group)):
            previous_Holdnings = group.iloc[i-1]['Holdnings']
            indexed_ocg_adjusted = group.iloc[i]['Indexed OCG Adjusted']
            
            adjusted_Holdnings_amount = previous_Holdnings * indexed_ocg_adjusted
            
            # Update the Holdnings
            combined_data.loc[group.index[i], 'Holdnings'] = adjusted_Holdnings_amount
            
            # Update the total portfolio amount
            if group.iloc[i]['Type'] == 'Asset':
                total_portfolio_amounts[i] = total_portfolio_amounts[i] + adjusted_Holdnings_amount
            else:
                total_portfolio_amounts_index[i] = total_portfolio_amounts_index[i] + adjusted_Holdnings_amount
    st.write("Combined Data:", combined_data)
    for name, group in combined_data.groupby('Name'):
        for i in range(1, len(group)):
            # Update the weight when the individul alloations have been updated for all groups, its not super efficient but all allocations have to be updated before the weigths are
            if group.iloc[i]['Type'] == 'Asset':
                combined_data.loc[group.index[i], 'Weight'] = combined_data.loc[group.index[i], 'Holdnings'] / total_portfolio_amounts[i]
            else:
                combined_data.loc[group.index[i], 'Weight'] = combined_data.loc[group.index[i], 'Holdnings'] / total_portfolio_amounts_index[i]

    return combined_data, total_portfolio_amounts, total_portfolio_amounts_index

def main():
    # Streamlit app
    st.title("Portfolio Illustration Tool")    
    combined_data = pd.DataFrame()

    selected_assets = st.multiselect("Select assets for the portfolio:", ASSETS)

    # Determine associated indices
    selected_indices = list({ASSETS_INDICES_MAP[asset]["index"] for asset in selected_assets})


    start_date = st.date_input("Start date", datetime(2020, 1, 1))
    end_date = st.date_input("End date", datetime.today())
        
    # Fetch data
    if st.button("Fetch Data") and selected_assets:
        combined_data = fetch_data_infront(selected_assets, selected_indices, start_date, end_date)

        combined_data = indexed_net_to_100(combined_data)
        combined_data = period_change(combined_data)
        combined_data = OCG_adjusted_Period_Change(combined_data)
        combined_data = indexed_OCG_adjusted_to_100(combined_data)

        # Store combined_data in session state
        st.session_state['combined_data'] = combined_data

    if 'combined_data' in st.session_state:
        combined_data = st.session_state['combined_data']
        st.write("Indexed OCG Adjusted Data:", combined_data)
    else:
        combined_data = None

    # User input for selecting weights
    weights = {}
    for asset in selected_assets:
        weight = st.number_input(f"Weight for {asset}", min_value=0.0, max_value=1.0, value=0.1)
        weights[asset] = weight
        weights[ASSETS_INDICES_MAP[asset]['index']] = weight #the weight for the index is the same as the asset

    # Check if weights add up to 1
    if sum(weights.values()) != 2.0: # double because of the index
        st.error("The weights must add up to 1. Please adjust the weights.")

    # User input for start investment amount and allocation limit
    start_investment = st.number_input("Start Investment Amount", min_value=0.0, value=100000.0)
    allocation_limit = st.number_input("Allocation Limit (%)", min_value=0.0, max_value=100.0, value=7.0)

    # Button to create portfolio outputs
    if st.button("Create Portfolio Outputs") and combined_data is not None:
        # Implement the logic to create portfolio outputs based on the inputs
        st.write(f"Start Investment Amount: {start_investment}")
        st.write(f"Allocation Limit: {allocation_limit}%")


        combined_data, total_portfolio_amounts, total_portfolio_amounts_index = create_portfolio(combined_data, weights, start_investment, allocation_limit)
        st.write("Portfolio Data:", combined_data)
        st.write("Total Portfolio Amounts:", total_portfolio_amounts)
        st.write("Total Portfolio Amounts Index:", total_portfolio_amounts_index)


if __name__=="__main__":
    main()