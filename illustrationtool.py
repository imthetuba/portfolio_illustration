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
# python streamlit run illustrationtool.py



# Prompt for username and password
username = st.text_input("Enter your Infront username:")
password = st.text_input("Enter your Infront password:", type="password")


if username and password:
    infront.InfrontConnect(user=username, password=password)  # Use the provided credentials
else:
    st.warning("Please enter your Infront username and password to continue.")
 
# Read the assets and indices map from the CSV file
def load_assets_indices_map(csv_file):
    df = pd.read_csv(csv_file)
    assets_indices_map = {}
    for _, row in df.iterrows():
        assets_indices_map[row['asset']] = {
            "index": row['index'],
            "display name": row['display_name'],
            "index name": row['index_name'],
            "type": row['type'],
            "OGC ex. post": row['OGC_ex_post'],
            "category": row['category']
        }
    return assets_indices_map


# Map structure for assets and their associated indices
# Index has no OGC ex. post value, so it is set to 0. It is PER YEAR so divided by period later
PERIOD = 252
ASSETS_INDICES_MAP = load_assets_indices_map('assets_indices_map.csv')
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


def OGC_adjusted_Period_Change(combined_data):
    # Calculate OGC adjusted period change for each asset
    def calculate_adjusted_period_change(row):
        if row['Type'] == 'Index':
            return row['Period Change'] # Index has no OGC ex. post value
        else:
            return row['Period Change'] - ASSETS_INDICES_MAP[row['Name']]["OGC ex. post"] / PERIOD
    
    combined_data['OGC Adjusted Period Change'] = combined_data.apply(calculate_adjusted_period_change, axis=1)
    combined_data['OGC Adjusted Period Change'] = combined_data['OGC Adjusted Period Change'].fillna(0)  # Fill missing values with 0
    return combined_data

def indexed_OGC_adjusted_to_100(combined_data):
    # Calculate indexed OGC adjusted net return to 100 for each asset
    combined_data['Indexed OGC Adjusted'] = combined_data.groupby('Name')['OGC Adjusted Period Change'].transform(lambda x: (1 + x).cumprod())
    return combined_data

def clean_data(combined_data):
    # Identify common dates across all groups
    # Ensure the date is a column
    if 'date' not in combined_data.columns:
        combined_data = combined_data.reset_index()

    # Find common dates across all groups
    common_dates = combined_data.groupby('Name')['date'].apply(set).agg(lambda x: set.intersection(*x))

    # Convert the result to a DataFrame
    common_dates = pd.DataFrame(list(common_dates), columns=['date'])

    # Filter the combined_data to include only common dates
    combined_data = combined_data[combined_data['date'].isin(common_dates['date'])]
    
    return combined_data

def find_breach(combined_data, allocation_limit, weights):
    # Find breaches in allocation limit
    combined_data['Breach'] = combined_data.apply(
    lambda row: (row['Weight'] > weights[row['Name']] + allocation_limit / 100) or 
                (row['Weight'] < weights[row['Name']] - allocation_limit / 100), axis=1)
    
    return combined_data

def reallocate_holdings_at_breach(combined_data, weights, date_holdings_df):

    # Find the first breach date
    first_breach_date = combined_data.loc[combined_data['Breach'], 'date'].min()
    
    if pd.isna(first_breach_date):
        return combined_data  # No breach found, return the original data
    
    # Reallocate holdings at the first breach date
    for index, row in combined_data.iterrows():
        if row['date'] == first_breach_date:
            name = row['Name']
            total_holdings = date_holdings_df.loc[
                (date_holdings_df['Type'] == row['Type']) & 
                (date_holdings_df['Date'] == row['date']), 
                'Total Holdings'
            ].values[0]

            combined_data.at[index, 'Holdings'] = weights[name] * total_holdings
            # from the first breach date, reallocate the holdings using cumulative product
            # Calculate the adjusted holdings using cumulative product
            def calculate_adjusted_holdings(group):
                # If the group is the sme type (ie asset or index) as the row where the breach was found, then set the holdings to the weight * total holdings
                if group['Type'].iloc[0] == row['Type']:
                    group = group.copy()
                    breach_index = group[group['date'] == first_breach_date].index[0]
                    nameofgroup = group['Name'].iloc[0]
                    total_holdings_local = date_holdings_df.loc[
                        (date_holdings_df['Type'] == group['Type'].iloc[0]) & 
                        (date_holdings_df['Date'] == first_breach_date), 
                        'Total Holdings'
                    ].values[0]
                    
                    group.at[breach_index, 'Holdings'] = weights[nameofgroup] * total_holdings_local
                    # calculate the adjusted holdings using cumulative product from the breach date
                    group.loc[breach_index:, 'Holdings'] = group.loc[breach_index, 'Holdings'] * (1 + group.loc[breach_index:, 'OGC Adjusted Period Change']).cumprod()
                    return group
                else:
                    #if the group is not the same type as the row where the breach was found, then just return the group
                    return group
            
            # Calculate the adjusted holdings for each row
            combined_data = combined_data.groupby('Name').apply(calculate_adjusted_holdings).reset_index(level=0, drop=True)
            
            # Calculate the total holdings for each date and asset or index
            date_holdings_map = combined_data.groupby(['date', 'Type'])['Holdings'].sum().unstack().to_dict()

            # Calculate the adjusted weights
            for index, row in combined_data.iterrows():
                date = row['date']
                type_ = row['Type']
                if type_ in date_holdings_map and date in date_holdings_map[type_]:
                    total_holdings = date_holdings_map[type_][date]
                    if total_holdings != 0:
                        combined_data.at[index, 'Weight'] = row['Holdings'] / total_holdings

            date_holdings_df = pd.DataFrame.from_dict(date_holdings_map, orient='index').reset_index()
            date_holdings_df = date_holdings_df.melt(id_vars=['index'], var_name='Type', value_name='Total Holdings')
            date_holdings_df.columns = ['Type', 'Date', 'Total Holdings']

            break
        else:
            continue


    return combined_data, date_holdings_df


def create_portfolio(combined_data, weights, start_investment, allocation_limit):

    combined_data['Weight'] = combined_data.apply(lambda row: weights[row['Name']], axis=1)
    combined_data['Holdings'] = combined_data['Weight'] * start_investment
    
    # Calculate the adjusted holdings using cumulative product
    def calculate_adjusted_holdings(group):
        group = group.copy()
        group['Holdings'] = group['Holdings'].iloc[0] * (1 + group['OGC Adjusted Period Change']).cumprod()
        return group
    
    # Calculate the adjusted holdings for each row
    combined_data = combined_data.groupby('Name').apply(calculate_adjusted_holdings).reset_index(level=0, drop=True)
    
    # Calculate the total holdings for each date and asset or index
    date_holdings_map = combined_data.groupby(['date', 'Type'])['Holdings'].sum().unstack().to_dict()

    
    # Calculate the adjusted weights
    for index, row in combined_data.iterrows():
        date = row['date']
        type_ = row['Type']
        if type_ in date_holdings_map and date in date_holdings_map[type_]:
            total_holdings = date_holdings_map[type_][date]
            if total_holdings != 0:
                combined_data.at[index, 'Weight'] = row['Holdings'] / total_holdings


    date_holdings_df = pd.DataFrame.from_dict(date_holdings_map, orient='index').reset_index()
    date_holdings_df = date_holdings_df.melt(id_vars=['index'], var_name='Type', value_name='Total Holdings')
    date_holdings_df.columns = ['Type', 'Date', 'Total Holdings']
    
    combined_data = find_breach(combined_data, allocation_limit, weights)
    combined_data['Initial Breaches'] = combined_data['Breach']

    with st.spinner("Fixing breaches in portfolio allocation..."):
        while combined_data['Breach'].any():
            combined_data, date_holdings_df = reallocate_holdings_at_breach(combined_data, weights, date_holdings_df)
            combined_data = find_breach(combined_data, allocation_limit, weights)

    return combined_data, date_holdings_df

def get_categorized_assets(assets_map):
    # Create a dictionary to map categories to their respective assets
    categories = {"Equity": [], "Alternative": [], "Interest Bearing": []}
    display_name_to_asset_id = {}
    for asset, attributes in assets_map.items():
        if attributes["type"] != "Index":  # Exclude items with type "Index"
            category = attributes.get("category", "Uncategorized")
            display_name = attributes.get("display name", asset)
            if category in categories:
                categories[category].append(display_name)
                display_name_to_asset_id[display_name] = asset

    # Sort the assets within each category alphabetically
    for category in categories:
        categories[category].sort()

    return categories, display_name_to_asset_id

def plot_holdings(combined_data):
    # Separate the data into assets and indices
    assets_data = combined_data[combined_data['Type'] == 'Asset']
    indices_data = combined_data[combined_data['Type'] == 'Index']

    # Create a mapping from asset IDs to display names
    asset_id_to_display_name = {asset: attributes["display name"] for asset, attributes in ASSETS_INDICES_MAP.items()}

    # Create a line plot for assets
    fig = px.line(assets_data, x='date', y='Holdings', color='Name', title='Holdings in Assets vs Indices')

    # Update the names in the legend to display names
    for trace in fig.data:
        trace.name = asset_id_to_display_name.get(trace.name, trace.name)

    # Add a line plot for indices
    for index_name in indices_data['Name'].unique():
        index_data = indices_data[indices_data['Name'] == index_name]
        display_name = asset_id_to_display_name.get(index_name, index_name)
        fig.add_scatter(x=index_data['date'], y=index_data['Holdings'], mode='lines', name=display_name)

    # Update layout for better visualization
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Holdings',
        legend_title='Assets and Indices'
    )

    return fig

def plot_date_vs_total_holdings(date_holdings_df):
    # Create a mapping from asset IDs to display names
    asset_id_to_display_name = {asset: attributes["display name"] for asset, attributes in ASSETS_INDICES_MAP.items()}

    # Create a line plot for total holdings
    fig = px.line(date_holdings_df, x='Date', y='Total Holdings', color='Type', title='Date vs Total Holdings',
                  color_discrete_map={
                      'Index': 'red',
                      'Asset': 'blue'
                  })

    # Update the names in the legend to display names
    for trace in fig.data:
        trace.name = asset_id_to_display_name.get(trace.name, trace.name)


    # Update layout for better visualization
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Total Holdings',
        legend_title='Type'
    )

    return fig

def calculate_drawdowns(data, column, window=500):
    """
    Calculate drawdowns for a given column in the DataFrame using a rolling maximum.
    """
    data = data.copy()
    data['Max'] = data[column].cummax()
    data['Drawdown'] = (data[column] - data['Max']) / data['Max']
    return data


def plot_drawdowns(portfolio_data, index_data, window=500):
    """
    Plot drawdowns for the portfolio and the index.
    """
    # Calculate drawdowns
    portfolio_data = calculate_drawdowns(portfolio_data, 'Total Holdings', window)
    index_data = calculate_drawdowns(index_data, 'Total Holdings', window)

    
    # Create a DataFrame for plotting
    drawdown_data = pd.DataFrame({
        'Date': portfolio_data['Date'],
        'Portfolio Drawdown': portfolio_data['Drawdown']
    })

    index_drawdown_data = pd.DataFrame({
        'Date': index_data['Date'],
        'Index Drawdown': index_data['Drawdown']
    })

    # Merge the two DataFrames on the 'Date' column
    drawdown_data = pd.merge(drawdown_data, index_drawdown_data, on='Date', how='outer')

    # Melt the DataFrame for Plotly
    drawdown_data = drawdown_data.melt(id_vars=['Date'], var_name='Type', value_name='Drawdown')

    # Filter out non-date values
    drawdown_data = drawdown_data.dropna(subset=['Date'])

    # Create the plot
    fig = px.line(drawdown_data, x='Date', y='Drawdown', color='Type', title='Drawdowns: Portfolio vs Index',
                  color_discrete_map={
                      'Portfolio Drawdown': 'blue',
                      'Index Drawdown': 'red'
                  })

    # Update layout for better visualization
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Drawdown',
        legend_title='Type',
        template='plotly_white'
    )

    return fig



def main():
    # Streamlit app
    st.title("OGC adjusted Portfolio Illustration Tool")    
    combined_data = pd.DataFrame()

   # Get categorized assets for the dropdown menu
    categories, display_name_to_asset_id = get_categorized_assets(ASSETS_INDICES_MAP)

    # Create three columns for asset selection
    col1, col2, col3 = st.columns(3)

    with col1:
        selected_shares = st.multiselect("Select Equity:", categories["Equity"])

    with col2:
        selected_alternative = st.multiselect("Select Alternative:", categories["Alternative"])

    with col3:
        selected_interest_bearing = st.multiselect("Select Interest Bearing:", categories["Interest Bearing"])

    # Combine selected assets
    selected_display_names = selected_shares + selected_alternative + selected_interest_bearing

    # Map selected display names to asset IDs
    selected_assets = [display_name_to_asset_id[name] for name in selected_display_names]

    # Determine associated indices
    selected_indices = list({ASSETS_INDICES_MAP[asset]["index"] for asset in selected_assets})


    start_date = st.date_input("Start date", datetime(2020, 1, 1))
    end_date = st.date_input("End date", datetime.today())
        
    # Fetch data
    if st.button("Fetch Data") and selected_assets:
        #st.write(selected_assets)
        #st.write(selected_indices)
        combined_data = fetch_data_infront(selected_assets, selected_indices, start_date, end_date)
        combined_data = clean_data(combined_data)
        combined_data = indexed_net_to_100(combined_data)
        combined_data = period_change(combined_data)
        combined_data = OGC_adjusted_Period_Change(combined_data)
        combined_data = indexed_OGC_adjusted_to_100(combined_data)

        # Store combined_data in session state
        st.session_state['combined_data'] = combined_data

    if 'combined_data' in st.session_state:
        combined_data = st.session_state['combined_data']
        # st.write("Indexed OGC Adjusted Data:", combined_data)
    else:
        combined_data = None

    # User input for selecting weights
    weights = {}
    for asset in selected_assets:
        display_name = ASSETS_INDICES_MAP[asset].get("display name", asset)
        weight = st.number_input(f"Weight for {display_name}", min_value=0.0, max_value=1.0, value=0.1)
        weights[asset] = weight
    # Sum the weights for the indices if there are duplicates
    for asset in selected_assets:
        index = ASSETS_INDICES_MAP[asset]['index']
        if index in weights:
            weights[index] += weights[asset]
        else:
            weights[index] = weights[asset]



    # Check if weights add up to 1
    if sum(weights.values()) != 2.0: # double because of the index
        st.error("The weights must add up to 1. Please adjust the weights.")

    # User input for start investment amount and allocation limit
    start_investment = st.number_input("Start Investment Amount", min_value=0.0, value=100000.0)
    allocation_limit = st.number_input("Allocation Limit (%)", min_value=0.0, max_value=100.0, value=7.0)

    # Button to create portfolio outputs
    if st.button("Create Portfolio Outputs") and combined_data is not None:
        # Implement the logic to create portfolio outputs based on the inputs
        # Implement the logic to create portfolio outputs based on the inputs
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Start Investment Amount", value=f"{start_investment} SEK")
        with col2:
            st.metric(label="Allocation Limit", value=f"{allocation_limit}%")


        combined_data, date_holdings_df  = create_portfolio(combined_data, weights, start_investment, allocation_limit)
        
        # Plot the holdings
        fig = plot_holdings(combined_data)
        st.plotly_chart(fig)
        
        # Plot the date vs total holdings
        fig_total_holdings = plot_date_vs_total_holdings(date_holdings_df)
        st.plotly_chart(fig_total_holdings)
        
        
        # Plot the drawdowns
        index_data = date_holdings_df[date_holdings_df['Type'] == 'Index']
        portfolio_data = date_holdings_df[date_holdings_df['Type'] == 'Asset']
        fig_drawdowns = plot_drawdowns(portfolio_data, index_data)
        st.plotly_chart(fig_drawdowns)


        st.write("Portfolio Data:", combined_data)
        st.write("Date vs Total Holdings:", date_holdings_df)

        




if __name__=="__main__":
    main()