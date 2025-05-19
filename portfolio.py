import pandas as pd
import streamlit as st
from InfrontConnect import infront
PERIOD = 252

# Load assets and indices map
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

ASSETS_INDICES_MAP = load_assets_indices_map('assets_indices_map.csv')

def get_categorized_assets(assets_map):
    categories = {"Equity": [], "Alternative": [], "Interest Bearing": []}
    display_name_to_asset_id = {}
    for asset, attributes in assets_map.items():
        if attributes["type"] != "Index":
            category = attributes.get("category", "Uncategorized")
            if category in categories:
                categories[category].append(attributes["display name"])
                display_name_to_asset_id[attributes["display name"]] = asset
    for category in categories:
        categories[category].sort()
    return categories, display_name_to_asset_id

def fetch_data_infront(tickers, index_tickers, start_date, end_date):
    try:
        history = infront.GetHistory(
            tickers=tickers,
            fields=["last"],
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
        data_frames = []
        i = 0
        for ticker, df in history.items():
            df['Type'] = 'Asset'
            df['Name'] = tickers[i]
            i += 1
            data_frames.append(df)
        index_history = infront.GetHistory(
            tickers=index_tickers,
            fields=["last"],
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
        index_data_frames = []
        i = 0
        for ticker, df in index_history.items():
            df['Type'] = 'Index'
            df['Name'] = index_tickers[i]
            i += 1
            index_data_frames.append(df)
        asset_data = pd.concat(data_frames)
        index_data = pd.concat(index_data_frames)
        combined_data = pd.concat([asset_data, index_data])
        return combined_data
    except Exception as e:
        raise RuntimeError(f"Error fetching data: {e}")

def clean_data(combined_data):
    if 'date' not in combined_data.columns:
        combined_data = combined_data.reset_index()
    common_dates = combined_data.groupby('Name')['date'].apply(set).agg(lambda x: set.intersection(*x))
    common_dates = pd.DataFrame(list(common_dates), columns=['date'])
    combined_data = combined_data[combined_data['date'].isin(common_dates['date'])]
    return combined_data

def indexed_net_to_100(combined_data):
    combined_data['Period Net Return'] = combined_data.groupby('Name')['last'].transform(lambda x: (x / x.iloc[0]) - 1)
    combined_data['Period Net Return'] = combined_data['Period Net Return'].fillna(0)
    combined_data['Indexed Net Return'] = combined_data.groupby('Name')['Period Net Return'].transform(lambda x: (1 + x))
    return combined_data

def period_change(combined_data):
    combined_data['Period Change'] = combined_data.groupby('Name')['Indexed Net Return'].transform(lambda x: x.pct_change())
    combined_data['Period Change'] = combined_data['Period Change'].fillna(0)
    return combined_data



def OGC_adjusted_Period_Change(combined_data):
    def calculate_adjusted_period_change(row):
        if row['Type'] == 'Index':
            return row['Period Change']
        else:
            return row['Period Change'] - ASSETS_INDICES_MAP[row['Name']]["OGC ex. post"] / PERIOD
    combined_data['OGC Adjusted Period Change'] = combined_data.apply(calculate_adjusted_period_change, axis=1)
    combined_data['OGC Adjusted Period Change'] = combined_data['OGC Adjusted Period Change'].fillna(0)
    return combined_data

def indexed_OGC_adjusted_to_100(combined_data):
    combined_data['Indexed OGC Adjusted'] = combined_data.groupby('Name')['OGC Adjusted Period Change'].transform(lambda x: (1 + x).cumprod())
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
        max_iterations = 100
        iteration = 0
        while combined_data['Breach'].any():
            iteration += 1
            combined_data, date_holdings_df = reallocate_holdings_at_breach(combined_data, weights, date_holdings_df)
            combined_data = find_breach(combined_data, allocation_limit, weights)
            if iteration >= max_iterations:
                st.warning("Reached maximum iterations while fixing breaches. There may be an issue with the breach resolution logic.")
                break
    return combined_data, date_holdings_df

