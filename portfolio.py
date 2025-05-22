import pandas as pd
import streamlit as st
from InfrontConnect import infront

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
            print(f"Fetching data for {ticker}")
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
            print(f"Fetching data for {ticker}")
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

def clean_data(combined_data, is_multiple_portfolio=False):
    if 'date' not in combined_data.columns:
        combined_data = combined_data.reset_index()
    common_dates = combined_data.groupby('Name')['date'].apply(set).agg(lambda x: set.intersection(*x))
    common_dates = pd.DataFrame(list(common_dates), columns=['date'])
    combined_data = combined_data[combined_data['date'].isin(common_dates['date'])]
    
    # Convert to datetime if not already
    combined_data['date'] = pd.to_datetime(combined_data['date'])
    # Sort by date
    combined_data = combined_data.sort_values('date')
    # Calculate the most common date difference per asset/index
    date_diffs = (
        combined_data
        .sort_values(['Name', 'date'])
        .groupby('Name')['date']
        .diff()
        .dropna()
    )
    most_common_diff = date_diffs.mode()[0]
    st.write(f"Most common date difference: {most_common_diff}")

    if most_common_diff.days > 15 or is_multiple_portfolio:
        st.info("Detected monthly data.")
        period = 12
        # Keep only end-of-month values (or after 25th)
        combined_data['day'] = combined_data['date'].dt.day
        combined_data['month'] = combined_data['date'].dt.month
        combined_data['year'] = combined_data['date'].dt.year
        # Keep only rows where day >= 25
        combined_data = combined_data[combined_data['day'] >= 25]
        # For each asset/index and month, keep the last available row
        combined_data = combined_data.sort_values('date').groupby(['Name', 'year', 'month']).tail(1)
        # Drop helper columns
        combined_data = combined_data.drop(columns=['day', 'month', 'year'])

    elif most_common_diff.days > 5:
        st.info("Detected weekly data.")
        period = 52
    else:
        st.info("Detected daily data.")
        period = 252

    return combined_data, period

def indexed_net_to_100(combined_data):
    combined_data['Period Net Return'] = combined_data.groupby('Name')['last'].transform(lambda x: (x / x.iloc[0]) - 1)
    combined_data['Period Net Return'] = combined_data['Period Net Return'].fillna(0)
    combined_data['Indexed Net Return'] = combined_data.groupby('Name')['Period Net Return'].transform(lambda x: (1 + x))
    return combined_data

def period_change(combined_data):
    combined_data['Period Change'] = combined_data.groupby('Name')['Indexed Net Return'].transform(lambda x: x.pct_change())
    combined_data['Period Change'] = combined_data['Period Change'].fillna(0)
    return combined_data



def OGC_adjusted_Period_Change(combined_data, period):
    def calculate_adjusted_period_change(row):
        if row['Type'] == 'Index':
            return row['Period Change']
        else:
            return row['Period Change'] - ASSETS_INDICES_MAP[row['Name']]["OGC ex. post"] / period
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
    breach_rows = combined_data[combined_data['Breach']]
    if breach_rows.empty:
        return combined_data, date_holdings_df

    # Fix all unique breach dates in this pass
    for breach_date in sorted(breach_rows['date'].unique()):
        for breach_type in breach_rows[breach_rows['date'] == breach_date]['Type'].unique():
            mask = (combined_data['date'] == breach_date) & (combined_data['Type'] == breach_type)
            total_holdings = date_holdings_df.loc[
                (date_holdings_df['Type'] == breach_type) & 
                (date_holdings_df['Date'] == breach_date), 
                'Total Holdings'
            ].values[0]
            affected_names = combined_data[mask]['Name'].unique()
            for name in affected_names:
                idx = combined_data[(combined_data['Name'] == name) & (combined_data['date'] == breach_date)].index
                combined_data.loc[idx, 'Holdings'] = weights[name] * total_holdings

    # Recalculate holdings forward from each breach date for affected names
    def recalc_holdings(group):
        group = group.copy()
        breach_dates = set(breach_rows[breach_rows['Name'] == group['Name'].iloc[0]]['date'])
        for breach_date in breach_dates:
            if breach_date in group['date'].values:
                breach_index = group[group['date'] == breach_date].index[0]
                group.loc[breach_index:, 'Holdings'] = group.loc[breach_index, 'Holdings'] * (1 + group.loc[breach_index:, 'OGC Adjusted Period Change']).cumprod()
        return group

    combined_data = combined_data.groupby('Name').apply(recalc_holdings).reset_index(level=0, drop=True)

    # Recalculate total holdings and weights
    date_holdings_map = combined_data.groupby(['date', 'Type'])['Holdings'].sum().unstack().to_dict()
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
        max_iterations = 1000
        iteration = 0
        while combined_data['Breach'].any():
            iteration += 1
            combined_data, date_holdings_df = reallocate_holdings_at_breach(combined_data, weights, date_holdings_df)
            combined_data = find_breach(combined_data, allocation_limit, weights)
            if iteration >= max_iterations:
                st.warning("Reached maximum iterations while fixing breaches. There may be an issue with the breach resolution logic.")
                break
            

    
    return combined_data, date_holdings_df

    output = pd.pivot_table(
        combined_data,
        index='date',
        columns='Name',
        values='Holdings'
    )

    # Replace asset IDs with display names in columns
    display_name_map = {asset: attrs["display name"] for asset, attrs in ASSETS_INDICES_MAP.items()}
    output.rename(columns=display_name_map, inplace=True)

    # Add total holdings per date as a new column
    output['Total Holdings'] = output.sum(axis=1)

    # Write to Excel
    excel_buffer = pd.ExcelWriter('portfolio_output.xlsx', engine='xlsxwriter')
    output.to_excel(excel_buffer, sheet_name='Portfolio Holdings')
    date_holdings_df.to_excel(excel_buffer, sheet_name='Date Holdings', index=False)
    excel_buffer.close()