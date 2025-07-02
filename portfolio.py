import pandas as pd
import streamlit as st
from InfrontConnect import infront

# Load assets and indices map
def load_assets_indices_map(csv_file):
    df = pd.read_csv(csv_file)
    assets_indices_map = {}
    for _, row in df.iterrows():
        static_value = row.get('static', False)
        if pd.isna(static_value):
            static_value = False

        assets_indices_map[row['asset']] = {
            "index": row['index'],
            "display name": row['display_name'],
            "index name": row['index_name'],
            "type": row['type'],
            "OGC ex. post": row['OGC_ex_post'],
            "category": row['category'],
            "static": static_value, 
            "currency": row['currency']
        }
    return assets_indices_map



ASSETS_INDICES_MAP = load_assets_indices_map('assets_indices_map.csv')
# Load standard OGCs
STANDARD_OGC_MAP = {row['asset']: row['OGC_ex_post'] for _, row in pd.read_csv('assets_indices_map.csv').iterrows()}

# Load company OGCs (only for some assets)
company_ogc_df = pd.read_csv('company_ogc.csv')
COMPANY_OGC_MAP = dict(zip(company_ogc_df['asset_id'], company_ogc_df['company_ogc']))

def get_ogc(asset_id, ogc_option="Standard OGC"):
    """
    Returns the company OGC if available and selected, otherwise the standard OGC.
    """
    if ogc_option == "No OGC (OGC = 0)":
        return 0.0
    if ogc_option == "Evisens OGC (if available)" and asset_id in COMPANY_OGC_MAP and pd.notnull(COMPANY_OGC_MAP[asset_id]):
        return COMPANY_OGC_MAP[asset_id]
    return STANDARD_OGC_MAP.get(asset_id, 0.0)

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

def get_categorized_indices(assets_map):
    categories = {"Equity Index": [], "Alternative Index": [], "Interest Bearing Index": []}
    display_name_to_asset_id = {}
    for asset, attributes in assets_map.items():
        if attributes["type"] == "Index":
            category = attributes.get("category", "Uncategorized")
            if category in categories:
                categories[category].append(attributes["display name"])
                display_name_to_asset_id[attributes["display name"]] = asset
    for category in categories:
        categories[category].sort()
    return categories, display_name_to_asset_id


def fetch_static_data(static_indices, start_date, end_date):
    """
    Fetch data for static indices from CSV file.
    Returns a dictionary similar to Infront API response format.
    """
    try:
        # Load the static data
        static_df = pd.read_csv('static_indices.csv')
        static_df['date'] = pd.to_datetime(static_df['date'])
        
                # Convert start_date and end_date to pandas datetime for comparison
        start_date_pd = pd.to_datetime(start_date)
        end_date_pd = pd.to_datetime(end_date)
        # Filter by date range
        static_df = static_df[
            (static_df['date'] >= start_date_pd) & 
            (static_df['date'] <= end_date_pd)
        ]
        
        # Create dictionary similar to Infront response
        static_history = {}
        
        for index_name in static_indices:
            # Filter data for this specific index
            # Remove "STATIC:" prefix for lookup in CSV
            lookup_name = index_name.replace("STATIC:", "")
            # Filter data for this specific index
            index_data = static_df[static_df['index_name'] == lookup_name].copy()
            
            if not index_data.empty:
                # Set date as index and keep only 'last' column
                index_data = index_data.set_index('date')[['last']]
                static_history[f"STATIC:{lookup_name}"] = index_data
                print(f"Loaded static data for {lookup_name}")
            else:
                print(f"Available indices in CSV: {static_df['index_name'].unique()}")
                st.warning(f"No static data found for {lookup_name}")
        
        return static_history
        
    except FileNotFoundError:
        st.error("static_indices.csv file not found")
        return {}
    except Exception as e:
        st.error(f"Error loading static data: {e}")
        return {}


def fetch_data_infront(tickers, index_tickers, start_date, end_date,FX_tickers=["WFX:USDSEK","WFX:EURSEK"]):
    
    with st.spinner("Fetching data for portfolios..."):
        try:
            

            if 'cached_data' not in st.session_state:
                st.session_state['cached_data'] = {}
            # Cache combined data to avoid redundant fetching
            cache_key = f"{start_date}_{end_date}_{'_'.join(tickers)}_{'_'.join(index_tickers)}"

            if cache_key in st.session_state['cached_data']:
                print(f"Using cached data for {cache_key}")
                combined_data = st.session_state['cached_data'][cache_key]

            else:
                # Separate static and dynamic indices
                static_indices = []
                dynamic_indices = []
                
                for index_ticker in index_tickers:
                    static_flag = ASSETS_INDICES_MAP.get(index_ticker, {}).get('static', False)
                    print(f"Index: {index_ticker}, Static flag: {static_flag}")
                    if static_flag:
                        static_indices.append(index_ticker)
                    else:
                        dynamic_indices.append(index_ticker)


                static_assets = []
                dynamic_assets = []
                for ticker in tickers:
                    if ASSETS_INDICES_MAP.get(ticker, {}).get('static', False):
                        static_assets.append(ticker)
                    else:
                        dynamic_assets.append(ticker)

                print(f"Static assets: {static_assets}")
                print(f"Dynamic assets: {dynamic_assets}")

                # Only fetch dynamic assets from Infront
                if dynamic_assets:
                    history = infront.GetHistory(
                        tickers=dynamic_assets,  # Use dynamic_assets instead of tickers
                        fields=["last"],
                        start_date=start_date.strftime('%Y-%m-%d'),
                        end_date=end_date.strftime('%Y-%m-%d')
                    )
                    data_frames = []
                    i = 0
                    for ticker, df in history.items():
                        print(f"Fetching data for {ticker}")
                        df['Type'] = 'Asset'
                        df['Name'] = dynamic_assets[i]
                        i += 1
                        data_frames.append(df)
                else:
                    data_frames = []

                # Fetch static assets if any
                if static_assets:
                    static_asset_history = fetch_static_data(static_assets, start_date, end_date)
                    for ticker, df in static_asset_history.items():
                        df['Type'] = 'Asset'
                        df['Name'] = ticker
                        data_frames.append(df)



                 # Fetch dynamic index data
                index_history = {}
                if dynamic_indices:
                    index_history = infront.GetHistory(
                        tickers=dynamic_indices,
                        fields=["last"],
                        start_date=start_date.strftime('%Y-%m-%d'),
                        end_date=end_date.strftime('%Y-%m-%d')
                    )
                
                # Fetch static index data
                static_history = fetch_static_data(static_indices, start_date, end_date)
                
                # Combine dynamic and static index history
                combined_index_history = {**index_history, **static_history}
                
                index_data_frames = []
                i = 0
                for ticker, df in combined_index_history.items():
                    print(f"Processing index data for {ticker}")
                    df['Type'] = 'Index'
                    df['Name'] = ticker
                    index_data_frames.append(df)
                    i += 1

                asset_data = pd.concat(data_frames)
                index_data = pd.concat(index_data_frames)

                if FX_tickers:
                    FX_history = infront.GetHistory(
                        tickers=FX_tickers,
                        fields=["last"],
                        start_date=start_date.strftime('%Y-%m-%d'),
                        end_date=end_date.strftime('%Y-%m-%d')
                    )
                    fx_data_frames = []
                    i = 0
                    for fx_ticker, fx_df in FX_history.items():
                        print(f"Fetching data for {fx_ticker}")
                        fx_df['Name'] = FX_tickers[i][-6:-3]
                        fx_df['currency'] = FX_tickers[i][-6:-3]
                        i += 1
                        fx_data_frames.append(fx_df)
                    fx_data = pd.concat(fx_data_frames)
                    print(fx_data)
                else:
                    fx_data = pd.DataFrame(columns=['date', 'Name', 'last', 'Type'])
                

                combined_data = pd.concat([asset_data, index_data])

                # 1. Add a 'currency' column to your df using ASSETS_INDICES_MAP
                combined_data['currency'] = combined_data['Name'].map(
                    lambda x: ASSETS_INDICES_MAP.get(x, {}).get('currency', 'SEK')
                )

                # Fix the SettingWithCopyWarning by being more explicit
                if 'date' not in combined_data.columns:
                    combined_data = combined_data.reset_index()
                if 'date' not in fx_data.columns:
                    fx_data = fx_data.reset_index()

                # Make sure we're working with copies to avoid warnings
                combined_data = combined_data.copy()
                fx_data = fx_data.copy()

                def create_fx_dict_with_forward_fill(fx_data, currency_code):
                    """
                    Create FX dictionary with forward fill for missing dates.
                    Uses the last available FX rate instead of defaulting to 1.
                    """
                    currency_data = fx_data[fx_data['currency'] == currency_code].copy()
                    if currency_data.empty:
                        return {}
                    
                    # Sort by date and forward fill missing values
                    currency_data = currency_data.sort_values('date').set_index('date')
                    currency_data = currency_data.reindex(
                        pd.date_range(currency_data.index.min(), currency_data.index.max(), freq='D')
                    ).ffill()  # Use .ffill() instead of fillna(method='ffill')
                    
                    return currency_data['last'].to_dict()

                # Create dictionaries for fast FX lookup with forward fill
                usd_fx = create_fx_dict_with_forward_fill(fx_data, 'USD')
                eur_fx = create_fx_dict_with_forward_fill(fx_data, 'EUR')

                # Get the last available FX rates as fallback
                last_usd_fx = fx_data[fx_data['currency'] == 'USD']['last'].iloc[-1] if not fx_data[fx_data['currency'] == 'USD'].empty else 1
                last_eur_fx = fx_data[fx_data['currency'] == 'EUR']['last'].iloc[-1] if not fx_data[fx_data['currency'] == 'EUR'].empty else 1


                # Function to apply FX conversion
                def apply_fx(row):
                    if row['currency'] == 'USD':
                        fx = usd_fx.get(row['date'], last_usd_fx)  # Use last available instead of 1
                        return row['last'] * fx
                    elif row['currency'] == 'EUR':
                        fx = eur_fx.get(row['date'], last_eur_fx)  # Use last available instead of 1
                        return row['last'] * fx
                    else:
                        return row['last']
                # Function to apply FX conversion

                combined_data['last'] = combined_data.apply(apply_fx, axis=1)

                st.session_state['cached_data'][cache_key] = combined_data




            return combined_data
        except Exception as e:
            raise RuntimeError(f"Error fetching data: {e}")

def clean_data(combined_data,data_frequency, is_multiple_portfolio=False):
    if 'date' not in combined_data.columns:
        combined_data = combined_data.reset_index()

    # Convert to datetime if not already
    combined_data['date'] = pd.to_datetime(combined_data['date'])


    # Find the earliest date for each asset/index
    earliest_dates = combined_data.groupby('Name')['date'].min()
    limiting_asset = earliest_dates.idxmax()
    limiting_date = earliest_dates.max()
    # Map asset id to display name if possible
    display_name = ASSETS_INDICES_MAP.get(limiting_asset, {}).get("display name", limiting_asset)
    st.write(f"Limiting asset/index: **{display_name}** (earliest available date: {limiting_date.date()})")
    # Show all earliest dates for reference, with display names
    earliest_dates_df = earliest_dates.reset_index().rename(columns={'date': 'Earliest Date', 'Name': 'Asset ID'})
    earliest_dates_df['Display Name'] = earliest_dates_df['Asset ID'].map(lambda aid: ASSETS_INDICES_MAP.get(aid, {}).get("display name", aid))



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

    if most_common_diff.days > 20:
        st.info("Detected monthly data.")
        period = 12
        # Keep only the last day of each month for each asset/index
        combined_data['year'] = combined_data['date'].dt.year
        combined_data['month'] = combined_data['date'].dt.month
        combined_data = combined_data.sort_values('date').groupby(['Name', 'year', 'month']).tail(1)
        combined_data = combined_data.drop(columns=['year', 'month'])
        data_frequency = "monthly"
    elif data_frequency == "monthly":
        st.info("Chosen monthly data.")
        st.session_state['data_frequency'] = "monthly"
        period = 12

        # Keep only the last day of each month for each asset/index
        combined_data['year'] = combined_data['date'].dt.year
        combined_data['month'] = combined_data['date'].dt.month
        combined_data = combined_data.sort_values('date').groupby(['Name', 'year', 'month']).tail(1)
        combined_data = combined_data.drop(columns=['year', 'month'])
    elif most_common_diff.days > 5:
        st.info("Detected weekly data.")
        period = 52

    elif most_common_diff.days > 300 or data_frequency == "Yearly":
        st.info("Detected yearly data.")
        period = 1
    else:
        st.info("Detected daily data.")
        period = 252

    return combined_data, period, data_frequency

def indexed_net_to_100(combined_data):
    combined_data['Period Net Return'] = combined_data.groupby('Name')['last'].transform(lambda x: (x / x.iloc[0]) - 1)
    combined_data['Period Net Return'] = combined_data['Period Net Return'].fillna(0)
    combined_data['Indexed Net Return'] = combined_data.groupby('Name')['Period Net Return'].transform(lambda x: (1 + x))
    return combined_data

def period_change(combined_data):
    combined_data['Period Change'] = combined_data.groupby('Name')['Indexed Net Return'].transform(lambda x: x.pct_change())
    combined_data['Period Change'] = combined_data['Period Change'].fillna(0)
    return combined_data


def OGC_adjusted_Period_Change(combined_data, period, ogc_option="Standard OGC"):
    def calculate_adjusted_period_change(row):
        if row['Type'] == 'Index':
            return row['Period Change']
        else:
            ogc = get_ogc(row['Name'], ogc_option)
            return row['Period Change'] - ogc / period
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

    # --- Only fix the first breach ---
    first_breach = breach_rows.sort_values('date').iloc[0]
    breach_date = first_breach['date']
    breach_type = first_breach['Type']

    # Rebalance ALL assets of this type at this date
    total_holdings = date_holdings_df.loc[
        (date_holdings_df['Type'] == breach_type) &
        (date_holdings_df['Date'] == breach_date),
        'Total Holdings'
    ].values[0]
    all_names = combined_data[(combined_data['date'] == breach_date) & (combined_data['Type'] == breach_type)]['Name'].unique()
    for name in all_names:
        idx = combined_data[(combined_data['Name'] == name) & (combined_data['date'] == breach_date)].index
        combined_data.loc[idx, 'Holdings'] = weights[name] * total_holdings

    # Recalculate holdings forward from the breach date for all assets of this type
    def recalc_holdings(group):
        group = group.copy()
        if group['Type'].iloc[0] == breach_type and breach_date in group['date'].values:
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
    # Debug: Print available names and weights keys
    available_names = combined_data['Name'].unique()
    weight_keys = list(weights.keys())
    
    print(f"Available names in data: {available_names}")
    print(f"Weight keys: {weight_keys}")
    
    # Find missing keys
    missing_keys = set(available_names) - set(weight_keys)
    if missing_keys:
        # Filter out rows with missing weights, since these have been set to 0
        combined_data = combined_data[combined_data['Name'].isin(weight_keys)].copy()
    
    # Now safely apply weights
    combined_data['Weight'] = combined_data['Name'].map(weights)
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
        total_breaches_fixed = 0
        while combined_data['Breach'].any():
            breaches_this_round = combined_data['Breach'].sum()
            total_breaches_fixed += breaches_this_round
            iteration += 1
            combined_data, date_holdings_df = reallocate_holdings_at_breach(combined_data, weights, date_holdings_df)
            combined_data = find_breach(combined_data, allocation_limit, weights)
            if iteration >= max_iterations:
                st.warning("Reached maximum iterations while fixing breaches. There may be an issue with the breach resolution logic.")
                break
        st.info(f"Total breaches fixed: {total_breaches_fixed}")
    
    # Calculate period return for total holdings in date_holdings_df
    date_holdings_df = date_holdings_df.sort_values(['Type', 'Date'])
    date_holdings_df['Period Return'] = date_holdings_df.groupby('Type')['Total Holdings'].pct_change().fillna(0)

    # Sort the date_holdings_df first by Type and then by Date
    date_holdings_df = date_holdings_df.sort_values(by=['Type', 'Date'])
    
    return combined_data, date_holdings_df
