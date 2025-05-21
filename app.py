import streamlit as st
from InfrontConnect import infront
from portfolio import (
    get_categorized_assets,
    fetch_data_infront,
    clean_data,
    indexed_net_to_100,
    period_change,
    OGC_adjusted_Period_Change,
    indexed_OGC_adjusted_to_100,
    create_portfolio,
    ASSETS_INDICES_MAP,
)
from datetime import datetime

from visualisation import generate_summary_report

# Prompt for username and password
#username = st.text_input("Enter your Infront username:")
#password = st.text_input("Enter your Infront password:", type="password")


#if username and password:
#    infront.InfrontConnect(user=username, password=password)  # Use the provided credentials
#else:
#    st.warning("Please enter your Infront username and password to continue.")
# Connect to Infront API
infront.InfrontConnect(user="David.Lundberg.ipt", password="Infront2022!") 




def show_stage_1():
    st.title("Portfolio Setup")
    st.write("Choose a standard portfolio or create your own:")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Equity Heavy Portfolio"):
            st.session_state['use_default'] = True
            st.session_state['portfolio_file'] = "standard_equity_portfolio.csv"
            st.session_state['page'] = 2
    with col2:
        if st.button("Interest Bearing Heavy Portfolio"):
            st.session_state['use_default'] = True
            st.session_state['portfolio_file'] = "standard_interest_portfolio.csv"
            st.session_state['page'] = 2
    with col3:
        if st.button("Create Custom Portfolio"):
            st.session_state['use_default'] = False
            st.session_state['portfolio_file'] = None
            st.session_state['page'] = 2

def show_stage_2():
    st.title("Portfolio Selection & Adjustment")
    categories, display_name_to_asset_id = get_categorized_assets(ASSETS_INDICES_MAP)

    if st.session_state.get('use_default', False) and st.session_state.get('portfolio_file'):
        import pandas as pd
        df = pd.read_csv(st.session_state['portfolio_file'])
        default_assets = [a for a in df['asset']]
        default_weights = dict(zip(df['asset'], df['weight']))
        selected_shares = [ASSETS_INDICES_MAP[a]["display name"] for a in default_assets if ASSETS_INDICES_MAP[a]["category"] == "Equity"]
        selected_alternative = [ASSETS_INDICES_MAP[a]["display name"] for a in default_assets if ASSETS_INDICES_MAP[a]["category"] == "Alternative"]
        selected_interest_bearing = [ASSETS_INDICES_MAP[a]["display name"] for a in default_assets if ASSETS_INDICES_MAP[a]["category"] == "Interest Bearing"]
    else:
        selected_shares = []
        selected_alternative = []
        selected_interest_bearing = []
        default_weights = {}

    col1, col2, col3 = st.columns(3)
    with col1:
        selected_shares = st.multiselect("Select Equity:", categories["Equity"], default=selected_shares)
    with col2:
        selected_alternative = st.multiselect("Select Alternative:", categories["Alternative"], default=selected_alternative)
    with col3:
        selected_interest_bearing = st.multiselect("Select Interest Bearing:", categories["Interest Bearing"], default=selected_interest_bearing)

    selected_display_names = selected_shares + selected_alternative + selected_interest_bearing
    selected_assets = [display_name_to_asset_id[name] for name in selected_display_names]
    st.session_state['selected_assets'] = selected_assets


    weights = {}
    for asset in selected_assets:
        display_name = ASSETS_INDICES_MAP[asset].get("display name", asset)
        default_weight = default_weights.get(asset, 0.1)
        weight = st.number_input(f"Weight for {display_name}", min_value=0.0, max_value=1.0, value=default_weight)
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

    st.session_state['weights'] = weights
    

    allocation_limit = st.number_input("Allocation limit (Plus/minus %)", min_value=0, max_value=100, value=7)
    st.session_state['allocation_limit'] = allocation_limit

    start_investment = st.number_input("Start investment amount (SEK)", min_value=0, value=100000)
    st.session_state['start_investment'] = start_investment

    start_date = st.date_input("Start date", datetime(2022, 1, 1))
    end_date = st.date_input("End date", datetime.today())
    st.session_state['start_date'] = start_date
    st.session_state['end_date'] = end_date

    if st.button("Calculate Portfolio"):
        st.session_state['page'] = 3

def show_stage_3():
    st.title("Results")
    selected_assets = st.session_state.get('selected_assets', [])
    weights = st.session_state.get('weights', {})
    allocation_limit = st.session_state.get('allocation_limit', 50)
    start_date = st.session_state.get('start_date', datetime(2022, 1, 1))
    end_date = st.session_state.get('end_date', datetime.today())
    start_investment = st.session_state.get('start_investment', 100000)
    
    if not selected_assets or not weights:
        st.warning("No portfolio selected. Please go back and select assets.")
        if st.button("Back"):
            st.session_state['page'] = 1
        return

    selected_indices = list({ASSETS_INDICES_MAP[asset]["index"] for asset in selected_assets})
    combined_data = fetch_data_infront(selected_assets, selected_indices, start_date, end_date)
    combined_data = clean_data(combined_data)
    combined_data = indexed_net_to_100(combined_data)
    combined_data = period_change(combined_data)
    combined_data = OGC_adjusted_Period_Change(combined_data)
    combined_data = indexed_OGC_adjusted_to_100(combined_data)
    st.session_state['combined_data'] = combined_data

    combined_data, date_holdings_df  = create_portfolio(combined_data, weights, start_investment, allocation_limit)    
    st.write("Portfolio calculation complete!")

    
    # Generate summary report
    generate_summary_report(combined_data, date_holdings_df, start_investment, allocation_limit, weights)

    if st.button("Back"):
        st.session_state['page'] = 1

def main():
    if 'page' not in st.session_state:
        st.session_state['page'] = 1

    if st.session_state['page'] == 1:
        show_stage_1()
    elif st.session_state['page'] == 2:
        show_stage_2()
    elif st.session_state['page'] == 3:
        show_stage_3()

if __name__ == "__main__":
    main()