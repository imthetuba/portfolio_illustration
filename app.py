import streamlit as st
import pandas as pd
from InfrontConnect import infront
from admin import show_asset_indices_admin
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

from visualisation import generate_summary_report, show_weights, generate_multi_summary_report

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
    # Instruction box
    st.info(
        """
        **Instructions:**
        1. Choose a standard portfolio or create your own by selecting assets.
        2. If you want to compare multiple portfolios, select the number and click "Compare Multiple Portfolios".
        3. On the next pages, select assets and set weights for each portfolio.
        4. Make sure the weights for each portfolio add up to 1.
        5. Click "Calculate Portfolio" or "Calculate & Compare" to see results. The results will be displayed in the next page.
        6. You can export the results to Excel. The plots are downloadable as transparent PNG files.
        7. You can go back to the previous page at any time.
        8. If you want to edit the asset/index information, click "Go to Admin".
        9. If you want to edit the color scheme or language of the plots, change the constants in the `visualisation.py` file.
        """
    )
    # Option to choose data frequency
    data_frequency = st.radio(
        "Data frequency:",
        options=["Daily (most detailed)", "Monthly (less detailed)", "Yearly (least detailed)"],
        index=0,
        help="The least detailed asset price will determine level of detail. Choose monthly or yearly if you want to override the default daily frequency.",
    )
    if data_frequency.startswith("Daily"):
        st.session_state['data_frequency'] = "daily"
    elif data_frequency.startswith("Monthly"):
        st.session_state['data_frequency'] = "monthly"
    elif data_frequency.startswith("Yearly"):
        st.session_state['data_frequency'] = "yearly"


    col1, col2, col3, col4, col5= st.columns(5)
    with col1:
        if st.button("Equity Heavy Portfolio"):
            st.session_state['use_default'] = True
            st.session_state['portfolio_file'] = "standard_equity_portfolio.csv"
            st.session_state['multiple_portfolios'] = False
            st.session_state['page'] = 2
    with col2:
        if st.button("Interest Bearing Heavy Portfolio"):
            st.session_state['use_default'] = True
            st.session_state['portfolio_file'] = "standard_interest_portfolio.csv"
            st.session_state['multiple_portfolios'] = False
            st.session_state['page'] = 2
    with col3:
        if st.button("Create One Custom Portfolio"):
            st.session_state['use_default'] = False
            st.session_state['portfolio_file'] = None
            st.session_state['multiple_portfolios'] = False
            st.session_state['page'] = 2

    with col4:
        num = st.number_input(
            "How many portfolios do you want to compare?",
            min_value=2, max_value=4, value=2, key="None"
        )
        
        if st.button("Compare Multiple Portfolios"):
            st.session_state['num_portfolios'] = num
            st.session_state['use_default'] = False
            st.session_state['portfolio_file'] = None
            st.session_state['multiple_portfolios'] = True
            st.session_state['page'] = 4
    with col5:
        if st.button("Load 3 Portfolio Preset from CSV"):
            df = pd.read_csv("preset_3portfolios.csv")
            st.session_state['num_portfolios'] = 3
            st.session_state['use_default'] = True
            st.session_state['portfolio_file'] = None
            st.session_state['multiple_portfolios'] = True
            st.session_state['page'] = 4

    st.markdown("---")
    if st.button("Go to Admin"):
        st.session_state['page'] = 99

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
    asset_only_weights = {}
    for asset in selected_assets:
        display_name = ASSETS_INDICES_MAP[asset].get("display name", asset)
        default_weight = default_weights.get(asset, 0.1)
        weight = st.number_input(f"Weight for {display_name}", min_value=0.0, max_value=1.0, value=default_weight)
        weights[asset] = weight
        asset_only_weights[asset] = weight
    
    # Sum the weights for the indices if there are duplicates
    for asset in selected_assets:
        index = ASSETS_INDICES_MAP[asset]['index']
        if index in weights:
            weights[index] += weights[asset]
        else:
            weights[index] = weights[asset]

    # dynamically show the asset weights
    
    show_weights(asset_only_weights)


    # Check if weights add up to 1
    if sum(weights.values()) != 2.0: # double because of the index
        st.error("The weights must add up to 1. Please adjust the weights.")

    st.session_state['weights'] = weights
    st.session_state['asset_only_weights'] = asset_only_weights
    

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
    asset_only_weights = st.session_state.get('asset_only_weights', {})
    allocation_limit = st.session_state.get('allocation_limit', 50)
    start_date = st.session_state.get('start_date', datetime(2022, 1, 1))
    end_date = st.session_state.get('end_date', datetime.today())
    start_investment = st.session_state.get('start_investment', 100000)
    
    data_frequency = st.session_state.get('data_frequency', "daily")
    
    if not selected_assets or not weights:
        st.warning("No portfolio selected. Please go back and select assets.")
        if st.button("Back"):
            st.session_state['page'] = 1
        return

    selected_indices = list({ASSETS_INDICES_MAP[asset]["index"] for asset in selected_assets})
    combined_data = fetch_data_infront(selected_assets, selected_indices, start_date, end_date)
    combined_data, period = clean_data(combined_data, data_frequency)
    combined_data = indexed_net_to_100(combined_data)
    combined_data = period_change(combined_data)
    combined_data = OGC_adjusted_Period_Change(combined_data, period)
    combined_data = indexed_OGC_adjusted_to_100(combined_data)
    st.session_state['combined_data'] = combined_data

    combined_data, date_holdings_df  = create_portfolio(combined_data, weights, start_investment, allocation_limit)    
    st.write("Portfolio calculation complete!")

    
    # Generate summary report
    generate_summary_report(combined_data, date_holdings_df, start_investment, allocation_limit, weights, asset_only_weights, period)

    if st.button("Back"):
        st.session_state['page'] = 1

def show_stage_4():
    num_portfolios = st.session_state.get('num_portfolios', 2)
    st.title("Compare Multiple Portfolios")
    categories, display_name_to_asset_id = get_categorized_assets(ASSETS_INDICES_MAP)
    use_default = st.session_state.get('use_default', False)
    if use_default:
        df = pd.read_csv("preset_3portfolios.csv")
        # Ensure 'multi_portfolios' is initialized
        if 'multi_portfolios' not in st.session_state or len(st.session_state['multi_portfolios']) != 3:
            st.session_state['multi_portfolios'] = [
                {'selected_shares': [], 'selected_alternative': [], 'selected_interest_bearing': [], 'weights': {}, 'asset_only_weights': {}, 'selected_assets': []}
                for _ in range(3)
            ]
        for i in range(3):
            pf = df[df['portfolio'] == i+1]
            # Only include display names that exist in the categories
            shares = [name for name in pf[pf['category'] == "Equity"]['display_name'] if name in categories["Equity"]]
            alt = [name for name in pf[pf['category'] == "Alternative"]['display_name'] if name in categories["Alternative"]]
            intb = [name for name in pf[pf['category'] == "Interest Bearing"]['display_name'] if name in categories["Interest Bearing"]]
            assets = pf['asset_id'].tolist()
            weights = dict(zip(pf['asset_id'], pf['weight']))
            asset_only_weights = dict(zip(pf['asset_id'], pf['weight']))
            st.session_state['multi_portfolios'][i]['selected_shares'] = shares
            st.session_state['multi_portfolios'][i]['selected_alternative'] = alt
            st.session_state['multi_portfolios'][i]['selected_interest_bearing'] = intb
            st.session_state['multi_portfolios'][i]['selected_assets'] = assets
            st.session_state['multi_portfolios'][i]['weights'] = weights
            st.session_state['multi_portfolios'][i]['asset_only_weights'] = asset_only_weights

    # Initialize session state for multi_portfolios if not already
        # Initialize session state for multi_portfolios if not already
    if 'multi_portfolios' not in st.session_state:
        st.session_state['multi_portfolios'] = [
            {'selected_shares': [], 'selected_alternative': [], 'selected_interest_bearing': [], 'weights': {}} 
            for _ in range(num_portfolios)
        ]
    else:
        # Adjust the length without resetting existing data
        current_len = len(st.session_state['multi_portfolios'])
        if num_portfolios > current_len:
            for _ in range(num_portfolios - current_len):
                st.session_state['multi_portfolios'].append(
                    {'selected_shares': [], 'selected_alternative': [], 'selected_interest_bearing': [], 'weights': {}}
                )
        elif num_portfolios < current_len:
            st.session_state['multi_portfolios'] = st.session_state['multi_portfolios'][:num_portfolios]

    cols = st.columns(num_portfolios)
    for i, col in enumerate(cols):
        with col:
            st.markdown(f"### Portfolio {i+1}")
            mp = st.session_state['multi_portfolios'][i]
            mp['selected_shares'] = st.multiselect(
                f"Select Equity (Portfolio {i+1})", 
                categories["Equity"], 
                default=mp['selected_shares'], 
                key=f"shares_{i}"
            )
            mp['selected_alternative'] = st.multiselect(
                f"Select Alternative (Portfolio {i+1})", 
                categories["Alternative"], 
                default=mp['selected_alternative'], 
                key=f"alt_{i}"
            )
            mp['selected_interest_bearing'] = st.multiselect(
                f"Select Interest Bearing (Portfolio {i+1})", 
                categories["Interest Bearing"], 
                default=mp['selected_interest_bearing'], 
                key=f"int_{i}"
            )
            selected_display_names = mp['selected_shares'] + mp['selected_alternative'] + mp['selected_interest_bearing']
            selected_assets = sorted([display_name_to_asset_id[name] for name in selected_display_names])
            mp['selected_assets'] = selected_assets

            # Weights
            
            weights = {}
            asset_only_weights = {}
            for asset in selected_assets:
                display_name = ASSETS_INDICES_MAP[asset].get("display name", asset)
                weight = st.number_input(
                    f"Weight for {display_name} (Portfolio {i+1})",
                    min_value=0.0, max_value=1.0,
                    value=mp['weights'].get(asset, 0.1),
                    key=f"weight_{i}_{asset}"
                )
                weights[asset] = weight
                asset_only_weights[asset] = weight
            # Sum the weights for the indices if there are duplicates
            for asset in selected_assets:
                index = ASSETS_INDICES_MAP[asset]['index']
                if index in weights:
                    weights[index] += weights[asset]
                else:
                    weights[index] = weights[asset]

            mp['weights'] = weights
            mp['asset_only_weights'] = asset_only_weights
            

                    
            # Optional: check if weights sum to 1
            if abs(sum(asset_only_weights.values()) - 1.0) > 0.01:
                st.error("The weights must add up to 1. Please adjust the weights.")

            # Show weights pie chart
            show_weights(asset_only_weights, key=f"weights_chart_{i}")


    #Start investment amount
    start_investment = st.number_input("Start investment amount (SEK)", min_value=0, value=100000)
    st.session_state['start_investment'] = start_investment

    # Start and end date
    start_date = st.date_input("Start date", datetime(2022, 1, 1))  
    end_date = st.date_input("End date", datetime.today())
    st.session_state['start_date'] = start_date
    st.session_state['end_date'] = end_date

    #allocation limit
    allocation_limit = st.number_input("Allocation limit (Plus/minus %)", min_value=0, max_value=100, value=7)
    st.session_state['allocation_limit'] = allocation_limit


    # Save back to session state
    st.session_state['multi_portfolios'] = st.session_state['multi_portfolios']


    if st.button("Calculate & Compare"):
        st.session_state['page'] = 5  # You can add a new stage for results/plotting

    if st.button("Back"):
        st.session_state['page'] = 1

def show_stage_5():
    st.title("Portfolio Comparison Results")
    portfolios = st.session_state.get('multi_portfolios', [])
    allocation_limit = st.session_state.get('allocation_limit', 50)
    start_date = st.session_state.get('start_date', datetime(2022, 1, 1))
    end_date = st.session_state.get('end_date', datetime.today())
    start_investment = st.session_state.get('start_investment', 100000)
    data_frequency = st.session_state.get('data_frequency', "daily")
    
    if not portfolios:
        st.warning("No portfolios selected. Please go back and select portfolios.")
        if st.button("Back"):
            st.session_state['page'] = 1
        return

    # Dict to hold the finished portfolios
    finished_portfolios = {}
    # Iterate through each portfolio and fetch data

    # Fetch data for each portfolio and calculate results
    for i, portfolio in enumerate(portfolios):
        selected_assets = portfolio['selected_assets']
        weights = portfolio['weights']
        asset_only_weights = portfolio['asset_only_weights']    

        
        if not selected_assets or not weights:
            st.warning(f"Portfolio {i+1} is incomplete. Please go back and select assets.")
            continue

        selected_indices = list({ASSETS_INDICES_MAP[asset]["index"] for asset in selected_assets})
        combined_data = fetch_data_infront(selected_assets, selected_indices, start_date, end_date)
        combined_data, period = clean_data(combined_data, data_frequency, True)
        combined_data = indexed_net_to_100(combined_data)
        combined_data = period_change(combined_data)
        combined_data = OGC_adjusted_Period_Change(combined_data, period)
        combined_data = indexed_OGC_adjusted_to_100(combined_data)

        combined_data, date_holdings_df  = create_portfolio(combined_data, weights, start_investment, allocation_limit)    
        
        # save the portfolio data
        finished_portfolios[f"Portfolio {i+1}"] = {
            "combined_data": combined_data,
            "date_holdings_df": date_holdings_df,
            "weights": weights,
            "asset_only_weights": asset_only_weights,
            "period": period
        }
        print("Period ", period)
        # Generate summary report for each portfolio
        #generate_summary_report(combined_data, date_holdings_df, start_investment, allocation_limit, weights, asset_only_weights, period)

    # Generate multi-portfolio summary report
    generate_multi_summary_report(finished_portfolios, allocation_limit)

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
    elif st.session_state['page'] == 4:
        show_stage_4()
    elif st.session_state['page'] == 5:
        show_stage_5()
    elif st.session_state['page'] == 99:
        show_asset_indices_admin()

if __name__ == "__main__":
    main()