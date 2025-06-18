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

from visualisation import generate_summary_report, show_weights, generate_multi_summary_report, show_predictions

# Prompt for username and password
#username = st.text_input("Enter your Infront username:")
#password = st.text_input("Enter your Infront password:", type="password")


#if username and password:
#    infront.InfrontConnect(user=username, password=password)  # Use the provided credentials
#else:
#    st.warning("Please enter your Infront username and password to continue.")
# Connect to Infront API
infront.InfrontConnect(user="David.Lundberg.ipt", password="Infront2022!") 

def show_footer():
    st.markdown(
        """
        <style>
        a:link, a:visited {
            color: #4da3ff;
            background-color: transparent;
            text-decoration: underline;
        }
        a:hover, a:active {
            color: #ff4d4d;
            background-color: transparent;
            text-decoration: underline;
        }
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #1E283C;
            color: #ccc;
            text-align: center;
            padding: 1em 0 1em 0;
            z-index: 100;
        }
        </style>
        <div class="footer">
            <p>
                Evisens Portfolio Illustration Tool &copy; 2025 |
                Developed with Streamlit
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


def show_stage_1():
    st.logo("logo.png")
    st.title("Portfolio Setup")
    st.write("Choose a standard portfolio or create your own:")
    # Instruction box
    st.info(
        """
        **Instructions:**

            Select a standard portfolio or create your own by choosing assets.

            To compare multiple portfolios, choose the count and click "Compare Multiple Portfolios" (only monthly data applies).

            Assign assets and weights for each portfolio on the following pages. Ensure weights sum to 1.

            Click "Calculate Portfolio" or "Calculate & Compare" for results, which can be exported to Excel or PNG.

            For single portfolios, view predictions using models like Monte Carlo.

            Navigate back anytime.

            Edit asset/index data via "Go to Admin."

        """
    )
    st.info(
        "Tip: If you want to analyze a long time period (many years), or realise prediction models, choose 'Monthly' data frequency."
    )
    # Option to choose data frequency
    data_frequency = st.radio(
        "Data frequency:",
        options=["Daily (most detailed)", "Monthly (less detailed)", "Yearly (least detailed)"],
        index=0,
        help="The least detailed asset price will determine level of detail for the whole portfolio. Choose monthly or yearly if you want to override the default daily frequency.",
    )
    if data_frequency.startswith("Daily"):
        st.session_state['data_frequency'] = "daily"
    elif data_frequency.startswith("Monthly"):
        st.session_state['data_frequency'] = "monthly"
    elif data_frequency.startswith("Yearly"):
        st.session_state['data_frequency'] = "yearly"


    col1, col2, col3, col4, col5= st.columns(5)
    with col1:
        if st.button("Agressive Portfolio"):
            st.session_state['use_default'] = True
            st.session_state['portfolio_file'] = "standard_equity_portfolio.csv"
            st.session_state['multiple_portfolios'] = False
            st.session_state['page'] = 2
            st.rerun()
    with col2:
        if st.button("Conservative Portfolio"):
            st.session_state['use_default'] = True
            st.session_state['portfolio_file'] = "standard_interest_portfolio.csv"
            st.session_state['multiple_portfolios'] = False
            st.session_state['page'] = 2
            st.rerun()
    with col3:
        if st.button("Custom Portfolio"):
            st.session_state['use_default'] = False
            st.session_state['portfolio_file'] = None
            st.session_state['multiple_portfolios'] = False
            st.session_state['page'] = 2
            st.rerun()

    with col4:
        
        num = st.number_input(
            "Nr of portfolios",
            min_value=2, max_value=4, value=2, key="None"
        )
        if st.button("Multiple Custom Portfolios"):
            st.session_state['num_portfolios'] = num
            st.session_state['use_default'] = False
            st.session_state['portfolio_file'] = None
            st.session_state['multiple_portfolios'] = True
            st.session_state['page'] = 4
            st.rerun()
        
    with col5:
        if st.button("3 Portfolio Preset"):
            st.session_state['num_portfolios'] = 3
            st.session_state['use_default'] = True
            st.session_state['portfolio_file'] = "preset_3portfolios.csv"
            st.session_state['multiple_portfolios'] = True
            st.session_state['page'] = 4
            st.rerun()

    st.markdown("---")
    if st.button("Go to Admin"):
        st.session_state['page'] = 99
        st.rerun()
    show_footer()

def show_stage_2():
    st.logo("logo.png")
    st.title("Portfolio Selection & Adjustment")
    categories, display_name_to_asset_id = get_categorized_assets(ASSETS_INDICES_MAP)

    if st.session_state.get('use_default', False) and st.session_state.get('portfolio_file'):
        import pandas as pd
        df = pd.read_csv(st.session_state['portfolio_file'])
        pf1 = df[df['portfolio'] == 1]
        # Get default category weights
        default_category_weights = pf1.groupby('category')['category_weight'].first().to_dict()

        # Get default asset weights within each category
        default_asset_weights = {}
        for cat in pf1['category'].unique():
            cat_assets = pf1[pf1['category'] == cat]
            default_asset_weights[cat] = dict(zip(cat_assets['asset_id'], cat_assets['asset_weight']))

        # Get default selected assets by category (for pre-selecting in multiselects)
        selected_shares = [row['display_name'] for _, row in pf1[pf1['category'] == "Equity"].iterrows()]
        selected_alternative = [row['display_name'] for _, row in pf1[pf1['category'] == "Alternative"].iterrows()]
        selected_interest_bearing = [row['display_name'] for _, row in pf1[pf1['category'] == "Interest Bearing"].iterrows()]
    else:
        selected_shares = []
        selected_alternative = []
        selected_interest_bearing = []
        default_category_weights = {}  
        default_asset_weights = {}     

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


        # Group selected assets by category
    assets_by_category = {'Equity': [], 'Interest Bearing': [], 'Alternative': []}
    for asset in selected_assets:
        cat = ASSETS_INDICES_MAP[asset]['category']
        assets_by_category[cat].append(asset)

   # Only show categories that have selected assets
    selected_cats = [c for c in ['Equity', 'Interest Bearing', 'Alternative'] if assets_by_category[c]]

    st.markdown("### Set Category Weights ")
    category_weights = {}
    for cat in selected_cats:
        category_weights[cat] = st.number_input(
            f"Weight for {cat}", min_value=0.0, max_value=1.0,
            value=default_category_weights.get(cat, round(1.0/len(selected_cats), 2)) if len(selected_cats) > 0 else 0.0,
            key=f"cat_weight_{cat}"
        )
    if abs(sum(category_weights.values()) - 1.0) > 0.01:
        st.error("Category weights must sum to 1.")

    # For each category, show assets and let user set weights within category
    asset_weights = {}
    for cat in selected_cats:
        if assets_by_category[cat]:
            st.markdown(f"**{cat}:**")
            total_assets = len(assets_by_category[cat])
            asset_weights[cat] = {}
            for asset in assets_by_category[cat]:
                display_name = ASSETS_INDICES_MAP[asset].get("display name", asset)
                asset_weights[cat][asset] = st.number_input(
                    f"Weight for {display_name} in {cat}",
                    min_value=0.0, max_value=1.0,
                    value=default_asset_weights.get(cat, {}).get(asset, round(1.0/total_assets, 2)) if total_assets > 0 else 0.0,
                    key=f"{cat}_asset_weight_{asset}"
                )
            # Check weights sum to 1 within category
            if abs(sum(asset_weights[cat].values()) - 1.0) > 0.01:
                st.error(f"Weights for {cat} must sum to 1.")

    # Calculate final weights
    weights = {}
    asset_only_weights = {}
    for cat in asset_weights:
        for asset, w in asset_weights[cat].items():
            weights[asset] = category_weights[cat] * w
            asset_only_weights[asset] = category_weights[cat] * w
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
        st.error(f"The weights must add up to 1. Please adjust the weights. The summed weights is now {sum(weights.values())}")

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

    ogc_option = st.selectbox(
        "OGC setting",
        options=["Standard OGC", "Evisens OGC (if available)", "No OGC"],
        index=0
    )
    st.session_state['ogc_option'] = ogc_option

    if st.button("Calculate Portfolio"):

        st.session_state['page'] = 3
        st.rerun()
    st.markdown("---")
    if st.button("Back"):
        st.session_state['page'] = 1
        st.rerun()
    show_footer()

def show_stage_3():
    st.logo("logo.png")
    st.title("Results")
    selected_assets = st.session_state.get('selected_assets', [])
    weights = st.session_state.get('weights', {})
    asset_only_weights = st.session_state.get('asset_only_weights', {})
    allocation_limit = st.session_state.get('allocation_limit', 50)
    start_date = st.session_state.get('start_date', datetime(2022, 1, 1))
    end_date = st.session_state.get('end_date', datetime.today())
    start_investment = st.session_state.get('start_investment', 100000)
    
    ogc_option = st.session_state.get('ogc_option', "Standard OGC")

    data_frequency = st.session_state.get('data_frequency', "daily")
    
    if not selected_assets or not weights:
        st.warning("No portfolio selected. Please go back and select assets.")
        if st.button("Back"):
            st.session_state['page'] = 1
            st.rerun()
        return

    selected_indices = list({ASSETS_INDICES_MAP[asset]["index"] for asset in selected_assets})
    combined_data = fetch_data_infront(selected_assets, selected_indices, start_date, end_date)
    combined_data, period, data_frequency = clean_data(combined_data, data_frequency)
    combined_data = indexed_net_to_100(combined_data)
    combined_data = period_change(combined_data)
    combined_data = OGC_adjusted_Period_Change(combined_data, period, ogc_option)
    combined_data = indexed_OGC_adjusted_to_100(combined_data)
    st.session_state['combined_data'] = combined_data

    combined_data, date_holdings_df  = create_portfolio(combined_data, weights, start_investment, allocation_limit)    
    st.write("Portfolio calculation complete!")

    
    # Generate summary report
    generate_summary_report(combined_data, date_holdings_df, start_investment, allocation_limit, weights, asset_only_weights, period)
    
    if data_frequency == "monthly":
        if st.button("Go to Predictions"):
            st.session_state['page'] = 6
            st.rerun()
    st.markdown("---")
    if st.button("Back"):
        st.session_state['page'] = 2
        st.rerun()
    show_footer()

def show_stage_4():
    st.logo("logo.png")
    num_portfolios = st.session_state.get('num_portfolios', 2)
    st.title("Compare Multiple Portfolios")
    categories, display_name_to_asset_id = get_categorized_assets(ASSETS_INDICES_MAP)
    use_default = st.session_state.get('use_default', False)
    # --- Load defaults if requested ---
    default_category_weights_list = [{} for _ in range(num_portfolios)]
    default_asset_weights_list = [{} for _ in range(num_portfolios)]
    default_selected_shares = [[] for _ in range(num_portfolios)]
    default_selected_alternative = [[] for _ in range(num_portfolios)]
    default_selected_interest_bearing = [[] for _ in range(num_portfolios)]

    if use_default and st.session_state.get('portfolio_file'):
        df = pd.read_csv(st.session_state['portfolio_file'])
        for i in range(num_portfolios):
            pf = df[df['portfolio'] == i+1]
            # Category weights
            default_category_weights_list[i] = pf.groupby('category')['category_weight'].first().to_dict()
            # Asset weights
            default_asset_weights = {}
            for cat in pf['category'].unique():
                cat_assets = pf[pf['category'] == cat]
                default_asset_weights[cat] = dict(zip(cat_assets['asset_id'], cat_assets['asset_weight']))
            default_asset_weights_list[i] = default_asset_weights
            # Selected assets by category
            default_selected_shares[i] = [row['display_name'] for _, row in pf[pf['category'] == "Equity"].iterrows()]
            default_selected_alternative[i] = [row['display_name'] for _, row in pf[pf['category'] == "Alternative"].iterrows()]
            default_selected_interest_bearing[i] = [row['display_name'] for _, row in pf[pf['category'] == "Interest Bearing"].iterrows()]


    # --- Initialize session state for multi_portfolios if not already ---
    if 'multi_portfolios' not in st.session_state or len(st.session_state['multi_portfolios']) != num_portfolios:
        st.session_state['multi_portfolios'] = [
            {'selected_shares': [], 'selected_alternative': [], 'selected_interest_bearing': [], 'weights': {}, 'asset_only_weights': {}, 'selected_assets': []}
            for _ in range(num_portfolios)
        ]

    cols = st.columns(num_portfolios)
    for i, col in enumerate(cols):
        with col:
            st.markdown(f"### Portfolio {i+1}")
            mp = st.session_state['multi_portfolios'][i]

            # Use defaults if available, else use session state
            shares = default_selected_shares[i] if use_default and default_selected_shares[i] else mp['selected_shares']
            alt = default_selected_alternative[i] if use_default and default_selected_alternative[i] else mp['selected_alternative']
            intb = default_selected_interest_bearing[i] if use_default and default_selected_interest_bearing[i] else mp['selected_interest_bearing']

            mp['selected_shares'] = st.multiselect(
                f"Select Equity (Portfolio {i+1})", 
                categories["Equity"], 
                default=shares, 
                key=f"shares_{i}"
            )
            mp['selected_alternative'] = st.multiselect(
                f"Select Alternative (Portfolio {i+1})", 
                categories["Alternative"], 
                default=alt, 
                key=f"alt_{i}"
            )
            mp['selected_interest_bearing'] = st.multiselect(
                f"Select Interest Bearing (Portfolio {i+1})", 
                categories["Interest Bearing"], 
                default=intb, 
                key=f"int_{i}"
            )
            selected_display_names = mp['selected_shares'] + mp['selected_alternative'] + mp['selected_interest_bearing']
            selected_assets = [display_name_to_asset_id[name] for name in selected_display_names]
            mp['selected_assets'] = selected_assets

            # Group selected assets by category
            assets_by_category = {'Equity': [], 'Interest Bearing': [], 'Alternative': []}
            for asset in selected_assets:
                cat = ASSETS_INDICES_MAP[asset]['category']
                assets_by_category[cat].append(asset)
            selected_cats = [c for c in ['Equity', 'Interest Bearing', 'Alternative'] if assets_by_category[c]]

            st.markdown("#### Set Category Weights")
            category_weights = {}
            default_cat_weights = default_category_weights_list[i]
            for cat in selected_cats:
                category_weights[cat] = st.number_input(
                    f"Weight for {cat} (Portfolio {i+1})", min_value=0.0, max_value=1.0,
                    value=default_cat_weights.get(cat, round(1.0/len(selected_cats), 2)) if len(selected_cats) > 0 else 0.0,
                    key=f"cat_weight_{cat}_{i}"
                )
            if abs(sum(category_weights.values()) - 1.0) > 0.01:
                st.error("Category weights must sum to 1.")

            # For each category, show assets and let user set weights within category
            asset_weights = {}
            default_asset_weights = default_asset_weights_list[i]
            for cat in selected_cats:
                if assets_by_category[cat]:
                    st.markdown(f"**{cat}:**")
                    total_assets = len(assets_by_category[cat])
                    asset_weights[cat] = {}
                    for asset in assets_by_category[cat]:
                        display_name = ASSETS_INDICES_MAP[asset].get("display name", asset)
                        asset_weights[cat][asset] = st.number_input(
                            f"Weight for {display_name} in {cat} (Portfolio {i+1})",
                            min_value=0.0, max_value=1.0,
                            value=default_asset_weights.get(cat, {}).get(asset, round(1.0/total_assets, 2)) if total_assets > 0 else 0.0,
                            key=f"{cat}_asset_weight_{asset}_{i}"
                        )
                    # Check weights sum to 1 within category
                    if abs(sum(asset_weights[cat].values()) - 1.0) > 0.01:
                        st.error(f"Weights for {cat} must sum to 1.")

            # Calculate final weights
            weights = {}
            asset_only_weights = {}
            for cat in asset_weights:
                for asset, w in asset_weights[cat].items():
                    weights[asset] = category_weights[cat] * w
                    asset_only_weights[asset] = category_weights[cat] * w
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
                st.error("The asset weights must add up to 1. Please adjust the weights.")


            # Show weights pie chart
            show_weights(asset_only_weights, key=f"weights_chart_{i}")

            # Choose to use OGC or not
            ogc_option = st.selectbox(
                f"OGC setting for Portfolio {i+1}",
                options=["Standard OGC", "Evisens OGC (if available)", "No OGC (OGC = 0)"],
                index=0,
                key=f"ogc_option_{i}"
            )

            #choose specific allocation limit   
            allocation_limit = st.number_input(
                f"Allocation limit for Portfolio {i+1} (Plus/minus %)", 
                min_value=0, max_value=100, value=7, key=f"allocation_limit_{i}"
            )


    #Start investment amount
    start_investment = st.number_input("Start investment amount (SEK)", min_value=0, value=100000)
    st.session_state['start_investment'] = start_investment

    # Start and end date
    start_date = st.date_input("Start date", datetime(2022, 1, 1))  
    end_date = st.date_input("End date", datetime.today())
    st.session_state['start_date'] = start_date
    st.session_state['end_date'] = end_date

    #allocation limit
    # allocation_limit = st.number_input("Allocation limit (Plus/minus %)", min_value=0, max_value=100, value=7)
    # st.session_state['allocation_limit'] = allocation_limit


    # Save back to session state
    st.session_state['multi_portfolios'] = st.session_state['multi_portfolios']


    if st.button("Calculate & Compare"):
        st.session_state['page'] = 5  # You can add a new stage for results/plotting
        st.rerun()

    st.markdown("---")
    if st.button("Back"):
        st.session_state['page'] = 1
        st.rerun()
    show_footer()

def show_stage_5():
    st.logo("logo.png")
    st.title("Portfolio Comparison Results")
    portfolios = st.session_state.get('multi_portfolios', [])
    # allocation_limit = st.session_state.get('allocation_limit', 50)
    start_date = st.session_state.get('start_date', datetime(2022, 1, 1))
    end_date = st.session_state.get('end_date', datetime.today())
    start_investment = st.session_state.get('start_investment', 100000)
    data_frequency = st.session_state.get('data_frequency', "daily")
    
    if not portfolios:
        st.warning("No portfolios selected. Please go back and select portfolios.")
        if st.button("Back"):
            st.session_state['page'] = 1
            st.rerun()
        return

    # Dict to hold the finished portfolios
    finished_portfolios = {}
    # Iterate through each portfolio and fetch data

    # Fetch data for each portfolio and calculate results
    for i, portfolio in enumerate(portfolios):
        allocation_limit = st.session_state.get(f'allocation_limit_{i}', 50)
        selected_assets = portfolio['selected_assets']
        weights = portfolio['weights']
        asset_only_weights = portfolio['asset_only_weights']
        ogc_option = st.session_state.get(f'ogc_option_{i}', "Standard OGC")
        st.write(f"OGC option for Portfolio {i+1}: {ogc_option}")
        
        if not selected_assets or not weights:
            st.warning(f"Portfolio {i+1} is incomplete. Please go back and select assets.")
            continue

        selected_indices = list({ASSETS_INDICES_MAP[asset]["index"] for asset in selected_assets})
        combined_data = fetch_data_infront(selected_assets, selected_indices, start_date, end_date)
        combined_data, period, data_frequency = clean_data(combined_data, data_frequency, True)
        combined_data = indexed_net_to_100(combined_data)
        combined_data = period_change(combined_data)
        combined_data = OGC_adjusted_Period_Change(combined_data, period, ogc_option)
        combined_data = indexed_OGC_adjusted_to_100(combined_data)

        combined_data, date_holdings_df  = create_portfolio(combined_data, weights, start_investment, allocation_limit)    
        
        # save the portfolio data
        finished_portfolios[f"Portfolio {i+1}"] = {
            "combined_data": combined_data,
            "date_holdings_df": date_holdings_df,
            "weights": weights,
            "asset_only_weights": asset_only_weights,
            "period": period,
            "start_investment": start_investment
        }
        print("Period ", period)
        # Generate summary report for each portfolio
        #generate_summary_report(combined_data, date_holdings_df, start_investment, allocation_limit, weights, asset_only_weights, period)

    # Generate multi-portfolio summary report
    generate_multi_summary_report(finished_portfolios, allocation_limit)
    st.markdown("---")
    if st.button("Back"):
        st.session_state['page'] = 1
        st.rerun()
    show_footer()


def show_stage_6():
    st.logo("logo.png")
    data_frequency = st.session_state.get('data_frequency', "daily")

    st.title("Predicted Portfolio")
    combined_data = st.session_state.get('combined_data', None)
    show_predictions(combined_data, data_frequency)
    st.markdown("---")
    if st.button("Back"):
        st.session_state['page'] = 3
        st.rerun()
    show_footer()


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
    elif st.session_state['page'] == 6:
        show_stage_6()

    elif st.session_state['page'] == 99:
        show_asset_indices_admin()

if __name__ == "__main__":
    main()