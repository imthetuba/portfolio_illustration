import streamlit as st
import pandas as pd
from InfrontConnect import infront
from admin import show_asset_indices_admin
from portfolio import (
    get_categorized_assets,
    get_categorized_indices,
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

from visualisation import generate_summary_report, show_weights, generate_multi_summary_report, show_predictions, generate_multi_summary_report_indices

# Prompt for username and password
username = st.text_input("Enter your Infront username:")
password = st.text_input("Enter your Infront password:", type="password")


if username and password:
   infront.InfrontConnect(user=username, password=password)  # Use the provided credentials
else:
   st.warning("Please enter your Infront username and password to continue.")

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
    st.title("Portfolio Analysis Tool")
    
    # Instruction box
    st.info(
        """
        **Welcome to the Portfolio Analysis Tool**

        Choose your analysis type from the tabs below:
        - **Index Portfolios**: Compare portfolios built with market indices
        - **Asset Portfolios**: Compare portfolios built with individual assets  
        - **Portfolio Optimization**: Optimize portfolios using mean-variance analysis

        Select data frequency, number of portfolios, and whether to use presets.
        Results can be exported to Excel or PNG format.
        """
    )
    
    # Data frequency selection (applies to all tabs)
    st.markdown("### Data Frequency")
    data_frequency = st.radio(
        "Choose data frequency for analysis:",
        options=["Daily", "Monthly", "Yearly"],
        index=0,
        help="The least detailed asset price will determine level of detail for the whole portfolio. Choose monthly or yearly if you want to override the default daily frequency. The detail will be as granular as it can be (if one position is monthly, the whole portfolio will be monthly, even if some positions are daily).",
    )
    if data_frequency.startswith("Daily"):
        st.session_state['data_frequency'] = "daily"
    elif data_frequency.startswith("Monthly"):
        st.session_state['data_frequency'] = "monthly"
    elif data_frequency.startswith("Yearly"):
        st.session_state['data_frequency'] = "yearly"

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📈 Index Portfolios", "💼 Asset Portfolios", "🎯 Portfolio Optimization"])
    
    with tab1:
        st.markdown("### Index Portfolio Analysis")
        st.write("Build and compare portfolios using market indices (MSCI, bond indices, alternative indices, etc.)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Portfolio Settings")
            num_index_portfolios = st.number_input(
                "Number of index portfolios to compare",
                min_value=1, max_value=4, value=3, key="num_index_portfolios"
            )
            
            use_index_preset = st.checkbox("Use preset portfolios", key="use_index_preset")
            
            if use_index_preset:
                st.info("Preset will create Conservative, Balanced, and Aggressive index portfolios")
        
        with col2:
            st.markdown("#### Start Analysis")
            if st.button("🚀 Analyze Index Portfolios", type="primary"):
                st.session_state['num_portfolios'] = num_index_portfolios
                st.session_state['use_default'] = use_index_preset
                if use_index_preset:
                    st.session_state['portfolio_file'] = "preset_3_indices.csv"
                else:
                    st.session_state['portfolio_file'] = None
                st.session_state['multiple_portfolios'] = True
                st.session_state['page'] = 7
                st.rerun()
    
    with tab2:
        st.markdown("### Asset Portfolio Analysis")
        st.write("Build and compare portfolios using individual assets (funds, ETFs, stocks, etc.)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Portfolio Settings")
            num_asset_portfolios = st.number_input(
                "Number of asset portfolios to compare",
                min_value=1, max_value=4, value=3, key="num_asset_portfolios"
            )
            
            preset_type = st.selectbox(
                "Choose portfolio type:",
                options=["Custom Portfolios", "3 Portfolio Preset", "Single Aggressive", "Single Conservative"],
                key="asset_preset_type"
            )
        
        with col2:
            st.markdown("#### Start Analysis")
            if st.button("🚀 Analyze Asset Portfolios", type="primary"):
                if preset_type == "Custom Portfolios":
                    st.session_state['num_portfolios'] = num_asset_portfolios
                    st.session_state['use_default'] = False
                    st.session_state['portfolio_file'] = None
                    st.session_state['multiple_portfolios'] = True
                    st.session_state['page'] = 4
                elif preset_type == "3 Portfolio Preset":
                    st.session_state['num_portfolios'] = 3
                    st.session_state['use_default'] = True
                    st.session_state['portfolio_file'] = "preset_3portfolios.csv"
                    st.session_state['multiple_portfolios'] = True
                    st.session_state['page'] = 4
                elif preset_type == "Single Aggressive":
                    st.session_state['use_default'] = True
                    st.session_state['portfolio_file'] = "standard_equity_portfolio.csv"
                    st.session_state['multiple_portfolios'] = False
                    st.session_state['page'] = 2
                elif preset_type == "Single Conservative":
                    st.session_state['use_default'] = True
                    st.session_state['portfolio_file'] = "standard_interest_portfolio.csv"
                    st.session_state['multiple_portfolios'] = False
                    st.session_state['page'] = 2
                st.rerun()
                
            
    
    with tab3:
        st.markdown("### Portfolio Optimization")
        st.write("Optimize portfolio weights using modern portfolio theory and mean-variance analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Optimization Settings")
            optimization_method = st.selectbox(
                "Optimization method:",
                options=["Mean-Variance (Markowitz)", "Risk Parity", "Maximum Sharpe Ratio", "Minimum Volatility"],
                key="optimization_method"
            )
            
            constraints = st.multiselect(
                "Constraints:",
                options=["Maximum weight per asset", "Minimum weight per asset", "Sector limits", "ESG constraints"],
                key="optimization_constraints"
            )
            
            risk_free_rate = st.number_input(
                "Risk-free rate (%)",
                min_value=0.0, max_value=10.0, value=2.0, step=0.1,
                key="risk_free_rate"
            )
        
        with col2:
            st.markdown("#### Start Optimization")
            st.warning("🚧 Portfolio optimization functionality is coming soon!")
            
            if st.button("🚀 Optimize Portfolio", disabled=True):
                # Future implementation
                st.info("This feature will be implemented in the next version")
                # st.session_state['optimization_method'] = optimization_method
                # st.session_state['optimization_constraints'] = constraints
                # st.session_state['risk_free_rate'] = risk_free_rate
                # st.session_state['page'] = 9  # New optimization page
                # st.rerun()
    
    # Admin section at bottom
    st.markdown("---")
    col_admin1, col_admin2 = st.columns([3, 1])
    with col_admin1:
        st.markdown("*For advanced users: Access the admin panel to manage assets and indices*")
    with col_admin2:
        if st.button("⚙️ Admin Panel"):
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
            f" {cat}", min_value=0.0, max_value=1.0,
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
                    f"{display_name}",
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

    start_date = st.date_input("Start date", datetime(2022, 1, 1), min_value=datetime(2005, 1, 1))
    end_date = st.date_input("End date", datetime.today(), min_value=datetime(2005, 1, 1))
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

# def show_stage_4():
#     st.logo("logo.png")
#     num_portfolios = st.session_state.get('num_portfolios', 2)
#     st.title("Compare Multiple Portfolios")
#     categories, display_name_to_asset_id = get_categorized_assets(ASSETS_INDICES_MAP)
#     use_default = st.session_state.get('use_default', False)
#     # --- Load defaults if requested ---
#     default_category_weights_list = [{} for _ in range(num_portfolios)]
#     default_asset_weights_list = [{} for _ in range(num_portfolios)]
#     default_selected_shares = [[] for _ in range(num_portfolios)]
#     default_selected_alternative = [[] for _ in range(num_portfolios)]
#     default_selected_interest_bearing = [[] for _ in range(num_portfolios)]

#     if use_default and st.session_state.get('portfolio_file'):
#         df = pd.read_csv(st.session_state['portfolio_file'])
#         for i in range(num_portfolios):
#             pf = df[df['portfolio'] == i+1]
#             # Category weights
#             default_category_weights_list[i] = pf.groupby('category')['category_weight'].first().to_dict()
#             # Asset weights
#             default_asset_weights = {}
#             for cat in pf['category'].unique():
#                 cat_assets = pf[pf['category'] == cat]
#                 default_asset_weights[cat] = dict(zip(cat_assets['asset_id'], cat_assets['asset_weight']))
#             default_asset_weights_list[i] = default_asset_weights
#             # Selected assets by category
#             default_selected_shares[i] = [row['display_name'] for _, row in pf[pf['category'] == "Equity"].iterrows()]
#             default_selected_alternative[i] = [row['display_name'] for _, row in pf[pf['category'] == "Alternative"].iterrows()]
#             default_selected_interest_bearing[i] = [row['display_name'] for _, row in pf[pf['category'] == "Interest Bearing"].iterrows()]


#     # --- Initialize session state for multi_portfolios if not already ---
#     if 'multi_portfolios' not in st.session_state or len(st.session_state['multi_portfolios']) != num_portfolios:
#         st.session_state['multi_portfolios'] = [
#             {'selected_shares': [], 'selected_alternative': [], 'selected_interest_bearing': [], 'weights': {}, 'asset_only_weights': {}, 'selected_assets': []}
#             for _ in range(num_portfolios)
#         ]

#     cols = st.columns(num_portfolios)
#     for i, col in enumerate(cols):
#         with col:
#             st.markdown(f"### Portfolio {i+1}")
#             mp = st.session_state['multi_portfolios'][i]

#             # Use defaults if available, else use session state
#             shares = default_selected_shares[i] if use_default and default_selected_shares[i] else mp['selected_shares']
#             alt = default_selected_alternative[i] if use_default and default_selected_alternative[i] else mp['selected_alternative']
#             intb = default_selected_interest_bearing[i] if use_default and default_selected_interest_bearing[i] else mp['selected_interest_bearing']

#             mp['selected_shares'] = st.multiselect(
#                 f"Select Equity (Portfolio {i+1})", 
#                 categories["Equity"], 
#                 default=shares, 
#                 key=f"shares_{i}"
#             )
#             mp['selected_alternative'] = st.multiselect(
#                 f"Select Alternative (Portfolio {i+1})", 
#                 categories["Alternative"], 
#                 default=alt, 
#                 key=f"alt_{i}"
#             )
#             mp['selected_interest_bearing'] = st.multiselect(
#                 f"Select Interest Bearing (Portfolio {i+1})", 
#                 categories["Interest Bearing"], 
#                 default=intb, 
#                 key=f"int_{i}"
#             )
#             selected_display_names = mp['selected_shares'] + mp['selected_alternative'] + mp['selected_interest_bearing']
#             selected_assets = [display_name_to_asset_id[name] for name in selected_display_names]
#             mp['selected_assets'] = selected_assets

#             # Group selected assets by category
#             assets_by_category = {'Equity': [], 'Interest Bearing': [], 'Alternative': []}
#             for asset in selected_assets:
#                 cat = ASSETS_INDICES_MAP[asset]['category']
#                 assets_by_category[cat].append(asset)
#             selected_cats = [c for c in ['Equity', 'Interest Bearing', 'Alternative'] if assets_by_category[c]]

#             st.markdown("#### Set Category Weights")
#             category_weights = {}
#             default_cat_weights = default_category_weights_list[i]
#             for cat in selected_cats:
#                 category_weights[cat] = st.number_input(
#                     f"{cat}", min_value=0.0, max_value=1.0,
#                     value=default_cat_weights.get(cat, round(1.0/len(selected_cats), 2)) if len(selected_cats) > 0 else 0.0,
#                     key=f"cat_weight_{cat}_{i}"
#                 )
#             if abs(sum(category_weights.values()) - 1.0) > 0.01:
#                 st.error("Category weights must sum to 1.")

#             # For each category, show assets and let user set weights within category
#             asset_weights = {}
#             default_asset_weights = default_asset_weights_list[i]
#             for cat in selected_cats:
#                 if assets_by_category[cat]:
#                     st.markdown(f"**{cat}:**")
#                     total_assets = len(assets_by_category[cat])
#                     asset_weights[cat] = {}
#                     for asset in assets_by_category[cat]:
#                         display_name = ASSETS_INDICES_MAP[asset].get("display name", asset)
#                         asset_weights[cat][asset] = st.number_input(
#                             f"{display_name}",
#                             min_value=0.0, max_value=1.0,
#                             value=default_asset_weights.get(cat, {}).get(asset, round(1.0/total_assets, 2)) if total_assets > 0 else 0.0,
#                             key=f"{cat}_asset_weight_{asset}_{i}"
#                         )
#                     # Check weights sum to 1 within category
#                     if abs(sum(asset_weights[cat].values()) - 1.0) > 0.01:
#                         st.error(f"Weights for {cat} must sum to 1.")

#             # Calculate final weights
#             weights = {}
#             asset_only_weights = {}
#             for cat in asset_weights:
#                 for asset, w in asset_weights[cat].items():
#                     weights[asset] = category_weights[cat] * w
#                     asset_only_weights[asset] = category_weights[cat] * w
#             # Sum the weights for the indices if there are duplicates
#             for asset in selected_assets:
#                 index = ASSETS_INDICES_MAP[asset]['index']
#                 if index in weights:
#                     weights[index] += weights[asset]
#                 else:
#                     weights[index] = weights[asset]

#             mp['weights'] = weights
#             mp['asset_only_weights'] = asset_only_weights

#             # Optional: check if weights sum to 1
#             if abs(sum(asset_only_weights.values()) - 1.0) > 0.01:
#                 st.error("The asset weights must add up to 1. Please adjust the weights.")


#             # Show weights pie chart
#             show_weights(asset_only_weights, key=f"weights_chart_{i}")

#             # Choose to use OGC or not
#             ogc_option = st.selectbox(
#                 f"OGC setting for Portfolio {i+1}",
#                 options=["Standard OGC", "Evisens OGC (if available)", "No OGC (OGC = 0)"],
#                 index=0,
#                 key=f"ogc_option_{i}"
#             )

#             #choose specific allocation limit   
#             allocation_limit = st.number_input(
#                 f"Allocation limit for Portfolio {i+1} (Plus/minus %)", 
#                 min_value=0, max_value=100, value=50, key=f"allocation_limit_{i}"
#             )


#     #Start investment amount
#     start_investment = st.number_input("Start investment amount (SEK)", min_value=0, value=100000)
#     st.session_state['start_investment'] = start_investment

#     # Start and end date
#     start_date = st.date_input("Start date", datetime(2022, 1, 1), min_value=datetime(2005, 1, 1))  
#     end_date = st.date_input("End date", datetime.today(), min_value=datetime(2005, 1, 1))
#     st.session_state['start_date'] = start_date
#     st.session_state['end_date'] = end_date

#     #allocation limit
#     # allocation_limit = st.number_input("Allocation limit (Plus/minus %)", min_value=0, max_value=100, value=7)
#     # st.session_state['allocation_limit'] = allocation_limit


#     # Save back to session state
#     st.session_state['multi_portfolios'] = st.session_state['multi_portfolios']


#     if st.button("Calculate & Compare"):
#         st.session_state['page'] = 5  # You can add a new stage for results/plotting
#         st.rerun()

#     st.markdown("---")
#     if st.button("Back"):
#         st.session_state['page'] = 1
#         st.rerun()
#     show_footer()

def show_stage_4():
    st.logo("logo.png")
    num_portfolios = st.session_state.get('num_portfolios', 2)
    st.title("Compare Multiple Asset Portfolios")
    categories, display_name_to_asset_id = get_categorized_assets(ASSETS_INDICES_MAP)
    use_default = st.session_state.get('use_default', False)
    
    # --- Load defaults if requested ---
    default_category_weights_list = [{} for _ in range(num_portfolios)]
    default_asset_weights_list = [{} for _ in range(num_portfolios)]
    default_selected_shares = []
    default_selected_alternative = []
    default_selected_interest_bearing = []

    # Handle preset for 3 portfolios or CSV file
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
        
        # Collect all unique assets from CSV
        all_csv_assets = df['display_name'].unique()
        for display_name in all_csv_assets:
            # Find the category for this display name
            matching_asset = df[df['display_name'] == display_name].iloc[0]
            category = matching_asset['category']
            if category == "Equity":
                default_selected_shares.append(display_name)
            elif category == "Alternative":
                default_selected_alternative.append(display_name)
            elif category == "Interest Bearing":
                default_selected_interest_bearing.append(display_name)

    # Show preset info if using preset
    if use_default and st.session_state.get('portfolio_file'):
        st.info(f"""
        **Using preset from: {st.session_state['portfolio_file']}**
        
        You can modify the asset selections and weights below.
        """)

    # --- Initialize session state for multi_portfolios if not already ---
    if 'multi_portfolios' not in st.session_state or len(st.session_state['multi_portfolios']) != num_portfolios:
        st.session_state['multi_portfolios'] = [
            {'selected_shares': [], 'selected_alternative': [], 'selected_interest_bearing': [], 'weights': {}, 'asset_only_weights': {}, 'selected_assets': []}
            for _ in range(num_portfolios)
        ]

    # Layout: Asset selection on the left, portfolio weights on the right
    col_assets, col_portfolios = st.columns([1, 2])
    
    with col_assets:
        st.markdown("### Select Assets")
        
        # Asset selection (common for all portfolios)
        selected_shares = st.multiselect(
            "Select Equity Assets:", 
            categories["Equity"], 
            default=default_selected_shares if use_default else [],
            key="assets_shares"
        )
        selected_alternative = st.multiselect(
            "Select Alternative Assets:", 
            categories["Alternative"], 
            default=default_selected_alternative if use_default else [],
            key="assets_alt"
        )
        selected_interest_bearing = st.multiselect(
            "Select Interest Bearing Assets:", 
            categories["Interest Bearing"], 
            default=default_selected_interest_bearing if use_default else [],
            key="assets_int"
        )
        
        # Combine all selected assets
        all_selected_assets = selected_shares + selected_alternative + selected_interest_bearing
        selected_assets = [display_name_to_asset_id[name] for name in all_selected_assets]
        
        if not all_selected_assets:
            st.warning("⚠️ Please select at least one asset to continue.")
            
    with col_portfolios:
        st.markdown("### Portfolio Weights")
        
        if all_selected_assets:
            # Create portfolio columns
            portfolio_cols = st.columns(num_portfolios)
            
            for i, pcol in enumerate(portfolio_cols):
                with pcol:
                    st.markdown(f"#### Portfolio {i+1}")
                    
                    # Group selected assets by category
                    assets_by_category = {'Equity': selected_shares, 'Interest Bearing': selected_interest_bearing, 'Alternative': selected_alternative}
                    selected_cats = [c for c in ['Equity', 'Interest Bearing', 'Alternative'] if assets_by_category[c]]

                    # STEP 1: Set Category Weights
                    st.markdown("**Category Weights:**")
                    category_weights = {}
                    default_cat_weights = default_category_weights_list[i] if i < len(default_category_weights_list) else {}
                    
                    for cat in selected_cats:
                        category_weights[cat] = st.number_input(
                            f"{cat}", 
                            min_value=0.0, 
                            max_value=1.0,
                            value=default_cat_weights.get(cat, 0.0),
                            step=0.01,
                            key=f"cat_weight_{cat}_{i}"
                        )
                    
                    # Check if category weights sum to 1
                    cat_total = sum(category_weights.values())
                    if cat_total > 0:
                        if abs(cat_total - 1.0) > 0.01:
                            st.error(f"⚠️ Category weights sum to {cat_total:.3f}, should be 1.0")
                        else:
                            st.success(f"✅ Category weights sum to {cat_total:.3f}")

                    # STEP 2: Set Asset Weights within each Category
                    asset_weights = {}
                    default_asset_weights = default_asset_weights_list[i] if i < len(default_asset_weights_list) else {}
                    
                    for cat in selected_cats:
                        if assets_by_category[cat]:
                            st.markdown(f"**{cat} - Asset Weights:**")
                            total_assets = len(assets_by_category[cat])
                            asset_weights[cat] = {}
                            
                            for display_name in assets_by_category[cat]:
                                asset_id = display_name_to_asset_id[display_name]
                                
                                # Get default weight within category - DEFAULT TO 0.0
                                default_weight_in_cat = 0.0
                                if cat in default_asset_weights and asset_id in default_asset_weights[cat]:
                                    default_weight_in_cat = float(default_asset_weights[cat][asset_id])
                                
                                asset_weights[cat][asset_id] = st.number_input(
                                    f"{display_name}",
                                    min_value=0.0, 
                                    max_value=1.0,
                                    value=default_weight_in_cat,
                                    step=0.01,
                                    key=f"{cat}_asset_weight_{asset_id}_{i}",
                                    help=f"Weight within {cat} category"
                                )
                            
                            # Check if asset weights within category sum to 1
                            cat_asset_total = sum(asset_weights[cat].values())
                            if cat_asset_total > 0:
                                if abs(cat_asset_total - 1.0) > 0.01:
                                    st.error(f"⚠️ {cat} asset weights sum to {cat_asset_total:.3f}, should be 1.0")
                                else:
                                    st.success(f"✅ {cat} weights: {cat_asset_total:.3f}")
                            elif len(asset_weights[cat]) > 0:
                                st.warning(f"ℹ️ {cat} has assets selected but no weights assigned")

                    # STEP 3: Calculate Final Portfolio Weights
                    final_weights = {}
                    asset_only_weights = {}
                    
                    for cat in asset_weights:
                        for asset_id, asset_weight_in_cat in asset_weights[cat].items():
                            final_weight = category_weights[cat] * asset_weight_in_cat
                            final_weights[asset_id] = final_weight
                            asset_only_weights[asset_id] = final_weight
                    
                    # Sum the weights for the indices if there are duplicates
                    weights_with_indices = final_weights.copy()
                    for asset_id in asset_only_weights:
                        index = ASSETS_INDICES_MAP[asset_id]['index']
                        if index in weights_with_indices:
                            weights_with_indices[index] += final_weights[asset_id]
                        else:
                            weights_with_indices[index] = final_weights[asset_id]

                    # Update session state for this portfolio
                    mp = st.session_state['multi_portfolios'][i]
                    mp['selected_shares'] = selected_shares
                    mp['selected_alternative'] = selected_alternative  
                    mp['selected_interest_bearing'] = selected_interest_bearing
                    mp['selected_assets'] = selected_assets
                    mp['weights'] = weights_with_indices.copy()
                    mp['asset_only_weights'] = asset_only_weights.copy()
                    
                    # Show final weights chart
                    if asset_only_weights:
                        st.markdown("**Final Portfolio Weights:**")
                        show_weights(asset_only_weights, key=f"weights_chart_{i}")
                        
                        # Show total portfolio weight
                        total_portfolio_weight = sum(asset_only_weights.values())
                        if abs(total_portfolio_weight - 1.0) > 0.01:
                            st.error(f"⚠️ Total portfolio weight: {total_portfolio_weight:.3f}, should be 1.0")
                        else:
                            st.success(f"✅ Total portfolio weight: {total_portfolio_weight:.3f}")
                    
                    # Settings for each portfolio
                    st.markdown("**Settings:**")
                    
                    # OGC option for each portfolio
                    ogc_option = st.selectbox(
                        f"OGC setting",
                        options=["Standard OGC", "Evisens OGC (if available)", "No OGC"],
                        index=0,
                        key=f"ogc_option_{i}",
                        help=f"OGC setting for Portfolio {i+1}"
                    )
                    
                    # Allocation limit for each portfolio
                    allocation_limit = st.number_input(
                        f"Allocation limit ±%", 
                        min_value=0, max_value=100, value=50, 
                        key=f"allocation_limit_{i}",
                        help=f"Rebalancing trigger for Portfolio {i+1}"
                    )

    # Common settings at the bottom
    st.markdown("---")
    st.markdown("### Common Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        start_investment = st.number_input("Start investment amount (SEK)", min_value=0, value=100000)
        st.session_state['start_investment'] = start_investment
        
        start_date = st.date_input("Start date", datetime(2022, 1, 1), min_value=datetime(2005, 1, 1))
        st.session_state['start_date'] = start_date
        
    with col2:
        end_date = st.date_input("End date", datetime.today(), min_value=datetime(2005, 1, 1))
        st.session_state['end_date'] = end_date

    # Calculate button
    if all_selected_assets:
        if st.button("Calculate & Compare", type="primary"):
            # Validate that all portfolios have weights
            valid_portfolios = 0
            for i in range(num_portfolios):
                if st.session_state['multi_portfolios'][i]['asset_only_weights']:
                    total_weight = sum(st.session_state['multi_portfolios'][i]['asset_only_weights'].values())
                    if abs(total_weight - 1.0) <= 0.01:
                        valid_portfolios += 1
            
            if valid_portfolios > 0:
                st.session_state['page'] = 5
                st.rerun()
            else:
                st.error("⚠️ Please ensure at least one portfolio has valid weights (summing to 1.0).")
    else:
        st.button("Calculate & Compare", disabled=True, help="Please select assets first")

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

def show_stage_7():
    st.logo("logo.png")
    num_portfolios = st.session_state.get('num_portfolios', 2)
    st.title("Compare Multiple Index Portfolios")
    categories, display_name_to_asset_id = get_categorized_indices(ASSETS_INDICES_MAP)
    use_default = st.session_state.get('use_default', False)
    
    # --- Load defaults if requested ---
    default_category_weights_list = [{} for _ in range(num_portfolios)]
    default_asset_weights_list = [{} for _ in range(num_portfolios)]
    default_selected_shares = []
    default_selected_alternative = []
    default_selected_interest_bearing = []

    # Handle preset for 3 portfolios or CSV file
    if use_default:
        if num_portfolios == 3 and not st.session_state.get('portfolio_file'):
            # 3 Portfolio Preset for indices
            # Conservative Portfolio (Portfolio 1)
            default_category_weights_list[0] = {"Interest Bearing Index": 0.7, "Equity Index": 0.3}
            default_asset_weights_list[0] = {
                "Interest Bearing Index": {"SII:CSNRXSE": 1.0},  # 100% in one bond index
                "Equity Index": {"MSCI:892400NIUSD": 1.0}  # 100% in one equity index
            }
            
            # Balanced Portfolio (Portfolio 2)
            default_category_weights_list[1] = {"Interest Bearing Index": 0.4, "Equity Index": 0.4, "Alternative Index": 0.2}
            default_asset_weights_list[1] = {
                "Interest Bearing Index": {"SII:CSNRXSE": 1.0},
                "Equity Index": {"MSCI:892400NIUSD": 1.0},
                "Alternative Index": {"STATIC:Real Estate (USD)": 1.0}
            }
            
            # Aggressive Portfolio (Portfolio 3)
            default_category_weights_list[2] = {"Equity Index": 0.8, "Alternative Index": 0.2}
            default_asset_weights_list[2] = {
                "Equity Index": {"MSCI:892400NIUSD": 1.0},
                "Alternative Index": {"STATIC:Real Estate (USD)": 1.0}
            }
            
            # Collect all unique indices used in presets
            all_preset_indices = set()
            for weights_dict in default_asset_weights_list:
                for cat_dict in weights_dict.values():
                    all_preset_indices.update(cat_dict.keys())
            
            # Convert to display names and categorize
            for asset_id in all_preset_indices:
                display_name = ASSETS_INDICES_MAP[asset_id]["display name"]
                category = ASSETS_INDICES_MAP[asset_id]["category"]
                if category == "Equity Index":
                    default_selected_shares.append(display_name)
                elif category == "Alternative Index":
                    default_selected_alternative.append(display_name)
                elif category == "Interest Bearing Index":
                    default_selected_interest_bearing.append(display_name)
            
        elif st.session_state.get('portfolio_file'):
            # Load from CSV file
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
            
            # Collect all unique indices from CSV
            all_csv_indices = df['display_name'].unique()
            for display_name in all_csv_indices:
                # Find the category for this display name
                matching_asset = df[df['display_name'] == display_name].iloc[0]
                category = matching_asset['category']
                if category == "Equity Index":
                    default_selected_shares.append(display_name)
                elif category == "Alternative Index":
                    default_selected_alternative.append(display_name)
                elif category == "Interest Bearing Index":
                    default_selected_interest_bearing.append(display_name)

    # Show preset info if using 3 portfolio preset
    if use_default and num_portfolios == 3 and not st.session_state.get('portfolio_file'):
        st.info("""
        **3 Portfolio Index Preset:**
        - **Portfolio 1 (Conservative)**: 70% Interest Bearing + 30% Equity
        - **Portfolio 2 (Balanced)**: 40% Interest Bearing + 40% Equity + 20% Alternative  
        - **Portfolio 3 (Aggressive)**: 80% Equity + 20% Alternative
        
        You can modify these selections below.
        """)

    # --- Initialize session state for multi_portfolios if not already ---
    if 'multi_portfolios' not in st.session_state or len(st.session_state['multi_portfolios']) != num_portfolios:
        st.session_state['multi_portfolios'] = [
            {'selected_shares': [], 'selected_alternative': [], 'selected_interest_bearing': [], 'weights': {}, 'asset_only_weights': {}, 'selected_assets': []}
            for _ in range(num_portfolios)
        ]

    # Layout: Index selection on the left, portfolio weights on the right
    col_indices, col_portfolios = st.columns([1, 2])
    
    with col_indices:
        st.markdown("### Select Indices")
        
        # Index selection (common for all portfolios)
        selected_shares = st.multiselect(
            "Select Equity Indices:", 
            categories["Equity Index"], 
            default=default_selected_shares if use_default else [],
            key="indices_shares"
        )
        selected_alternative = st.multiselect(
            "Select Alternative Indices:", 
            categories["Alternative Index"], 
            default=default_selected_alternative if use_default else [],
            key="indices_alt"
        )
        selected_interest_bearing = st.multiselect(
            "Select Interest Bearing Indices:", 
            categories["Interest Bearing Index"], 
            default=default_selected_interest_bearing if use_default else [],
            key="indices_int"
        )
        
        # Combine all selected indices
        all_selected_indices = selected_shares + selected_alternative + selected_interest_bearing
        selected_assets = [display_name_to_asset_id[name] for name in all_selected_indices]
        
        if not all_selected_indices:
            st.warning("⚠️ Please select at least one index to continue.")
            
    with col_portfolios:
        st.markdown("### Portfolio Weights")
        
        if all_selected_indices:
            # Create portfolio columns
            portfolio_cols = st.columns(num_portfolios)
            
            for i, pcol in enumerate(portfolio_cols):
                with pcol:
                    # Add portfolio type label for preset
                    if use_default and num_portfolios == 3 and not st.session_state.get('portfolio_file'):
                        portfolio_types = ["Conservative", "Balanced", "Aggressive"]
                        st.markdown(f"#### Portfolio {i+1}")
                        st.markdown(f"*({portfolio_types[i]})*")
                    else:
                        st.markdown(f"#### Portfolio {i+1}")
                    
                    # Group selected assets by category
                    assets_by_category = {'Equity Index': selected_shares, 'Interest Bearing Index': selected_interest_bearing, 'Alternative Index': selected_alternative}
                    selected_cats = [c for c in ['Equity Index', 'Interest Bearing Index', 'Alternative Index'] if assets_by_category[c]]

                    # STEP 1: Set Category Weights
                    st.markdown("**Category Weights:**")
                    category_weights = {}
                    default_cat_weights = default_category_weights_list[i] if i < len(default_category_weights_list) else {}
                    
                    for cat in selected_cats:
                        

                        category_weights[cat] = st.number_input(
                            f"{cat}", 
                            min_value=0.0, 
                            max_value=1.0,
                            value=default_cat_weights.get(cat, 0.0),
                            step=0.01,
                            key=f"cat_weight_{cat}_{i}"
                        )
                    
                    # Check if category weights sum to 1
                    cat_total = sum(category_weights.values())
                    if cat_total > 0:
                        if abs(cat_total - 1.0) > 0.01:
                            st.error(f"⚠️ Category weights sum to {cat_total:.3f}, should be 1.0")
                        else:
                            st.success(f"✅ Category weights sum to {cat_total:.3f}")

                    # STEP 2: Set Asset Weights within each Category
                    asset_weights = {}
                    default_asset_weights = default_asset_weights_list[i] if i < len(default_asset_weights_list) else {}
                    
                    for cat in selected_cats:
                        if assets_by_category[cat]:
                            st.markdown(f"**{cat} - Asset Weights:**")
                            total_assets = len(assets_by_category[cat])
                            asset_weights[cat] = {}
                            
                            for display_name in assets_by_category[cat]:
                                asset_id = display_name_to_asset_id[display_name]
                                
                                # Get default weight within category
                                default_weight_in_cat = 0.0
                                if cat in default_asset_weights and asset_id in default_asset_weights[cat]:
                                    default_weight_in_cat = default_asset_weights[cat][asset_id]
                                elif total_assets > 0:
                                    default_weight_in_cat = 0.0
                                
                                asset_weights[cat][asset_id] = st.number_input(
                                    f"{display_name}",
                                    min_value=0.0, 
                                    max_value=1.0,
                                    value=default_weight_in_cat,
                                    step=0.01,
                                    key=f"{cat}_asset_weight_{asset_id}_{i}",
                                    help=f"Weight within {cat} category"
                                )
                            
                            # Check if asset weights within category sum to 1
                            cat_asset_total = sum(asset_weights[cat].values())
                            if cat_asset_total > 0:
                                if abs(cat_asset_total - 1.0) > 0.01:
                                    st.error(f"⚠️ {cat} asset weights sum to {cat_asset_total:.3f}, should be 1.0")
                                else:
                                    st.success(f"✅ {cat} weights: {cat_asset_total:.3f}")

                    # STEP 3: Calculate Final Portfolio Weights
                    final_weights = {}
                    asset_only_weights = {}
                    
                    for cat in asset_weights:
                        for asset_id, asset_weight_in_cat in asset_weights[cat].items():
                            final_weight = category_weights[cat] * asset_weight_in_cat
                            final_weights[asset_id] = final_weight
                            asset_only_weights[asset_id] = final_weight

                    # Update session state for this portfolio
                    mp = st.session_state['multi_portfolios'][i]
                    mp['selected_shares'] = selected_shares
                    mp['selected_alternative'] = selected_alternative  
                    mp['selected_interest_bearing'] = selected_interest_bearing
                    mp['selected_assets'] = selected_assets
                    mp['weights'] = final_weights.copy()
                    mp['asset_only_weights'] = asset_only_weights.copy()
                    
                    # Show final weights chart
                    if asset_only_weights:
                        st.markdown("**Final Portfolio Weights:**")
                        show_weights(asset_only_weights, key=f"weights_chart_{i}")
                        
                        # Show total portfolio weight
                        total_portfolio_weight = sum(asset_only_weights.values())
                        if abs(total_portfolio_weight - 1.0) > 0.01:
                            st.error(f"⚠️ Total portfolio weight: {total_portfolio_weight:.3f}, should be 1.0")
                        else:
                            st.success(f"✅ Total portfolio weight: {total_portfolio_weight:.3f}")
                    
                    # Allocation limit for each portfolio
                    st.markdown("**Settings:**")
                    allocation_limit = st.number_input(
                        f"Allocation limit ±%", 
                        min_value=0, max_value=100, value=50, 
                        key=f"allocation_limit_{i}",
                        help=f"Rebalancing trigger for Portfolio {i+1}"
                    )

    # Common settings at the bottom
    st.markdown("---")
    st.markdown("### Common Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        start_investment = st.number_input("Start investment amount (SEK)", min_value=0, value=100000)
        st.session_state['start_investment'] = start_investment
        
        start_date = st.date_input("Start date", datetime(2022, 1, 1), min_value=datetime(2005, 1, 1))
        st.session_state['start_date'] = start_date
        
    with col2:
        end_date = st.date_input("End date", datetime.today(), min_value=datetime(2005, 1, 1))
        st.session_state['end_date'] = end_date

    # Calculate button
    if all_selected_indices:
        if st.button("Calculate & Compare", type="primary"):
            # Validate that all portfolios have weights
            valid_portfolios = 0
            for i in range(num_portfolios):
                if st.session_state['multi_portfolios'][i]['asset_only_weights']:
                    valid_portfolios += 1
            
            if valid_portfolios > 0:
                st.session_state['page'] = 8
                st.rerun()
            else:
                st.error("⚠️ Please set weights for at least one portfolio.")
    else:
        st.button("Calculate & Compare", disabled=True, help="Please select indices first")

    st.markdown("---")
    if st.button("Back"):
        st.session_state['page'] = 1
        st.rerun()
    show_footer()

def show_stage_8():
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
        ogc_option = st.session_state.get(f'ogc_option_{i}', "No OGC (OGC = 0)")
        
        if not selected_assets or not weights:
            st.warning(f"Portfolio {i+1} is incomplete. Please go back and select assets.")
            continue

        combined_data = fetch_data_infront(selected_assets, selected_assets, start_date, end_date)
        combined_data = combined_data[combined_data['Type'] != 'Index']
        # Remove duplicate rows with the same date and name, keeping the first occurrence
       

        combined_data, period, data_frequency = clean_data(combined_data, data_frequency, True)
        combined_data = indexed_net_to_100(combined_data)
        combined_data = period_change(combined_data)
        combined_data = OGC_adjusted_Period_Change(combined_data, period, ogc_option)
        combined_data = indexed_OGC_adjusted_to_100(combined_data)
        
        # Filter out data points of type 'Index'
        

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
    generate_multi_summary_report_indices(finished_portfolios, allocation_limit)
    st.markdown("---")
    if st.button("Back"):
        st.session_state['page'] = 1
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
    elif st.session_state['page'] == 7:
        show_stage_7()
    elif st.session_state['page'] == 8:
        show_stage_8()

    elif st.session_state['page'] == 99:
        show_asset_indices_admin()

if __name__ == "__main__":
    main()