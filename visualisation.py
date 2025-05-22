import pandas as pd
import numpy as np
import streamlit as st
import pdfkit
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio


# default colors

pio.templates["my_custom"] = pio.templates["plotly_white"]
pio.templates["my_custom"].layout.colorway = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA"]
pio.templates.default = "my_custom"

from portfolio import ASSETS_INDICES_MAP

RISK_FREE_RATE = 0.01  # Example risk-free rate, adjust as needed

# You may need to configure pdfkit with the path to wkhtmltopdf
config = pdfkit.configuration()  # or pdfkit.configuration(wkhtmltopdf='/path/to/wkhtmltopdf')

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

def plot_all_holdings(combined_data):
    """
    Create one plot for each asset and its corresponding index.
    Returns a dictionary mapping asset display names to Plotly figures.
    """
    # Create a mapping from asset display names to asset ID
    asset_display_name_to_id = {attributes["display name"]: asset for asset, attributes in ASSETS_INDICES_MAP.items()}
    asset_id_to_display_name = {asset: attributes["display name"] for asset, attributes in ASSETS_INDICES_MAP.items()}

    # Find all unique assets (Type == 'Asset')
    assets = combined_data[combined_data['Type'] == 'Asset']['Name'].unique()
    plots = {}

    for asset in assets:
        asset_id = asset_display_name_to_id.get(asset, asset)
        # Find the corresponding index for this asset
        index_id = ASSETS_INDICES_MAP.get(asset_id, {}).get("index")
        if not index_id:
            continue  # Skip if no index is mapped

        index = asset_id_to_display_name.get(index_id, index_id)

        # Filter data for this asset and its index
        asset_data = combined_data[(combined_data['Type'] == 'Asset') & (combined_data['Name'] == asset)]
        index_data = combined_data[(combined_data['Type'] == 'Index') & (combined_data['Name'] == index)]

        # Create the plot
        fig = px.line(title=f"{asset} vs {index} Holdings Over Time")
        if not asset_data.empty:
            fig.add_scatter(x=asset_data['date'], y=asset_data['Holdings'], mode='lines', name=asset)
        if not index_data.empty:
            fig.add_scatter(x=index_data['date'], y=index_data['Holdings'], mode='lines', name=index)

        fig.update_layout(
            xaxis_title='Date',
            yaxis_title='Holdings',
            legend_title='Type',
            template='plotly_white'
        )

        plots[asset] = fig

    return plots


def calculate_drawdowns(data, column, window=500):
    """
    Calculate drawdowns for a given column in the DataFrame using a rolling maximum.
    """
    data = data.copy()
    data['Max'] = data[column].cummax()
    data['Drawdown'] = (data[column] - data['Max']) / data['Max']
    return data

def show_weights(weights, key=None):
    """
    Display a pie chart of asset weights using Plotly and Streamlit.
    """
    asset_labels = [ASSETS_INDICES_MAP[asset].get("display name", asset) for asset in weights.keys()]
    values = list(weights.values())

    fig = px.pie(
        names=asset_labels,
        values=values,
        title="Allocation Weights",
        hole=0.3
    )
    fig.update_traces(textinfo='percent+label')
    st.plotly_chart(fig, key=key)

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

def calculate_sharpe_ratio(returns,period, risk_free_rate=RISK_FREE_RATE ):
    """
    Calculate the Sharpe Ratio for a given series of returns.
    """
    excess_returns = returns - risk_free_rate / period
    annualized_excess_returns = excess_returns.mean() * period
    annualized_volatility = excess_returns.std() * np.sqrt(period)
    sharpe_ratio = annualized_excess_returns / annualized_volatility
    return sharpe_ratio

def calculate_variance(returns):
    """
    Calculate the variance of a given series of returns.
    """
    return np.var(returns)

def calculate_annualized_return(returns, period):
    """
    Calculate the annualized return for a given series of returns.
    """
    return (1 + returns).prod() ** (period / len(returns)) - 1

def calculate_maximum_drawdown(returns):
    """
    Calculate the maximum drawdown for a given series of returns.
    """
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()
    return max_drawdown

def calculate_volatility(returns,period):
    """
    Calculate the volatility for a given series of returns.
    """
    return returns.std() * np.sqrt(period)

def export_report_to_excel(combined_data, date_holdings_df, start_investment, allocation_limit, weights, sharpe_ratio):
    """
    Export the summary report to an Excel file.
    """
    with pd.ExcelWriter('summary_report.xlsx') as writer:
        # Write key metrics to a sheet
        metrics_df = pd.DataFrame({
            "Metric": ["Start Investment Amount", "Allocation Limit", "Sharpe Ratio"],
            "Value": [f"{start_investment} SEK", f"{allocation_limit}%", f"{sharpe_ratio:.2f}"]
        })
        metrics_df.to_excel(writer, sheet_name='Key Metrics', index=False)

        # Write asset weights to a sheet
        weights_df = pd.DataFrame(list(weights.items()), columns=['Asset', 'Weight'])
        weights_df['Display Name'] = weights_df['Asset'].apply(lambda x: ASSETS_INDICES_MAP[x].get("display name", x))
        weights_df.to_excel(writer, sheet_name='Asset Weights', index=False)

        # Write portfolio data to a sheet
        combined_data.to_excel(writer, sheet_name='Portfolio Data', index=False)

        # Write date vs total holdings data to a sheet
        date_holdings_df.to_excel(writer, sheet_name='Date vs Total Holdings Data', index=False)

    st.success("Report exported to summary_report.xlsx")

def plot_multi_portfolio_total_holdings_assets(finished_portfolios):
    """
    Plot total holdings in assets over time for each portfolio.
    """
    fig = go.Figure()
    for name, data in finished_portfolios.items():
        df = data["date_holdings_df"]
        # Filter for assets only
        assets_df = df[df['Type'] == 'Asset']
        fig.add_trace(go.Scatter(
            x=assets_df['Date'],
            y=assets_df['Total Holdings'],
            mode='lines',
            name=name
        ))
    fig.update_layout(
        title="Total Holdings in Assets Over Time (All Portfolios)",
        xaxis_title='Date',
        yaxis_title='Total Holdings',
        legend_title='Portfolio',
        template='plotly_white'
    )
    return fig

def plot_multi_portfolio_total_holdings_indices(finished_portfolios):
    """
    Plot total holdings in indices over time for each portfolio.
    """
    fig = go.Figure()
    for name, data in finished_portfolios.items():
        df = data["date_holdings_df"]
        # Filter for indices only
        indices_df = df[df['Type'] == 'Index']
        fig.add_trace(go.Scatter(
            x=indices_df['Date'],
            y=indices_df['Total Holdings'],
            mode='lines',
            name=name
        ))
    fig.update_layout(
        title="Total Holdings in Indices Over Time (All Portfolios)",
        xaxis_title='Date',
        yaxis_title='Total Holdings',
        legend_title='Portfolio',
        template='plotly_white'
    )
    return fig

def plot_multi_portfolio_drawdowns_assets(finished_portfolios):
    """
    Plot drawdowns for assets only, for each portfolio.
    """
    fig = go.Figure()
    for name, data in finished_portfolios.items():
        df = data["date_holdings_df"]
        assets_df = df[df['Type'] == 'Asset']
        if not assets_df.empty:
            drawdown = calculate_drawdowns(assets_df, 'Total Holdings')
            fig.add_trace(go.Scatter(
                x=drawdown['Date'],
                y=drawdown['Drawdown'],
                mode='lines',
                name=name
            ))
    fig.update_layout(
        title="Drawdowns in Assets (All Portfolios)",
        xaxis_title="Date",
        yaxis_title="Drawdown",
        legend_title="Portfolio",
        template="plotly_white"
    )
    return fig

def plot_multi_portfolio_drawdowns_indices(finished_portfolios):
    """
    Plot drawdowns for indices only, for each portfolio.
    """
    fig = go.Figure()
    for name, data in finished_portfolios.items():
        df = data["date_holdings_df"]
        indices_df = df[df['Type'] == 'Index']
        if not indices_df.empty:
            drawdown = calculate_drawdowns(indices_df, 'Total Holdings')
            fig.add_trace(go.Scatter(
                x=drawdown['Date'],
                y=drawdown['Drawdown'],
                mode='lines',
                name=name
            ))
    fig.update_layout(
        title="Drawdowns in Indices (All Portfolios)",
        xaxis_title="Date",
        yaxis_title="Drawdown",
        legend_title="Portfolio",
        template="plotly_white"
    )
    return fig

def generate_summary_report(combined_data, date_holdings_df, start_investment, allocation_limit, weights, asset_weights, period):

    """
    Generate a summary report for the portfolio.
    """
    st.header("Summary Report")

    st.subheader("Key Metrics")
    st.write("This section provides key metrics for the portfolio. Variance and Sharpe Ratio are calculated based on the portfolio returns after ongoing charge (OGC).")
    st.write("The Sharpe Ratio is a measure of risk-adjusted return, while variance indicates the volatility of the portfolio.")
    st.write("The Sharpe Ratio is calculated using a risk-free rate of " + str(RISK_FREE_RATE) + " per year.")
    st.write("Period value " + str(period) + " is used to annualize the returns and volatility.")
    portfolio_returns = combined_data[combined_data['Type'] == 'Asset']['OGC Adjusted Period Change']
    sharpe_ratio = calculate_sharpe_ratio(portfolio_returns, period)
    variance = calculate_variance(portfolio_returns)
    max_drawdown = calculate_maximum_drawdown(portfolio_returns)
    volatility = calculate_volatility(portfolio_returns, period)
    annualized_return = calculate_annualized_return(portfolio_returns, period)
    metrics_df = pd.DataFrame({
        "Metric": [
            "Start Investment Amount",
            "Allocation Limit",
            "Sharpe Ratio",
            "Variance",
            "Max Drawdown",
            "Volatility",
            "Annualized Return"
        ],
        "Value": [
            f"{start_investment} SEK",
            f"{allocation_limit}%",
            f"{sharpe_ratio:.2f}",
            f"{variance * 100:.2f}%",
            f"{max_drawdown * 100:.2f}%",
            f"{volatility * 100:.2f}%",
            f"{annualized_return * 100:.2f}%"
        ]
    })
    st.table(metrics_df)

    # Display weights
    st.subheader("Asset Weights")
    for asset, weight in weights.items():
        display_name = ASSETS_INDICES_MAP[asset].get("display name", asset)
        st.write(f"{display_name}: {weight:.2f}")

    # Show weights pie chart
    st.subheader("Asset Allocation Weights")     
    show_weights(asset_weights)

    # Plot the holdings
    st.subheader("Holdings Over Time")
    fig_holdings = plot_holdings(combined_data)
    st.plotly_chart(fig_holdings)

    # Plot the date vs total holdings
    st.subheader("Total Holdings Over Time")
    fig_total_holdings = plot_date_vs_total_holdings(date_holdings_df)
    st.plotly_chart(fig_total_holdings)

    # Plot the drawdowns
    st.subheader("Drawdowns")
    index_data = date_holdings_df[date_holdings_df['Type'] == 'Index']
    portfolio_data = date_holdings_df[date_holdings_df['Type'] == 'Asset']
    fig_drawdowns = plot_drawdowns(portfolio_data, index_data)
    st.plotly_chart(fig_drawdowns)

    # Plot all holdings
    #st.subheader("All Holdings")    
    #all_holdings_plots = plot_all_holdings(combined_data)
    #for asset_display_name, fig in all_holdings_plots.items():
    #    st.markdown(f"**{asset_display_name}**")
    #    st.plotly_chart(fig)


    # Replace asset IDs with display names in 'Name' column of combined_data
    combined_data['Name'] = combined_data['Name'].map(lambda x: ASSETS_INDICES_MAP[x]["display name"] if x in ASSETS_INDICES_MAP else x)
   


    # Display portfolio data
    st.subheader("Portfolio Data")
    st.write(combined_data)

    # Display date vs total holdings data
    st.subheader("Date vs Total Holdings Data")
    st.write(date_holdings_df)
    # Export report to Excel
    if st.button("Export Report to Excel"):
        export_report_to_excel(combined_data, date_holdings_df, start_investment, allocation_limit, weights, sharpe_ratio)


def generate_multi_summary_report(finished_portfolios):
    """
    Display a summary report comparing multiple portfolios.
    finished_portfolios: dict of {portfolio_name: {combined_data, date_holdings_df, weights, asset_only_weights, period}}
    """
    st.header("Multi-Portfolio Comparison")

    # Show a table of key metrics for each portfolio
    metrics_list = []
    for name, data in finished_portfolios.items():
        df = data["date_holdings_df"]
        period = data["period"]
        returns = df['Total Holdings'].pct_change().dropna()
        sharpe = calculate_sharpe_ratio(returns, period)
        max_dd = calculate_maximum_drawdown(returns)
        volatility = calculate_volatility(returns, period)
        variance = calculate_variance(returns)
        ann_return = calculate_annualized_return(returns, period)

        metrics_list.append({
            "Portfolio": name,
            "Sharpe Ratio": f"{sharpe:.2f}",
            "Variance": f"{variance:.2%}",
            "Standard deviation": f"{volatility:.2%}",
            "Max Drawdown": f"{max_dd:.2%}",
            "Annualized Return": f"{ann_return:.2%}"
        })
    if metrics_list:
        metrics_df = pd.DataFrame(metrics_list).set_index('Portfolio')
        st.subheader("Key Metrics Comparison")
        st.write("This section provides key metrics for each portfolio. Variance and Sharpe Ratio are calculated based on the portfolio returns after ongoing charge (OGC).")
        st.write("The Sharpe Ratio is a measure of risk-adjusted return, while variance indicates the volatility of the portfolio.")
        st.write("The Sharpe Ratio is calculated using a risk-free rate of " + str(RISK_FREE_RATE) + " per year.")
        st.write("Period value " + str(period) + " is used to annualize the returns and volatility.")
        st.write("The table below shows the key metrics for each portfolio.")
        st.table(metrics_df)

    # Plot all portfolios on the same chart
    st.subheader("Portfolio Value Comparison")
    st.write("This section shows the total holdings for each portfolio over time.")

    fig_portfolio = plot_multi_portfolio_total_holdings_assets(finished_portfolios)
    st.plotly_chart(fig_portfolio)

        # Plot drawdowns for all portfolios
    st.subheader("Drawdowns Comparison")
    st.write("This section shows the drawdowns for each portfolio over time.")
    st.write("The drawdown is calculated as the percentage drop from the maximum value.")
    fig_drawdowns_assets = plot_multi_portfolio_drawdowns_assets(finished_portfolios)
    st.plotly_chart(fig_drawdowns_assets)

    # Show the weights for each portfolio
    for name, data in finished_portfolios.items():
        st.subheader(f"Portfolio: {name}")
        asset_only_weights = data["asset_only_weights"]
        st.markdown("### Asset Weights:")
        show_weights(asset_only_weights, key=name + "_asset_only")

        # Show index weights for each portfolio
        # Remove assets in asset_only_weights from weights to get only indices
        weights = data["weights"].copy()
        asset_only_weights = data["asset_only_weights"]
        index_only_weights = {k: v for k, v in weights.items() if k not in asset_only_weights}
        st.markdown("### Index Weights:")
        show_weights(index_only_weights, key=name + "_index_only")

    

    # Plot portfolios vs their respctive indices
    st.subheader("Portfolio vs Index Comparison")   
    st.write("This section shows the total holdings for each portfolio and its corresponding index over time.")
    for name, data in finished_portfolios.items():
        df = data["date_holdings_df"]
        # Plot both assets and indices for each portfolio in the same plot
        st.markdown("#### Assets vs Indices for Portfolio: " + name)
    
        st.plotly_chart(plot_date_vs_total_holdings(df), key=name + "_assets_vs_indices")
    
    

    # Chnage the display names in the combined data
    for name, data in finished_portfolios.items():
        combined_data = data["combined_data"]
        combined_data['Name'] = combined_data['Name'].map(lambda x: ASSETS_INDICES_MAP[x]["display name"] if x in ASSETS_INDICES_MAP else x)
        data["combined_data"] = combined_data
    # Show the portfolio data for each portfolio
    for name, data in finished_portfolios.items():
        st.subheader(f"Portfolio Data: {name}")
        st.write(data["combined_data"])
    # export report to Excel
    if st.button("Export Multi-Portfolio Report to Excel"):
        with pd.ExcelWriter('multi_summary_report.xlsx') as writer:
            for name, data in finished_portfolios.items():
                combined_data = data["combined_data"]
                date_holdings_df = data["date_holdings_df"]
                combined_data.to_excel(writer, sheet_name=f'{name}_Portfolio_Data', index=False)
                date_holdings_df.to_excel(writer, sheet_name=f'{name}_Date_vs_Total_Holdings', index=False)
        st.success("Multi-Portfolio Report exported to multi_summary_report.xlsx")