import pandas as pd
import numpy as np
import streamlit as st
import pdfkit

import plotly.express as px

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

def show_weights(weights):
    """
    Display a pie chart of asset weights using Plotly and Streamlit.
    """
    # Map asset IDs to display names
    asset_labels = [ASSETS_INDICES_MAP[asset].get("display name", asset) for asset in weights.keys()]
    values = list(weights.values())

    fig = px.pie(
        names=asset_labels,
        values=values,
        title="Asset Allocation Weights",
        hole=0.3
    )
    fig.update_traces(textinfo='percent+label')
    st.plotly_chart(fig)

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

