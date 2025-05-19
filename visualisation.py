import pandas as pd
import numpy as np
import streamlit as st
import pdfkit

import plotly.express as px

from portfolio import ASSETS_INDICES_MAP, PERIOD

# You may need to configure pdfkit with the path to wkhtmltopdf
config = pdfkit.configuration()  # or pdfkit.configuration(wkhtmltopdf='/path/to/wkhtmltopdf')

# Make sure ASSETS_INDICES_MAP and PERIOD are defined elsewhere in your code
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

def calculate_sharpe_ratio(returns, risk_free_rate=0.01):
    """
    Calculate the Sharpe Ratio for a given series of returns.
    """
    excess_returns = returns - risk_free_rate / PERIOD
    annualized_excess_returns = excess_returns.mean() * PERIOD
    annualized_volatility = excess_returns.std() * np.sqrt(PERIOD)
    sharpe_ratio = annualized_excess_returns / annualized_volatility
    return sharpe_ratio

def export_report_to_pdf(combined_data, date_holdings_df, start_investment, allocation_limit, weights, sharpe_ratio):
    """
    Export the summary report to a PDF file.
    """
    st.write("Exporting the report to PDF...")
    # Generate HTML content for the report
    html_content = f"""
    <html>
    <head>
        <title>Summary Report</title>
    </head>
    <body>
        <h1>Summary Report</h1>
        <h2>Key Metrics</h2>
        <p>Start Investment Amount: {start_investment} SEK</p>
        <p>Allocation Limit: {allocation_limit}%</p>
        <p>Sharpe Ratio: {sharpe_ratio:.2f}</p>
        <h2>Asset Weights</h2>
        <ul>
    """
    for asset, weight in weights.items():
        display_name = ASSETS_INDICES_MAP[asset].get("display name", asset)
        html_content += f"<li>{display_name}: {weight:.2f}</li>"
    html_content += """
        </ul>
        <h2>Portfolio Data</h2>
        {combined_data.to_html()}
        <h2>Date vs Total Holdings Data</h2>
        {date_holdings_df.to_html()}
    </body>
    </html>
    """

     # Convert HTML content to PDF
    try:
        pdfkit.from_string(html_content, 'summary_report.pdf', configuration=config)
        st.success("Report exported to summary_report.pdf")
    except Exception as e:
        st.error(f"Error exporting report to PDF: {e}")

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


def generate_summary_report(combined_data, date_holdings_df, start_investment, allocation_limit, weights):
    """
    Generate a summary report for the portfolio.
    """
    st.header("Summary Report")

    # Display key metrics
    st.subheader("Key Metrics")
    st.metric(label="Start Investment Amount", value=f"{start_investment} SEK")
    st.metric(label="Allocation Limit", value=f"{allocation_limit}%")

    # Calculate and display Sharpe Ratio
    portfolio_returns = combined_data[combined_data['Type'] == 'Asset']['OGC Adjusted Period Change']
    sharpe_ratio = calculate_sharpe_ratio(portfolio_returns)
    st.metric(label="Sharpe Ratio", value=f"{sharpe_ratio:.2f}")

    # Display weights
    st.subheader("Asset Weights")
    for asset, weight in weights.items():
        display_name = ASSETS_INDICES_MAP[asset].get("display name", asset)
        st.write(f"{display_name}: {weight:.2f}")

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

    # Display portfolio data
    st.subheader("Portfolio Data")
    st.write(combined_data)

    # Display date vs total holdings data
    st.subheader("Date vs Total Holdings Data")
    st.write(date_holdings_df)

    # button to export the report to PDF or Excel
    if st.button("Export Report to PDF"):
        export_report_to_pdf(combined_data, date_holdings_df, start_investment, allocation_limit, weights, sharpe_ratio)

    if st.button("Export Report to Excel"):
        export_report_to_excel(combined_data, date_holdings_df, start_investment, allocation_limit, weights, sharpe_ratio)

