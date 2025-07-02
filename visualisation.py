import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from future_simulations import monte_carlo_simulation
from InfrontConnect import infront
import io




TOTAL_HOLDINGS = 'Totalt innehav'
# TOTAL_HOLDINGS = 'Total Holdings'
DATE = 'Datum'
# DATE = 'Date'
RETURN = 'Avkastning'
# RETURN = 'Return'
ROLLING_AVG_RETURN = 'Rullande medelvärde av avkastning'
# ROLLING_AVG_RETURN = 'Rolling Average Return'

PORTFOLIO = 'Portfölj'
# PORTFOLIO = 'Portfolio'
ALLOCATION_WEIGHTS = 'Allokeringsvikter'
# ALLOCATION_WEIGHTS = 'Allocation Weights'
DATE_VS_TOTAL_HOLDINGS = 'Datum vs Totalt Innehav'
# DATE_VS_TOTAL_HOLDINGS = 'Date vs Total Holdings'
DRAWDOWN = 'Värdefall'
# DRAWDOWN = 'Drawdown'
HOLDINGS_IN_ASSETS_VS_INDICES = 'Innehav i tillgångar vs index'
# HOLDINGS_IN_ASSETS_VS_INDICES = 'Holdings in Assets vs Indices'
HOLDINGS = 'Innehav'
# HOLDINGS = 'Holdings'
ASSETS_AND_INDICES = 'Tillgångar och Index'
# ASSETS_AND_INDICES = 'Assets and Indices'
TYPE = 'Typ'
# TYPE = 'Type'
TOTAL_HOLDINGS_IN_ASSETS_OVER_TIME = 'Totalt innehav i tillgångar över tid (alla portföljer)'
# TOTAL_HOLDINGS_IN_ASSETS_OVER_TIME = 'Total Holdings in Assets Over Time (All Portfolios)'
TOTAL_HOLDINGS_IN_INDICES_OVER_TIME_ALL_PORTFOLIOS = 'Totalt innehav i index över tid (alla portföljer)'
# TOTAL_HOLDINGS_IN_INDICES_OVER_TIME_ALL_PORTFOLIOS = 'Total Holdings in Indices Over Time (All Portfolios)'
DRAWDOWNS_IN_ASSETS_ALL_PORTFOLIOS = 'Värdefall i tillgångar (alla portföljer)'
# DRAWDOWNS_IN_ASSETS_ALL_PORTFOLIOS = 'Drawdowns in Assets (All Portfolios)'
DRAWDOWN_PORTFOLIO_VS_INDEX = 'Värdefall portfölj vs index'
# DRAWDOWN_PORTFOLIO_VS_INDEX = 'Drawdown Portfolio vs Index'
DRAWDOWNS_IN_INDICES_ALL_PORTFOLIOS = 'Värdefall i index (alla portföljer)'
# DRAWDOWNS_IN_INDICES_ALL_PORTFOLIOS = 'Drawdowns in Indices (All Portfolios)'


# high contrast colors

HIGH_CONTRAST_COLORWAY = ["#1ABC9C", "#F06B4B", "#6A3A9C","#30405F", "#000000"]
# default colors
pio.templates["my_custom"] = pio.templates["simple_white"]
pio.templates["my_custom"].layout.colorway = [ "#1ABC9C","#7E8BA7", "#30405F",  "#000000", "#6A3A9C", "#F06B4B", "#4A69F1", "#CAA8F5", "#F5B700", "#FF6F61", "#2D3047", "#493657", "#8D1919", "#F0E68C", "#FF4500", "#2E8B57", "#4682B4", "#D2691E", "#8B008B"]
pio.templates["my_custom"].layout.paper_bgcolor = "rgba(0,0,0,0)" 
pio.templates["my_custom"].layout.plot_bgcolor = "rgba(0,0,0,0)"
pio.templates["my_custom"].layout.font = dict(
    family="Segoe UI, sans-serif",
    color="#404040",
    size=15
)

pio.templates["my_custom"].layout.xaxis = dict(
    showgrid=False,           # Remove vertical grid lines
    showline=True,            # Show axis line
    linecolor="#404040",      # Color for axis line (bottom line)
    linewidth=1
)
pio.templates["my_custom"].layout.yaxis = dict(
    showgrid=False,           # Remove horizontal grid lines
    showline=False            # No left/right axis line
)

pio.templates["my_custom"].layout.title = dict(
    x=0.5,  # Center the title
    xanchor='center'
)

# Set legend position for all custom plots
pio.templates["my_custom"].layout.legend = dict(
    orientation="h",
    x=0.5,
    xanchor="center",
    y=-0.2,
    yanchor="top"

)

pio.templates["my_custom"].data.scatter = [go.Scatter(line=dict(width=1.3))]
pio.templates["my_custom"].data.scattergl = [go.Scattergl(line=dict(width=1.3))]
pio.templates.default = "my_custom"

from portfolio import ASSETS_INDICES_MAP

RISK_FREE_RATE = 0.01  # Example risk-free rate, adjust as needed


def fetch_risk_free_rate(end_date, start_date, period='month'):
    history = infront.GetHistory(
        tickers=["NSx:OMRXTBILL"],
        fields=["last"],
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )
    df = next(iter(history.values()))  # or history["OMRXTBILL:2087"]
    df = df.reset_index()  # Moves 'date' from index to column
    df.rename(columns={"last": "OMRXTBILL"}, inplace=True)

    # Ensure the 'date' column is in datetime format
    df['date'] = pd.to_datetime(df['date'])

    # Sort the DataFrame by date
    df = df.sort_values('date')

    # Calculate the period return based on the specified period
    if period == 'month':
        df['Period Return'] = df['OMRXTBILL'].pct_change(periods=30)
    elif period == 'week':
        df['Period Return'] = df['OMRXTBILL'].pct_change(periods=7)
    elif period == 'day':
        df['Period Return'] = df['OMRXTBILL'].pct_change(periods=1)
    else:
        raise ValueError("Invalid period specified. Choose from 'month', 'week', or 'day'.")

    return df.set_index('date')['Period Return']
    

        

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
        xaxis_title=DATE,
        yaxis_title=HOLDINGS,
        legend_title=ASSETS_AND_INDICES
    )
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

    return fig

def plot_date_vs_total_holdings(date_holdings_df):
    # Create a mapping from asset IDs to display names
    asset_id_to_display_name = {asset: attributes["display name"] for asset, attributes in ASSETS_INDICES_MAP.items()}

    # Create a line plot for total holdings
    fig = px.line(date_holdings_df, x='Date', y='Total Holdings', color='Type', title='Date vs Total Holdings')

    # Update the names in the legend to display names
    for trace in fig.data:
        trace.name = asset_id_to_display_name.get(trace.name, trace.name)


    # Update layout for better visualization
    fig.update_layout(
        xaxis_title=DATE,
        yaxis_title=TOTAL_HOLDINGS,
        legend_title=TYPE
    )
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

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
            xaxis_title=DATE,
            yaxis_title=HOLDINGS,
            legend_title=TYPE
        )
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

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

def calculate_rolling_avg_returns(data, value_column='Total Holdings', date_column='Date', years=None):
    """
    Calculate rolling average of returns with an adjusted time window.
    
    Parameters:
    - data: DataFrame with date and value columns
    - value_column: Column name containing the values (default 'Total Holdings')
    - date_column: Column name containing dates (default 'Date')
    - years: Number of years for rolling window (if None, auto-calculated as 1/5 of total period)
    
    Returns:
    - pandas Series with rolling average returns, indexed by date
    """
    df = data.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values(date_column).set_index(date_column)
    
    # Calculate period returns
    returns = df[value_column].pct_change().dropna()
    
    if len(returns) == 0:
        return pd.Series(dtype=float)
    
    # Calculate time window
    n_years = (returns.index.max() - returns.index.min()).days / 365.25
    
    if years is None:
        # Auto-calculate: use 1/5 of total period, minimum 1 year
        print("YEARS ARE NONE")
        use_years = max(1, int(n_years // 5))
    else:
        use_years = years
    
    # Calculate periods per year based on data frequency
    periods_per_year = int(round(len(returns) / n_years)) if n_years > 0 else 1
    
    # Calculate window size
    window = max(1, int(use_years * periods_per_year))
    
    # Calculate rolling average
    rolling_avg_returns = returns.rolling(window=window, min_periods=window).mean().dropna()
    
    return rolling_avg_returns

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
    
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
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
    fig = px.line(drawdown_data, x='Date', y='Drawdown', color='Type', title=DRAWDOWN_PORTFOLIO_VS_INDEX)

    # Update layout for better visualization
    fig.update_layout(
        xaxis_title=DATE,
        yaxis_title=DRAWDOWN,
        legend_title=TYPE
    )
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

    return fig

def calculate_sharpe_ratio(returns,period, risk_free_rate=RISK_FREE_RATE ):
    """
    Calculate the Sharpe Ratio for a given series of returns.
    """
    if isinstance(risk_free_rate, pd.Series):
        risk_free_rate = risk_free_rate.reindex(returns.index, method='pad').fillna(0)
    excess_returns = returns - risk_free_rate / period
    annualized_excess_returns = excess_returns.mean() * period
    annualized_volatility = excess_returns.std() * np.sqrt(period)
    sharpe_ratio = annualized_excess_returns / annualized_volatility
    return sharpe_ratio

def calculate_variance(returns, period):
    """
    Calculate the variance of a given series of returns.
    """
    return np.var(returns) * period

def calculate_annualized_return(returns, period):
    """
    Calculate the annualized return for a given series of returns.
    """
    cumulative_returns = (1 + returns).cumprod()
    final_indexed_value = cumulative_returns.iloc[-1] * 100
    
    # Calculate annualized return 
    length_of_series = len(returns)
    if length_of_series > 0:
        annualized_return = (final_indexed_value / 100) ** (period / length_of_series) - 1
    else:
        annualized_return = 0
    
    return annualized_return

def calculate_maximum_drawdown(returns):
    """
    Calculate the maximum drawdown for a given series of returns.
    """
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()
    return max_drawdown

def calculate_stdev(returns,period):
    """
    Calculate the standard deviation for a given series of returns.
    """
    return returns.std() * np.sqrt(period)   

def calculate_alpha(returns_with_dates,index_returns_with_dates):
    """
    Calculate the alpha of a portfolio against a benchmark index. Returns and index returns should be aligned.
    """
    # Align the returns and index returns by their dates
    aligned_data = pd.merge(returns_with_dates, index_returns_with_dates, left_index=True, right_index=True, how='inner')
    returns = aligned_data.iloc[:, 0]
    index_returns = aligned_data.iloc[:, 1]

    b =  calculate_beta(returns_with_dates, index_returns_with_dates)
    returns_total = (1 + returns).prod() - 1
    index_returns_total = (1 + index_returns).prod() - 1
    alpha = returns_total - b * index_returns_total
    return alpha


def calculate_beta(returns_with_dates, index_returns_with_dates):
    """
    Calculate the beta of a portfolio against a benchmark index. Returns and index returns should be aligned.
    """
    # Align the returns and index returns by their dates
    aligned_data = pd.merge(returns_with_dates, index_returns_with_dates, left_index=True, right_index=True, how='inner')
    returns = aligned_data.iloc[:, 0]
    index_returns = aligned_data.iloc[:, 1]

    covariance = np.cov(returns, index_returns)[0][1]
    variance = np.var(index_returns)
    beta = covariance / variance if variance != 0 else 0
    return beta

def export_report_to_excel(combined_data, date_holdings_df, start_investment, allocation_limit, weights, sharpe_ratio, std_dev):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Write key metrics to a sheet
        metrics_df = pd.DataFrame({
            "Metric": ["Start Investment Amount", "Allocation Limit", "Sharpe Ratio", "Standard Deviation"],
            "Value": [f"{start_investment} SEK", f"{allocation_limit}%", f"{sharpe_ratio:.2f}", f"{std_dev:.2f}"]
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
    output.seek(0)
    st.download_button(
        label="Download Summary Report Excel",
        data=output,
        file_name="summary_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


def export_multi_port_to_excel(finished_portfolios):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for name, data in finished_portfolios.items():
            # Write key metrics for each portfolio
            metrics = {
                "Start Investment Amount": [f"{data.get('start_investment', '')} SEK"],
                "Allocation Limit": [f"{data.get('allocation_limit', '')}%"],
                "Sharpe Ratio": [f"{data.get('sharpe_ratio', float('nan')):.2f}"],
                "Std Dev": [f"{data.get('std_dev', float('nan')):.2f}"]
            }
            metrics_df = pd.DataFrame(metrics)
            metrics_df.to_excel(writer, sheet_name=f'{name}_Metrics', index=False)

            # Write asset weights
            weights = data.get('weights', {})
            if weights:
                weights_df = pd.DataFrame(list(weights.items()), columns=['Asset', 'Weight'])
                # If you have ASSETS_INDICES_MAP available, you can add display names
                # weights_df['Display Name'] = weights_df['Asset'].apply(lambda x: ASSETS_INDICES_MAP[x].get("display name", x))
                weights_df.to_excel(writer, sheet_name=f'{name}_Weights', index=False)

            # Write combined data
            combined_data = data.get('combined_data')
            if combined_data is not None:
                combined_data.to_excel(writer, sheet_name=f'{name}_Data', index=False)

            # Write date vs total holdings data
            date_holdings_df = data.get('date_holdings_df')
            if date_holdings_df is not None:
                date_holdings_df.to_excel(writer, sheet_name=f'{name}_Holdings', index=False)

    output.seek(0)
    st.download_button(
        label="Export Multi-Portfolio Summary Report",
        data=output,
        file_name="multi_summary_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

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
        title=TOTAL_HOLDINGS_IN_ASSETS_OVER_TIME,
        xaxis_title=DATE,
        yaxis_title=TOTAL_HOLDINGS,
        legend_title=PORTFOLIO, 
        colorway=HIGH_CONTRAST_COLORWAY,
    )
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

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
        title=TOTAL_HOLDINGS_IN_INDICES_OVER_TIME_ALL_PORTFOLIOS,
        xaxis_title=DATE,
        yaxis_title=TOTAL_HOLDINGS,
        legend_title=PORTFOLIO, 
        colorway=HIGH_CONTRAST_COLORWAY,
    )
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
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
        title=DRAWDOWNS_IN_ASSETS_ALL_PORTFOLIOS,
        xaxis_title=DATE,
        yaxis_title=DRAWDOWN,
        legend_title=PORTFOLIO,
        colorway=HIGH_CONTRAST_COLORWAY,
    )
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
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
        title=DRAWDOWNS_IN_INDICES_ALL_PORTFOLIOS,
        xaxis_title=DATE,
        yaxis_title=DRAWDOWN,
        legend_title=PORTFOLIO,
        colorway=HIGH_CONTRAST_COLORWAY,
    )
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

def plot_rolling_average_returns_vs_index(date_holdings_df, years=None, date_col='Date'):
    """
    Plot rolling average of returns for the portfolio and its index.
    """
    df = date_holdings_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    # Separate assets and indices
    asset_df = df[df['Type'] == 'Asset'].set_index(date_col)
    index_df = df[df['Type'] == 'Index'].set_index(date_col)

    # Calculate period returns
    asset_returns = asset_df['Total Holdings'].pct_change().dropna()
    index_returns = index_df['Total Holdings'].pct_change().dropna()

    # Calculate years in data
    n_years = (asset_returns.index.max() - asset_returns.index.min()).days / 365.25
    if years is None:
        years = max(1, int(n_years // 5))
    periods_per_year = int(round(len(asset_returns) / n_years)) if n_years > 0 else 1
    window = max(1, int(years * periods_per_year))

    # Rolling average
    asset_rolling = asset_returns.rolling(window=window, min_periods=window).mean().dropna()
    index_rolling = index_returns.rolling(window=window, min_periods=window).mean().dropna()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=asset_rolling.index, y=asset_rolling,
        mode='lines', name=f"Portfölj ({years}å rullande medelvärde)"
    ))
    fig.add_trace(go.Scatter(
        x=index_rolling.index, y=index_rolling,
        mode='lines', name=f"Index ({years}å rullande medelvärde)"
    ))
    fig.update_layout(
        title=ROLLING_AVG_RETURN,
        xaxis_title=DATE,
        yaxis_title=RETURN,
        legend_title=PORTFOLIO,
        colorway=HIGH_CONTRAST_COLORWAY,
    )
    return fig

def plot_multi_portfolio_rolling_average_returns(finished_portfolios, years=3, date_col='Date'):
    """
    Plot rolling average of returns for multiple portfolios.
    """
    fig = go.Figure()
    for name, data in finished_portfolios.items():
        df = data["date_holdings_df"].copy()
        # Filter for assets only
        assets_df = df[df['Type'] == 'Asset']
        
        # Use the existing calculate_rolling_avg_returns function
        rolling_returns = calculate_rolling_avg_returns(
            assets_df, 
            value_column='Total Holdings', 
            date_column=date_col, 
            years=years
        )
        
        if not rolling_returns.empty:
            # Determine the years used for display
            df_temp = assets_df.copy()
            df_temp[date_col] = pd.to_datetime(df_temp[date_col])
            
            fig.add_trace(go.Scatter(
                x=rolling_returns.index,
                y=rolling_returns,
                mode='lines',
                name=f"{name} ({years}y avg)"
            ))

    fig.update_layout(
        title=ROLLING_AVG_RETURN,
        xaxis_title=DATE,
        yaxis_title=RETURN,
        legend_title=PORTFOLIO,
        colorway=HIGH_CONTRAST_COLORWAY,
    )
    return fig

def generate_summary_report(combined_data, date_holdings_df, start_investment, allocation_limit, weights, asset_weights, period):

    """
    Generate a summary report for the portfolio. Old verison (since we are not only looking at one portfolio most times)
    """
    st.header("Summary Report")

    st.write("This report provides a summary of the portfolio's performance, including key metrics, asset weights, and visualizations.")
    st.write("The portfolio returns are adjusted for ongoing charges (OGC) and the allocation limit is set to " + str(allocation_limit) + "%.")
    st.write("The portfolio is rebalanced when allocation breaches the allocation limit for any position. This means that all the overflow of assets (or index holdings if the breach is in the index) are sold and the proceeds are used to buy the other assets in the portfolio such that they amount to the normal \% holdings specified.")

    risk_free_rate = fetch_risk_free_rate(date_holdings_df['Date'].max(), date_holdings_df['Date'].min())
    st.subheader("Key Metrics")
    st.write("This section provides key metrics for the portfolio. Variance and Sharpe Ratio are calculated based on the portfolio returns after ongoing charge (OGC).")
    st.write("The Sharpe Ratio is a measure of risk-adjusted return, while variance indicates the volatility of the portfolio.")
    st.write("The Sharpe Ratio is calculated using T-Bills as the risk-free rate. Alpha and Beta are calculated against the index returns. Alpha indicates the excess return of the portfolio compared to the index, while Beta measures the sensitivity of the portfolio's returns to the index's returns.")
    st.write("Period value " + str(period) + " is used to annualize the returns and volatility.")
    portfolio_returns = date_holdings_df[date_holdings_df['Type'] == 'Asset']["Period Return"].dropna()
    index_returns_with_dates = date_holdings_df.loc[date_holdings_df['Type'] == 'Index', ['Date', 'Period Return']].copy()
    index_returns_with_dates = index_returns_with_dates.set_index('Date')["Period Return"].dropna()
    returns_with_dates = date_holdings_df.loc[date_holdings_df['Type'] == 'Asset', ['Date', 'Period Return']].copy()
    returns_with_dates = returns_with_dates.set_index('Date')["Period Return"].dropna()
    sharpe_ratio = calculate_sharpe_ratio(returns_with_dates, period, risk_free_rate)
    variance = calculate_variance(portfolio_returns, period)
    max_drawdown = calculate_maximum_drawdown(portfolio_returns)
    stdev = calculate_stdev(portfolio_returns, period)
    annualized_return = calculate_annualized_return(portfolio_returns, period)
    final_holdings = date_holdings_df[date_holdings_df['Type'] == 'Asset']['Total Holdings'].iloc[-1]
    total_return = (final_holdings - start_investment) / start_investment
    total_return_index = (date_holdings_df[date_holdings_df['Type'] == 'Index']['Total Holdings'].iloc[-1] - start_investment) / start_investment
    stdev_index = calculate_stdev(date_holdings_df[date_holdings_df['Type'] == 'Index']["Period Return"].dropna(), period)
    alpha = calculate_alpha(returns_with_dates, index_returns_with_dates)
    beta = calculate_beta(returns_with_dates, index_returns_with_dates)

    metrics_df = pd.DataFrame({
        "Metric": [
            
            "Allocation Limit",
            "Sharpe Ratio",
            "Alpha",
            "Beta",
            "Standard Deviation",
            "Standard Deviation (Index)",
            "Variance",
            "Max Drawdown",
            "Annualized Return",
            "Start Investment Amount",
            "Final Holdings",
            "Total Return",
            "Total Return (Index)"

        ],
        "Value": [
            
            f"{allocation_limit}%",
            f"{sharpe_ratio:.2f}",
            f"{alpha:.2f}",
            f"{beta:.2f}",
            f"{stdev * 100:.2f}%",
            f"{stdev_index * 100:.2f}%",
            f"{variance * 100:.2f}%",
            f"{max_drawdown * 100:.2f}%",
            f"{annualized_return * 100:.2f}%",
            f"{start_investment} SEK",
            f"{final_holdings:.2f} SEK",
            f"{total_return:.2%}",
            f"{total_return_index:.2%}"
        ]
    })
    st.table(metrics_df)


    # Show weights pie chart
    st.subheader("Asset Allocation")     
    show_weights(asset_weights, key="asset_only")

    # Show the index weights
    st.subheader("Index allocation")
    index_only_weights = {k: v for k, v in weights.items() if k not in asset_weights}
    show_weights(index_only_weights, key="index_only")


    # Plot the holdings
    st.subheader("Holdings Over Time")
    fig_holdings = plot_holdings(combined_data)
    st.plotly_chart(fig_holdings)

    # Plot the date vs total holdings
    st.subheader("Total Holdings Over Time")
    fig_total_holdings = plot_date_vs_total_holdings(date_holdings_df)
    st.plotly_chart(fig_total_holdings)

    # Plot rolling average
    st.subheader("Rolling Average of Returns (Portfolio vs Index)")
    fig_rolling = plot_rolling_average_returns_vs_index(date_holdings_df)
    st.plotly_chart(fig_rolling)
    

    # Plot the drawdowns
    st.subheader("Drawdowns")
    index_data = date_holdings_df[date_holdings_df['Type'] == 'Index']
    portfolio_data = date_holdings_df[date_holdings_df['Type'] == 'Asset']
    fig_drawdowns = plot_drawdowns(portfolio_data, index_data)
    st.plotly_chart(fig_drawdowns)

    # Replace asset IDs with display names in 'Name' column of combined_data
    combined_data['Name'] = combined_data['Name'].map(lambda x: ASSETS_INDICES_MAP[x]["display name"] if x in ASSETS_INDICES_MAP else x)
   
    # Display portfolio data
    st.subheader("Portfolio Data")
    st.write(combined_data)

    # Display date vs total holdings data
    st.subheader("Date vs Total Holdings Data")
    st.write(date_holdings_df)
    # Export report to Excel
    
    export_report_to_excel(combined_data, date_holdings_df, fig_drawdowns, start_investment, allocation_limit, weights, sharpe_ratio, stdev)


def generate_multi_summary_report(finished_portfolios, allocation_limit,rolling_avg_period=12):
    """
    Display a summary report comparing multiple portfolios.
    finished_portfolios: dict of {portfolio_name: {combined_data, date_holdings_df, weights, asset_only_weights, period}}
    """
    st.header("Multi-Portfolio Comparison")

    st.write("This report provides a summary of the portfolios performances, including key metrics, asset weights, and visualizations.")
    st.write("The portfolios returns are adjusted for ongoing charges (OGC) and the allocation limit is set to " + str(allocation_limit) + "%.")
    st.write("The portfolios is rebalanced when allocation breaches the allocation limit for any position. This means that all the overflow of assets (or index holdings if the breach is in the index) are sold and the proceeds are used to buy the other assets in the portfolios such that they amount to the normal \% holdings specified.")
    risk_free_rate = fetch_risk_free_rate(max(data["date_holdings_df"]['Date'].max() for data in finished_portfolios.values()), 
                                            min(data["date_holdings_df"]['Date'].min() for data in finished_portfolios.values()))
    # Show a table of key metrics for each portfolio
    metrics_list = []
    for name, data in finished_portfolios.items():
        df = data["date_holdings_df"]
        period = data["period"]
        returns = df[df['Type'] == 'Asset']["Period Return"].dropna()
        index_returns_with_dates = df.loc[df['Type'] == 'Index', ['Date', 'Period Return']].copy()
        index_returns_with_dates = index_returns_with_dates.set_index('Date')["Period Return"].dropna()
        returns_with_dates = df.loc[df['Type'] == 'Asset', ['Date', 'Period Return']].copy()
        returns_with_dates = returns_with_dates.set_index('Date')["Period Return"].dropna()
        sharpe = calculate_sharpe_ratio(returns_with_dates, period,risk_free_rate)
        max_dd = calculate_maximum_drawdown(returns)
        variance = calculate_variance(returns, period)
        standard_deviation = calculate_stdev(returns, period)
        ann_return = calculate_annualized_return(returns, period)
        final_holdings = df[(df['Type'] == 'Asset') & (df['Date'] == df['Date'].max())]['Total Holdings'].iloc[-1]
        total_return = (final_holdings - data["start_investment"]) / data["start_investment"]
        alpha = calculate_alpha(returns_with_dates, index_returns_with_dates)
        beta = calculate_beta(returns_with_dates, index_returns_with_dates)
        metrics_list.append({
            "Portfolio": name,
            "Sharpe Ratio": f"{sharpe:.2f}",
            "Alpha": f"{alpha:.2f}",
            "Beta": f"{beta:.2f}",
            "Variance": f"{variance:.2%}",
            "Standard Deviation": f"{standard_deviation:.2%}",
            "Max Drawdown": f"{max_dd:.2%}",
            "Annualized Return": f"{ann_return:.2%}",
            "Start Investment": f"{data['start_investment']:.2f} SEK",
            "Final Holdings": f"{final_holdings:.2f} SEK",
            "Total Return": f"{total_return:.2%}"
        })
    if metrics_list:
        metrics_df = pd.DataFrame(metrics_list).set_index('Portfolio')
        st.subheader("Key Metrics Comparison")
        st.write("This section provides key metrics for each portfolio. Variance and Sharpe Ratio are calculated based on the portfolio returns after ongoing charge (OGC).")
        st.write("The Sharpe Ratio is a measure of risk-adjusted return, while variance indicates the volatility of the portfolio.")
        st.write("The Sharpe Ratio is calculated using T-Bills as the risk-free rate. Alpha and Beta are calculated against the index returns. Alpha indicates the excess return of the portfolio compared to the index, while Beta measures the sensitivity of the portfolio's returns to the index's returns.")
        st.write("Period value " + str(period) + " is used to annualize the returns and volatility.")
        st.write("The table below shows the key metrics per year for each portfolio.")
        st.table(metrics_df)

    # Plot all portfolios on the same chart
    st.subheader("Portfolio Value Comparison")
    st.write("This section shows the total holdings for each portfolio over time.")

    fig_portfolio = plot_multi_portfolio_total_holdings_assets(finished_portfolios)
    st.plotly_chart(fig_portfolio)

    st.subheader("Rolling Average of Returns Comparison")
    fig_rolling_multi = plot_multi_portfolio_rolling_average_returns(finished_portfolios, years=rolling_avg_period)
    st.plotly_chart(fig_rolling_multi)

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
    # Chnage the display names in the combined data
    for name, data in finished_portfolios.items():
        combined_data = data["combined_data"]
        combined_data['Name'] = combined_data['Name'].map(lambda x: ASSETS_INDICES_MAP[x]["display name"] if x in ASSETS_INDICES_MAP else x)
        data["combined_data"] = combined_data
   
   
    for name, data in finished_portfolios.items():
        df = data["date_holdings_df"].copy()
        
        # Calculate drawdowns for each Type separately (Asset and Index)
        df_with_drawdowns = df.groupby('Type', group_keys=False).apply(
            lambda x: calculate_drawdowns(x, 'Total Holdings')
        ).reset_index(drop=True)
        
        # Add rolling average returns column
        df_with_drawdowns['Rolling Avg Return'] = None
        
        for type_name in df['Type'].unique():
            type_mask = df_with_drawdowns['Type'] == type_name
            type_data = df_with_drawdowns[type_mask]
            
            if not type_data.empty:
                rolling_returns = calculate_rolling_avg_returns(
                    type_data, 
                    value_column='Total Holdings', 
                    date_column='Date',
                    years=rolling_avg_period
                )
                
                # Map rolling returns back to the main DataFrame
                for date, value in rolling_returns.items():
                    date_mask = (df_with_drawdowns['Date'] == date) & (df_with_drawdowns['Type'] == type_name)
                    df_with_drawdowns.loc[date_mask, 'Rolling Avg Return'] = value

        # Add indexed return (rebased to 100 at the first available value for each Type)
        for type_name in df['Type'].unique():
            type_mask = df_with_drawdowns['Type'] == type_name
            type_data = df_with_drawdowns[type_mask].sort_values('Date')
            if not type_data.empty:
                first_value = type_data['Total Holdings'].iloc[0]
                if first_value != 0:
                    indexed_return = (type_data['Total Holdings'] / first_value) * 100
                    df_with_drawdowns.loc[type_mask, 'Indexed Return (100)'] = indexed_return.values
                else:
                    df_with_drawdowns.loc[type_mask, 'Indexed Return (100)'] = np.nan
        
        st.subheader(f"Date vs Total Holdings, Drawdowns, Indexed return Rolling Avg Returns: {name}")
        st.write(df_with_drawdowns)


    export_multi_port_to_excel(finished_portfolios)


def show_predictions(combined_data, data_frequency):
    st.header("Monte Carlo Simulation")
    st.write("This section allows you to run a Monte Carlo simulation on the portfolio returns.")
    st.write("The simulation generates a range of possible future values for the portfolio based on historical returns.")
    st.write("The simulation uses a normal distribution to model the returns, and the number of simulations can be adjusted.")
    st.write("This simulation only works with monthly data. If the data is not in monthly format, please select monthly data.")
    if combined_data is None:
        st.warning("Please calculate a portfolio first.")
        return

    

    # Use 'OGC Adjusted Period Change' as returns
    if 'OGC Adjusted Period Change' in combined_data.columns:
        returns = combined_data['OGC Adjusted Period Change'].dropna()
    else:
        st.error("No 'OGC Adjusted Period Change' column found in portfolio data.")
        return

    start_value = st.number_input("Start Value", value=100000)
    n_years = st.number_input("Years", min_value=1, max_value=50, value=10)
    n_simulations = st.number_input("Simulations", min_value=10, max_value=10000, value=100)

    if st.button("Run Monte Carlo Simulation"):
        simulations = monte_carlo_simulation(returns, start_value, n_years, n_simulations)
        # Generate a date range starting from the last date in combined_data
        if 'date' in combined_data.columns:
            last_date = pd.to_datetime(combined_data['date'].iloc[-1])
        else:
            last_date = pd.to_datetime('today')

        dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=simulations.shape[0], freq='M')
        simulations_df = pd.DataFrame(simulations)
        simulations_df['Date'] = dates
        simulations_df.set_index('Date', inplace=True)
        st.line_chart(simulations_df)
        st.success("Simulation complete!")

def generate_multi_summary_report_indices(finished_portfolios, allocation_limit,rolling_avg_period=12):
    """
    Display a summary report comparing multiple portfolios.
    finished_portfolios: dict of {portfolio_name: {combined_data, date_holdings_df, weights, asset_only_weights, period}}
    """
    st.header("Multi-Index Portfolio Comparison")

    st.write("This report provides a summary of the portfolios performances, including key metrics, asset weights, and visualizations.")
    st.write("The portfolios is rebalanced when allocation breaches the allocation limit for any position. This means that all the overflow of assets (or index holdings if the breach is in the index) are sold and the proceeds are used to buy the other assets in the portfolios such that they amount to the normal \% holdings specified.")
    risk_free_rate = fetch_risk_free_rate(max(data["date_holdings_df"]['Date'].max() for data in finished_portfolios.values()), 
                                            min(data["date_holdings_df"]['Date'].min() for data in finished_portfolios.values()))
    # Show a table of key metrics for each portfolio
    metrics_list = []
    for name, data in finished_portfolios.items():
        df = data["date_holdings_df"]
        period = data["period"]
        returns = df[df['Type'] == 'Asset']["Period Return"].dropna()
        returns_with_dates = df[df['Type'] == 'Asset'].set_index('Date')["Period Return"].dropna()
        sharpe = calculate_sharpe_ratio(returns_with_dates, period,risk_free_rate)
        max_dd = calculate_maximum_drawdown(returns)
        variance = calculate_variance(returns, period)
        standard_deviation = calculate_stdev(returns, period)
        ann_return = calculate_annualized_return(returns, period)
        final_holdings = df[(df['Type'] == 'Asset') & (df['Date'] == df['Date'].max())]['Total Holdings'].iloc[-1]
        total_return = (final_holdings - data["start_investment"]) / data["start_investment"]

        metrics_list.append({
            "Portfolio": name,
            "Sharpe Ratio": f"{sharpe:.2f}",
            "Variance": f"{variance:.2%}",
            "Standard Deviation": f"{standard_deviation:.2%}",
            "Max Drawdown": f"{max_dd:.2%}",
            "Annualized Return": f"{ann_return:.2%}",
            "Start Investment": f"{data['start_investment']:.2f} SEK",
            "Final Holdings": f"{final_holdings:.2f} SEK",
            "Total Return": f"{total_return:.2%}"
        })
    if metrics_list:
        metrics_df = pd.DataFrame(metrics_list).set_index('Portfolio')
        st.subheader("Key Metrics Comparison")
        st.write("This section provides key metrics for each portfolio. Variance and Sharpe Ratio are calculated based on the portfolio returns after ongoing charge (OGC).")
        st.write("The Sharpe Ratio is a measure of risk-adjusted return, while variance indicates the volatility of the portfolio.")
        st.write("The Sharpe Ratio is calculated using American T-Bills as risk-free rate.")
        st.write("Period value " + str(period) + " is used to annualize the returns and volatility.")
        st.write("The table below shows the key metrics per year for each portfolio.")
        st.table(metrics_df)

    # Plot all portfolios on the same chart
    st.subheader("Portfolio Value Comparison")
    st.write("This section shows the total holdings for each portfolio over time.")

    fig_portfolio = plot_multi_portfolio_total_holdings_assets(finished_portfolios)
    st.plotly_chart(fig_portfolio)

    st.subheader("Rolling Average of Returns Comparison")
    fig_rolling_multi = plot_multi_portfolio_rolling_average_returns(finished_portfolios,years=rolling_avg_period)
    st.plotly_chart(fig_rolling_multi)

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
        st.markdown("### Weights:")
        show_weights(asset_only_weights, key=name + "_asset_only")


    # Chnage the display names in the combined data
    for name, data in finished_portfolios.items():
        combined_data = data["combined_data"]
        combined_data['Name'] = combined_data['Name'].map(lambda x: ASSETS_INDICES_MAP[x]["display name"] if x in ASSETS_INDICES_MAP else x)
        data["combined_data"] = combined_data
   
   
    for name, data in finished_portfolios.items():
        df = data["date_holdings_df"].copy()
        
        # Calculate drawdowns for each Type separately (Asset and Index)
        df_with_drawdowns = df.groupby('Type', group_keys=False).apply(
            lambda x: calculate_drawdowns(x, 'Total Holdings')
        ).reset_index(drop=True)
        
        # Add rolling average returns column
        df_with_drawdowns['Rolling Avg Return'] = None
        
        for type_name in df['Type'].unique():
            type_mask = df_with_drawdowns['Type'] == type_name
            type_data = df_with_drawdowns[type_mask]
            
            if not type_data.empty:
                rolling_returns = calculate_rolling_avg_returns(
                    type_data, 
                    value_column='Total Holdings', 
                    date_column='Date',
                    years=rolling_avg_period
                )
                
                # Map rolling returns back to the main DataFrame
                for date, value in rolling_returns.items():
                    date_mask = (df_with_drawdowns['Date'] == date) & (df_with_drawdowns['Type'] == type_name)
                    df_with_drawdowns.loc[date_mask, 'Rolling Avg Return'] = value

        # Add indexed return (rebased to 100 at the first available value for each Type)
        for type_name in df['Type'].unique():
            type_mask = df_with_drawdowns['Type'] == type_name
            type_data = df_with_drawdowns[type_mask].sort_values('Date')
            if not type_data.empty:
                first_value = type_data['Total Holdings'].iloc[0]
                if first_value != 0:
                    indexed_return = (type_data['Total Holdings'] / first_value) * 100
                    df_with_drawdowns.loc[type_mask, 'Indexed Return (100)'] = indexed_return.values
                else:
                    df_with_drawdowns.loc[type_mask, 'Indexed Return (100)'] = np.nan
        
        st.subheader(f"Date vs Total Holdings, Drawdowns, Indexed return Rolling Avg Returns: {name}")
        st.write(df_with_drawdowns)

    # export report to Excel
    export_multi_port_to_excel(finished_portfolios)