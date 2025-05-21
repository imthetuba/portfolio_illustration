import pandas as pd
from InfrontConnect import infront
from datetime import datetime, timedelta

# Connect to Infront API (use your credentials)
infront.InfrontConnect(user="David.Lundberg.ipt", password="Infront2022!")

# Load the asset/index map
df = pd.read_csv('assets_indices_map.csv')

# Collect all unique tickers (assets and indices)
tickers = set(df['asset']).union(set(df['index']))

# Test fetching data for each ticker
start_date = (datetime.today() - timedelta(days=10)).strftime('%Y-%m-%d')
end_date = datetime.today().strftime('%Y-%m-%d')

results = []
for ticker in tickers:
    try:
        data = infront.GetHistory(
            tickers=[ticker],
            fields=["last"],
            start_date=start_date,
            end_date=end_date
        )
        # Check if data is returned and not empty
        if data and ticker in data and isinstance(data[ticker], pd.DataFrame) and not data[ticker].empty:
            status = "OK"
        else:
            status = "No Data"
    except Exception as e:
        status = f"Error: {e}"
    results.append({"ticker": ticker, "status": status})

# Output results
results_df = pd.DataFrame(results)
print(results_df)