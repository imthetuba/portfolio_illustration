import pandas as pd
from InfrontConnect import infront
from datetime import datetime, timedelta

# Connect to Infront API (use your credentials)
infront.InfrontConnect(user="David.Lundberg.ipt", password="Infront2022!")

# Load the asset/index map
df = pd.read_csv('assets_indices_map.csv')

# Collect all unique tickers (assets and indices), drop NaNs and empty strings
tickers = set(df['asset'].dropna().astype(str)).union(set(df['index'].dropna().astype(str)))
tickers = {t for t in tickers if t and t.lower() != 'nan'}

# Test fetching data for each ticker
start_date = (datetime.today() - timedelta(days=30)).strftime('%Y-%m-%d')
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
        print(f"Ticker: {ticker}, Data keys: {list(data.keys()) if data else data}")
        # Try to match ignoring case and spaces
        found = False
        if data:
            for key in data.keys():
                if key.strip().lower() == ticker.strip().lower() and isinstance(data[key], pd.DataFrame) and not data[key].empty:
                    status = "OK"
                    found = True
                    break
        if not found:
            status = "No Data"
    except Exception as e:
        status = f"Error: {e}"
    results.append({"ticker": ticker, "status": status})

# Output results
results_df = pd.DataFrame(results)
print(results_df)