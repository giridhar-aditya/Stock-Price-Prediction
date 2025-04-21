import yfinance as yf
import pandas as pd

# Define stock symbol and date range
stock_symbol = "AAPL"
start_date = "2019-06-01"
end_date = "2024-06-01"

# Download stock data
df = yf.download(stock_symbol, start=start_date, end=end_date)

# Save to CSV
csv_filename = "stock_data.csv"
df.to_csv(csv_filename)

print(f"Stock data saved as {csv_filename}")
