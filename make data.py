import yfinance as yf
import pandas as pd
import ta

# Parameters
stock_symbol = "AAPL"
start_date = "2010-01-01"
end_date = "2024-01-01"
csv_filename = "aapl_dataset.csv"

print(f"📥 Downloading {stock_symbol} data from {start_date} to {end_date}...")
df = yf.download(stock_symbol, start=start_date, end=end_date, auto_adjust=False)

# Flatten MultiIndex columns if necessary
if isinstance(df.columns, pd.MultiIndex):
    df.columns = [col[0] for col in df.columns]

df.reset_index(inplace=True)

# Keep only required columns
required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
df = df[required_columns]

# Ensure correct dtypes
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df['Open'] = pd.to_numeric(df['Open'], errors='coerce')
df['High'] = pd.to_numeric(df['High'], errors='coerce')
df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')

# Drop rows with any missing values after type conversion
df.dropna(subset=['Close', 'Open', 'High', 'Low', 'Volume'], inplace=True)

# Add technical indicators
print("🔧 Calculating technical indicators...")

# SMA and EMA
df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
df['EMA_20'] = ta.trend.ema_indicator(df['Close'], window=20)

# RSI
df['RSI'] = ta.momentum.rsi(df['Close'], window=14)

# MACD
macd = ta.trend.MACD(close=df['Close'])
df['MACD'] = macd.macd()
df['MACD_Signal'] = macd.macd_signal()
df['MACD_Diff'] = macd.macd_diff()

# Bollinger Bands
bb = ta.volatility.BollingerBands(close=df['Close'], window=20)
df['BB_High'] = bb.bollinger_hband()
df['BB_Low'] = bb.bollinger_lband()
df['BB_Width'] = bb.bollinger_wband()

# Stochastic Oscillator
stoch = ta.momentum.StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=14)
df['Stoch_K'] = stoch.stoch()
df['Stoch_D'] = stoch.stoch_signal()

# ADX
df['ADX'] = ta.trend.adx(high=df['High'], low=df['Low'], close=df['Close'], window=14)

# CCI
df['CCI'] = ta.trend.cci(high=df['High'], low=df['Low'], close=df['Close'], window=20)

# ATR
df['ATR'] = ta.volatility.average_true_range(high=df['High'], low=df['Low'], close=df['Close'], window=14)

# Drop rows with any remaining NaN values from indicators
df.dropna(inplace=True)

# Save to CSV
print(f"✅ Dataset ready: {df.shape[0]} rows | {df.shape[1]} columns")
df.to_csv(csv_filename, index=False)
