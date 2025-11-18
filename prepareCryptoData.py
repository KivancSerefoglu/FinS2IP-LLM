import pandas as pd
import pandas_ta as ta
from binance import Client # Import the Binance client
import os

print("Fetching BTC/USDT data from Binance...")

# 1. FETCH DATA
# You don't need an API key for public historical data
client = Client(tld='us')

# Get daily klines (candles) for BTC/USDT starting from 1 Jan, 2017
klines = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_1DAY, "1 Jan, 2017")

# Define the column names for the DataFrame
columns = ['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 
           'Close Time', 'Quote Asset Volume', 'Number of Trades', 
           'Taker Buy Base', 'Taker Buy Quote', 'Ignore']

# Create the initial DataFrame
df = pd.DataFrame(klines, columns=columns)

# -----------------------------------------------------
# 2. PROCESS DATA & CALCULATE INDICATORS
# -----------------------------------------------------
print("Processing data and calculating indicators...")

# A) Convert timestamp to datetime and set as index
df['Date'] = pd.to_datetime(df['Open Time'], unit='ms')
df.set_index('Date', inplace=True)

# B) Select only the columns we need
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

# C) Convert price and volume columns to numeric (they are strings by default)
for col in df.columns:
    df[col] = pd.to_numeric(df[col])

# D) Calculate indicators (from your proposal)
df['RSI'] = ta.rsi(df['Close'], length=14)
df.ta.macd(close='Close', fast=12, slow=26, signal=9, append=True)
df['WILLR'] = ta.willr(df['High'], df['Low'], df['Close'], length=14)
df['MOM'] = ta.mom(df['Close'], length=10)

# E) Clean up data (remove NaNs from indicator calculations)
df = df.dropna()

# -----------------------------------------------------
# 3. SAVE TO CSV
# -----------------------------------------------------
output_path = './dataset/my_financial_data/BTC_with_indicators.csv'

# Ensure the directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Save the final DataFrame
df.to_csv(output_path)

print(f"Data saved successfully to {output_path}")
print("Data head:")
print(df.head())