import yfinance as yf
import pandas as pd
import pandas_ta as ta # For technical indicators

print("Fetching S&P 500 data from yfinance...")

# 1. FETCH DATA (Example: S&P 500 ETF)
# Using 'SPY' as the ticker for S&P 500
data = yf.Ticker("SPY").history(period="max", interval="1d")

# -----------------------------------------------------
# 2. CALCULATE INDICATORS (from your proposal)
# -----------------------------------------------------
print("Calculating technical indicators...")

# Calculate RSI
data['RSI'] = ta.rsi(data['Close'], length=14)

# Calculate MACD
# This adds 3 columns: MACD, MACD_signal, MACD_hist
data.ta.macd(close='Close', fast=12, slow=26, signal=9, append=True)

# Calculate Williams %R
data['WILLR'] = ta.willr(data['High'], data['Low'], data['Close'], length=14)

# Calculate Momentum (using 'mom' from pandas-ta)
data['MOM'] = ta.mom(data['Close'], length=10)

# Clean up data (remove rows with NaN values created by indicators)
data = data.dropna()

# -----------------------------------------------------
# 3. SAVE TO CSV
# -----------------------------------------------------

# This is the path the data_loader.py will read from
output_path = './dataset/my_financial_data/SPY_with_indicators.csv'

# Ensure the directory exists (create it if not)
import os
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Save the final DataFrame
data.to_csv(output_path)

print(f"Data saved successfully to {output_path}")
print("Data head:")
print(data.head())