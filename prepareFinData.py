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
# This adds 3 columns: MACD_12_26_9, MACDs_12_26_9, MACDh_12_26_9
data.ta.macd(close='Close', fast=12, slow=26, signal=9, append=True)

# Rename MACD columns to match expected format (pandas_ta might use different names)
macd_cols = [col for col in data.columns if 'MACD' in col.upper()]
if 'MACD_12_26_9' not in data.columns:
    # pandas_ta might name them differently, rename to expected format
    for col in macd_cols:
        if 'MACD_' in col and 'MACDS' not in col.upper() and 'MACDH' not in col.upper():
            if col != 'MACD_12_26_9':
                data.rename(columns={col: 'MACD_12_26_9'}, inplace=True)
        elif 'SIGNAL' in col.upper() or 'MACDS' in col.upper():
            data.rename(columns={col: 'MACDs_12_26_9'}, inplace=True)
        elif 'HIST' in col.upper() or 'MACDH' in col.upper():
            data.rename(columns={col: 'MACDh_12_26_9'}, inplace=True)

# Calculate Williams %R
data['WILLR'] = ta.willr(data['High'], data['Low'], data['Close'], length=14)

# Calculate Momentum (using 'mom' from pandas-ta)
data['MOM'] = ta.mom(data['Close'], length=10)

# Reset index if Date is the index (yfinance returns Date as index)
if isinstance(data.index, pd.DatetimeIndex):
    data.reset_index(inplace=True)
    if 'Date' not in data.columns and 'Datetime' in data.columns:
        data.rename(columns={'Datetime': 'Date'}, inplace=True)

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

# Save the final DataFrame (without index if Date is a column)
if 'Date' in data.columns:
    data.to_csv(output_path, index=False)
else:
    data.to_csv(output_path)

print(f"Data saved successfully to {output_path}")
print(f"Data shape: {data.shape}")
print(f"Columns: {list(data.columns)}")
print("\nData head:")
print(data.head())