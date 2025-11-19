"""
Data Preparation Script for Google Colab
Run this BEFORE running the training script
"""
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import os

print("=" * 60)
print("Fetching S&P 500 (SPY) data from yfinance...")
print("=" * 60)

# 1. FETCH DATA (S&P 500 ETF)
data = yf.Ticker("SPY").history(period="max", interval="1d")
print(f"Fetched {len(data)} rows of data")
print(f"Date range: {data.index[0]} to {data.index[-1]}")

# -----------------------------------------------------
# 2. CALCULATE INDICATORS (from your proposal)
# -----------------------------------------------------
print("\nCalculating technical indicators...")

# Calculate RSI
data['RSI'] = ta.rsi(data['Close'], length=14)

# Calculate MACD (adds MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9)
data.ta.macd(close='Close', fast=12, slow=26, signal=9, append=True)

# Rename MACD columns to match expected format
if 'MACD_12_26_9' not in data.columns:
    # pandas_ta might name them differently, let's check and rename
    macd_cols = [col for col in data.columns if 'MACD' in col]
    print(f"MACD columns found: {macd_cols}")
    # Standard naming: MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
    if 'MACD_12_26_9' in data.columns:
        pass  # Already correct
    elif 'MACD' in data.columns:
        data.rename(columns={
            'MACD': 'MACD_12_26_9',
            'MACDs_12_26_9': 'MACDs_12_26_9',
            'MACDh_12_26_9': 'MACDh_12_26_9'
        }, inplace=True, errors='ignore')

# Calculate Williams %R
data['WILLR'] = ta.willr(data['High'], data['Low'], data['Close'], length=14)

# Calculate Momentum
data['MOM'] = ta.mom(data['Close'], length=10)

# Clean up data (remove rows with NaN values created by indicators)
initial_len = len(data)
data = data.dropna()
print(f"Removed {initial_len - len(data)} rows with NaN values")
print(f"Final data shape: {data.shape}")

# Reset index to make Date a column (if it's currently the index)
if data.index.name == 'Date' or isinstance(data.index, pd.DatetimeIndex):
    data.reset_index(inplace=True)
    if 'Date' not in data.columns and 'Datetime' in data.columns:
        data.rename(columns={'Datetime': 'Date'}, inplace=True)

# Ensure Date column exists
if 'Date' not in data.columns:
    if data.index.name == 'Date':
        data.reset_index(inplace=True)
    else:
        print("WARNING: Date column not found!")

# -----------------------------------------------------
# 3. SAVE TO CSV
# -----------------------------------------------------
# This is the path the data_loader.py will read from
output_path = './dataset/my_financial_data/SPY_with_indicators.csv'

# Ensure the directory exists (create it if not)
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Save the final DataFrame
data.to_csv(output_path, index=False)

print("\n" + "=" * 60)
print(f"✓ Data saved successfully to {output_path}")
print("=" * 60)
print(f"\nData shape: {data.shape}")
print(f"\nColumns: {list(data.columns)}")
print(f"\nFirst few rows:")
print(data.head())
print(f"\nLast few rows:")
print(data.tail())

# Verify the file exists
if os.path.exists(output_path):
    file_size = os.path.getsize(output_path) / (1024 * 1024)  # Size in MB
    print(f"\n✓ File verified! Size: {file_size:.2f} MB")
else:
    print("\n✗ ERROR: File was not created!")

