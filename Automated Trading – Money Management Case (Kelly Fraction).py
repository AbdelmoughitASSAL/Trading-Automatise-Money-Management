import MetaTrader5 as mt5
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

# Connection to MetaTrader 5
if not mt5.initialize():
    print("MetaTrader 5 could not be started")
    mt5.shutdown()

# Define the symbol and timeframe
symbol = "EURUSD"
timeframe = mt5.TIMEFRAME_D1  # Daily data

# Define the date range
start_date = datetime(2020, 9, 10)
end_date = datetime(2024, 6, 29)

# Download historical data for the specified date range
rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)

# Convert to a Pandas DataFrame
data = pd.DataFrame(rates)
data['time'] = pd.to_datetime(data['time'], unit='s')

# Display the first rows of the data
print(data)

# Save the data to a CSV file
data.to_csv('data_analyzed1.csv', index=False)

# Extract only OHLC columns
data = data[['time','open', 'high', 'low', 'close']]
data_ohlc = data[['open', 'high', 'low', 'close']]

# Apply Min-Max scaling
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data_ohlc)
scaled_data = pd.DataFrame(scaled_data, index=data_ohlc.index, columns=data_ohlc.columns)

print(data)

# Calculate returns
data['Return'] = data['close'].pct_change()

# Function to calculate Kelly fraction
def calculate_kelly_fraction(win_prob, win_loss_ratio):
    return (win_prob * (win_loss_ratio + 1) - 1) / win_loss_ratio

# Initialize Kelly parameters
rolling_window = 30  # Window size for probability calculation
kelly_fractions = []

for i in range(len(data)):
    if i < rolling_window:
        kelly_fractions.append(0)  # Not enough data to calculate
    else:
        window_data = data.iloc[i - rolling_window:i]
        wins = window_data[window_data['Return'] > 0]
        losses = window_data[window_data['Return'] <= 0]
        win_prob = len(wins) / rolling_window
        avg_win = wins['Return'].mean() if len(wins) > 0 else 0
        avg_loss = abs(losses['Return'].mean()) if len(losses) > 0 else 0
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        kelly_fraction = calculate_kelly_fraction(win_prob, win_loss_ratio) if win_loss_ratio > 0 else 0
        kelly_fractions.append(kelly_fraction)

# Print Kelly parameters
print(f"Line {1000} : win_prob={win_prob:.2f}, avg_win={avg_win:.4f}, avg_loss={avg_loss:.4f}, win_loss_ratio={win_loss_ratio:.2f}, kelly_fraction={kelly_fraction:.4f}")

data['Kelly Fraction'] = kelly_fractions

# Current Kelly fraction value
current_kelly_value = data['Kelly Fraction'].iloc[-1]
print(f"Current Kelly fraction value: {current_kelly_value}")

# Kelly check for leverage
if current_kelly_value < 0:
    print("Kelly fraction is negative. Investment is not recommended. Leverage should be 0.")
else:
    recommended_leverage = current_kelly_value  # You can adjust this according to your criteria
    print(f"Kelly fraction is positive. Recommended leverage is {recommended_leverage}.")

# Save the data to CSV
data.to_csv('data_analyzed1.csv', index=False)

# Plot Kelly fraction evolution
plt.figure(figsize=(14, 7))
plt.plot(data['time'], data['Kelly Fraction'], label='Kelly Fraction')
plt.xlabel('Date')
plt.ylabel('Kelly Fraction')
plt.title('Optimal Kelly Fraction Evolution for EURUSD')
plt.legend()
plt.grid(True)
plt.show()

print(data)

# Disconnect MetaTrader 5
mt5.shutdown()

# Reload previously analyzed data
data = pd.read_csv('data_analyzed1.csv') 

# Initialize capital
initial_capital = 10000

# Leverage to apply
leverage = 1  # Change this manually to test different leverage levels
results = []

# Trading simulation with specified leverage
capital_kelly = initial_capital
capital_kelly_values = [capital_kelly]
for i in range(1, len(data)):
    trade_return = data['Return'].iloc[i]
    kelly_fraction = data['Kelly Fraction'].iloc[i]
    if kelly_fraction > 0:  # Do not invest if Kelly fraction is negative or zero
        trade_kelly = trade_return * kelly_fraction * leverage
        capital_kelly *= (1 + trade_kelly)
    capital_kelly_values.append(capital_kelly)
results = capital_kelly_values

# Convert 'time' column to datetime if necessary
data['time'] = pd.to_datetime(data['time'])

# Plot capital evolution with specified leverage
plt.figure(figsize=(14, 7))
plt.plot(data['time'], results, label=f'Leverage {leverage}')
plt.xlabel('Date')
plt.ylabel('Capital')
plt.title('Capital Evolution with Leverage for EURUSD')
plt.legend()
plt.grid(True)
plt.show()

# Repeat the same process for US500Cash
# Connection to MetaTrader 5
if not mt5.initialize():
    print("MetaTrader 5 could not be started")
    mt5.shutdown()

# Define symbol and timeframe
symbol = "US500"
timeframe = mt5.TIMEFRAME_D1  # Daily data

# Define date range
start_date = datetime(2020, 8, 18)
end_date = datetime(2024, 6, 29)

# Download historical data
rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)

# Convert to DataFrame
data = pd.DataFrame(rates)
data['time'] = pd.to_datetime(data['time'], unit='s')
print(data)

# Save to CSV
data.to_csv('data_analyzed2.csv', index=False)

# Extract only OHLC columns
data = data[['time','open', 'high', 'low', 'close']]
data_ohlc = data[['open', 'high', 'low', 'close']]

# Apply Min-Max scaling
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data_ohlc)
scaled_data = pd.DataFrame(scaled_data, index=data_ohlc.index, columns=data_ohlc.columns)
print(data)

# Calculate returns
data['Return'] = data['close'].pct_change()

# Calculate Kelly fraction
kelly_fractions = []
for i in range(len(data)):
    if i < rolling_window:
        kelly_fractions.append(0)
    else:
        window_data = data.iloc[i - rolling_window:i]
        wins = window_data[window_data['Return'] > 0]
        losses = window_data[window_data['Return'] <= 0]
        win_prob = len(wins) / rolling_window
        avg_win = wins['Return'].mean() if len(wins) > 0 else 0
        avg_loss = abs(losses['Return'].mean()) if len(losses) > 0 else 0
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        kelly_fraction = calculate_kelly_fraction(win_prob, win_loss_ratio) if win_loss_ratio > 0 else 0
        kelly_fractions.append(kelly_fraction)

print(f"Line {1000} : win_prob={win_prob:.2f}, avg_win={avg_win:.4f}, avg_loss={avg_loss:.4f}, win_loss_ratio={win_loss_ratio:.2f}, kelly_fraction={kelly_fraction:.4f}")
data['Kelly Fraction'] = kelly_fractions

current_kelly_value = data['Kelly Fraction'].iloc[-1]
print(f"Current Kelly fraction value: {current_kelly_value}")

if current_kelly_value < 0:
    print("Kelly fraction is negative. Investment is not recommended. Leverage should be 0.")
else:
    recommended_leverage = current_kelly_value
    print(f"Kelly fraction is positive. Recommended leverage is {recommended_leverage}.")

data.to_csv('data_analyzed2.csv', index=False)

plt.figure(figsize=(14, 7))
plt.plot(data['time'], data['Kelly Fraction'], label='Kelly Fraction')
plt.xlabel('Date')
plt.ylabel('Kelly Fraction')
plt.title('Optimal Kelly Fraction Evolution for US500Cash')
plt.legend()
plt.grid(True)
plt.show()

mt5.shutdown()
print(data)

# Reload previously analyzed data
data = pd.read_csv('data_analyzed2.csv') 

# Initialize capital
initial_capital = 10000
leverage = 1
results = []

capital_kelly = initial_capital
capital_kelly_values = [capital_kelly]
for i in range(1, len(data)):
    trade_return = data['Return'].iloc[i]
    kelly_fraction = data['Kelly Fraction'].iloc[i]
    if kelly_fraction > 0:
        trade_kelly = trade_return * kelly_fraction * leverage
        capital_kelly *= (1 + trade_kelly)
    capital_kelly_values.append(capital_kelly)
results = capital_kelly_values

data['time'] = pd.to_datetime(data['time'])

plt.figure(figsize=(14, 7))
plt.plot(data['time'], results, label=f'Leverage {leverage}')
plt.xlabel('Date')
plt.ylabel('Capital')
plt.title('Capital Evolution with Leverage for US500Cash')
plt.legend()
plt.grid(True)
plt.show()
