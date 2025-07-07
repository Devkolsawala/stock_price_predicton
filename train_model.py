import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)

# --- 1. Data Loading and Cleaning ---
print("Loading and cleaning data...")

data_folder = 'VI Data'
file_names = [
    os.path.join(data_folder, '2015-2016.csv'),
    os.path.join(data_folder, '2016-2017.csv'),
    os.path.join(data_folder, '2017-2018.csv'),
    os.path.join(data_folder, '2018-2019.csv'),
    os.path.join(data_folder, '2019-2020.csv'),
    os.path.join(data_folder, '2020-2021.csv'),
    os.path.join(data_folder, '2021-2022.csv'),
    os.path.join(data_folder, '2022-2023.csv'),
    os.path.join(data_folder, '2023-2024.csv'),
    os.path.join(data_folder, '2024-2025.csv')
]

def convert_volume_to_float(vol_str):
    """Convert volume string to float"""
    try:
        if isinstance(vol_str, str):
            vol_str = vol_str.strip()
            if vol_str.endswith('B'):
                return float(vol_str[:-1]) * 1_000_000_000
            elif vol_str.endswith('M'):
                return float(vol_str[:-1]) * 1_000_000
            elif vol_str.endswith('K'):
                return float(vol_str[:-1]) * 1_000
        return float(vol_str)
    except (ValueError, TypeError):
        return np.nan

try:
    all_dfs = []
    for filename in file_names:
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            all_dfs.append(df)
    
    if not all_dfs:
        raise FileNotFoundError("No CSV files found in VI Data folder")
    
    data = pd.concat(all_dfs, ignore_index=True)
    
    # Clean data
    data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
    data.sort_values('Date', inplace=True)
    data.set_index('Date', inplace=True)
    
    # Clean volume column
    data['Vol.'] = data['Vol.'].apply(convert_volume_to_float)
    
    # Clean percentage columns
    for col in ['Change %']:
        if col in data.columns:
            data[col] = data[col].str.replace('%', '').astype(float)
    
    # Remove rows with missing values
    data.dropna(inplace=True)
    
    print(f"Data loaded successfully. Shape: {data.shape}")
    print(f"Date range: {data.index.min()} to {data.index.max()}")
    print(f"Price range: {data['Price'].min():.2f} to {data['Price'].max():.2f}")
    
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# --- 2. Proper Feature Engineering (No Data Leakage) ---
def create_proper_features(df, lookback_window=20):
    """Create features without data leakage"""
    df_features = df.copy()
    
    # Basic price features (using historical data only)
    df_features['Price_Lag_1'] = df_features['Price'].shift(1)
    df_features['Price_Lag_2'] = df_features['Price'].shift(2)
    df_features['Price_Lag_3'] = df_features['Price'].shift(3)
    df_features['Price_Lag_5'] = df_features['Price'].shift(5)
    
    # Returns (most important for prediction)
    df_features['Return_1'] = df_features['Price'].pct_change(1)
    df_features['Return_2'] = df_features['Price'].pct_change(2)
    df_features['Return_5'] = df_features['Price'].pct_change(5)
    df_features['Return_10'] = df_features['Price'].pct_change(10)
    
    # Volume features
    df_features['Volume_Lag_1'] = df_features['Vol.'].shift(1)
    df_features['Volume_Lag_2'] = df_features['Vol.'].shift(2)
    df_features['Volume_Change'] = df_features['Vol.'].pct_change(1)
    
    # Historical moving averages (using only past data)
    for window in [5, 10, 20]:
        df_features[f'MA_{window}'] = df_features['Price'].shift(1).rolling(window=window).mean()
        df_features[f'Price_to_MA_{window}'] = df_features['Price'] / df_features[f'MA_{window}']
    
    # Volatility features
    for window in [5, 10, 20]:
        df_features[f'Volatility_{window}'] = df_features['Return_1'].shift(1).rolling(window=window).std()
    
    # Price momentum
    df_features['Momentum_5'] = df_features['Price'] / df_features['Price'].shift(5) - 1
    df_features['Momentum_10'] = df_features['Price'] / df_features['Price'].shift(10) - 1
    
    # Historical highs and lows
    for window in [10, 20]:
        df_features[f'High_{window}'] = df_features['Price'].shift(1).rolling(window=window).max()
        df_features[f'Low_{window}'] = df_features['Price'].shift(1).rolling(window=window).min()
        df_features[f'Price_Position_{window}'] = (df_features['Price'] - df_features[f'Low_{window}']) / (df_features[f'High_{window}'] - df_features[f'Low_{window}'])
    
    # Time-based features
    df_features['DayOfWeek'] = df_features.index.dayofweek
    df_features['Month'] = df_features.index.month
    df_features['Quarter'] = df_features.index.quarter
    
    # Technical indicators (using historical data)
    # RSI
    delta = df_features['Price'].diff()
    gain = (delta.where(delta > 0, 0)).shift(1).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).shift(1).rolling(window=14).mean()
    rs = gain / loss
    df_features['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands position
    ma_20 = df_features['Price'].shift(1).rolling(window=20).mean()
    std_20 = df_features['Price'].shift(1).rolling(window=20).std()
    df_features['BB_Position'] = (df_features['Price'] - ma_20) / (2 * std_20)
    
    return df_features

print("Creating proper features...")
data_features = create_proper_features(data)

# Remove rows with NaN values
data_features.dropna(inplace=True)
print(f"Shape after feature engineering: {data_features.shape}")

# --- 3. Select Best Features ---
feature_columns = [
    'Price_Lag_1', 'Price_Lag_2', 'Price_Lag_3', 'Price_Lag_5',
    'Return_1', 'Return_2', 'Return_5', 'Return_10',
    'Volume_Lag_1', 'Volume_Lag_2', 'Volume_Change',
    'MA_5', 'MA_10', 'MA_20',
    'Price_to_MA_5', 'Price_to_MA_10', 'Price_to_MA_20',
    'Volatility_5', 'Volatility_10', 'Volatility_20',
    'Momentum_5', 'Momentum_10',
    'Price_Position_10', 'Price_Position_20',
    'DayOfWeek', 'Month', 'Quarter',
    'RSI', 'BB_Position'
]

# Filter available features
available_features = [col for col in feature_columns if col in data_features.columns]
print(f"Using {len(available_features)} features: {available_features}")

# --- 4. Prepare Data for LSTM ---
# Use price returns as target instead of absolute prices
data_features['Target'] = data_features['Price'].pct_change(1).shift(-1)  # Next day return
data_features.dropna(inplace=True)

# Create feature matrix
X = data_features[available_features].values
y = data_features['Target'].values

print(f"Final data shape: X={X.shape}, y={y.shape}")

# --- 5. Time Series Split ---
train_size = int(len(X) * 0.7)
val_size = int(len(X) * 0.15)

X_train = X[:train_size]
y_train = y[:train_size]
X_val = X[train_size:train_size + val_size]
y_val = y[train_size:train_size + val_size]
X_test = X[train_size + val_size:]
y_test = y[train_size + val_size:]

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# --- 6. Scaling ---
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# --- 7. Create Sequences ---
def create_sequences(X, y, time_steps=10):
    """Create sequences for LSTM"""
    X_seq, y_seq = [], []
    for i in range(len(X) - time_steps):
        X_seq.append(X[i:(i + time_steps)])
        y_seq.append(y[i + time_steps])
    return np.array(X_seq), np.array(y_seq)

time_steps = 10
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, time_steps)
X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val_scaled, time_steps)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, time_steps)

print(f"Sequence shapes: X_train={X_train_seq.shape}, y_train={y_train_seq.shape}")

# --- 8. Build Optimized LSTM Model ---
def build_lstm_model(input_shape):
    """Build a simple, effective LSTM model"""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        BatchNormalization(),
        
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        BatchNormalization(),
        
        Dense(25, activation='relu'),
        Dropout(0.1),
        
        Dense(1, activation='linear')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    return model

model = build_lstm_model((time_steps, len(available_features)))
print("\nModel Summary:")
model.summary()

# --- 9. Train Model ---
print("\nTraining model...")

callbacks = [
    EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
]

history = model.fit(
    X_train_seq, y_train_seq,
    validation_data=(X_val_seq, y_val_seq),
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# --- 10. Evaluate Model ---
def evaluate_model(model, X_test, y_test, scaler_y):
    """Evaluate model performance"""
    y_pred_scaled = model.predict(X_test)
    
    # Inverse transform predictions
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
    
    # Calculate metrics
    mse = mean_squared_error(y_actual, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_actual, y_pred)
    r2 = r2_score(y_actual, y_pred)
    
    # Directional accuracy
    direction_actual = np.sign(y_actual)
    direction_pred = np.sign(y_pred)
    direction_accuracy = np.mean(direction_actual == direction_pred) * 100
    
    print(f"\nModel Performance:")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Directional Accuracy: {direction_accuracy:.2f}%")
    
    return y_pred, y_actual

y_pred, y_actual = evaluate_model(model, X_test_seq, y_test_seq, scaler_y)

# --- 11. Convert Returns to Prices for Visualization ---
def convert_returns_to_prices(returns, initial_price):
    """Convert returns to price series"""
    prices = [initial_price]
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    return np.array(prices[1:])

# Get the last known price for reconstruction
last_price_index = train_size + val_size + time_steps - 1
initial_price = data_features['Price'].iloc[last_price_index]

# Convert returns to prices
y_actual_prices = convert_returns_to_prices(y_actual, initial_price)
y_pred_prices = convert_returns_to_prices(y_pred, initial_price)

# Calculate price-based metrics
price_mse = mean_squared_error(y_actual_prices, y_pred_prices)
price_rmse = np.sqrt(price_mse)
price_mae = mean_absolute_error(y_actual_prices, y_pred_prices)
price_r2 = r2_score(y_actual_prices, y_pred_prices)

print(f"\nPrice-based Performance:")
print(f"Price RMSE: {price_rmse:.4f}")
print(f"Price MAE: {price_mae:.4f}")
print(f"Price R² Score: {price_r2:.4f}")

# --- 12. Visualization ---
def plot_results(y_actual_prices, y_pred_prices, history):
    """Plot results"""
    plt.figure(figsize=(15, 10))
    
    # Price predictions
    plt.subplot(2, 2, 1)
    plt.plot(y_actual_prices, label='Actual Prices', color='red', linewidth=2)
    plt.plot(y_pred_prices, label='Predicted Prices', color='blue', linewidth=2)
    plt.title('VODA Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Training history
    plt.subplot(2, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Scatter plot
    plt.subplot(2, 2, 3)
    plt.scatter(y_actual_prices, y_pred_prices, alpha=0.6)
    plt.plot([y_actual_prices.min(), y_actual_prices.max()], 
             [y_actual_prices.min(), y_actual_prices.max()], 'r--', lw=2)
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title('Actual vs Predicted Prices')
    plt.grid(True, alpha=0.3)
    
    # Error distribution
    plt.subplot(2, 2, 4)
    errors = y_actual_prices - y_pred_prices
    plt.hist(errors, bins=30, alpha=0.7, edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('voda_optimized_results.png', dpi=300, bbox_inches='tight')
    plt.show()

plot_results(y_actual_prices, y_pred_prices, history)

# --- 13. Future Predictions ---
def predict_future(model, last_sequence, scaler_X, scaler_y, available_features, days=5):
    """Predict future prices"""
    predictions = []
    current_sequence = last_sequence.copy()
    
    for day in range(days):
        # Predict next return
        next_return_scaled = model.predict(current_sequence.reshape(1, time_steps, len(available_features)))
        next_return = scaler_y.inverse_transform(next_return_scaled.reshape(-1, 1)).flatten()[0]
        
        predictions.append(next_return)
        
        # Update sequence (simplified - in practice you'd need to update all features)
        # For now, we'll use the predicted return to update the sequence
        new_features = current_sequence[-1].copy()
        new_features[0] = next_return  # Update the first feature (return)
        
        # Shift sequence
        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1] = new_features
    
    return predictions

# Get last sequence for prediction
last_sequence = X_test_seq[-1]
future_returns = predict_future(model, last_sequence, scaler_X, scaler_y, available_features)

# Convert to prices
current_price = data_features['Price'].iloc[-1]
future_prices = []
current_p = current_price

print(f"\nFuture Predictions:")
print(f"Current Price: {current_price:.2f}")
print("-" * 40)

for i, ret in enumerate(future_returns, 1):
    future_price = current_p * (1 + ret)
    future_prices.append(future_price)
    print(f"Day {i}: {future_price:.2f} (Return: {ret:.4f})")
    current_p = future_price

# --- 14. Summary ---
print(f"\n{'='*50}")
print("OPTIMIZED MODEL SUMMARY")
print(f"{'='*50}")
print(f"Model Type: LSTM with Return Prediction")
print(f"Features: {len(available_features)} (no data leakage)")
print(f"Time Steps: {time_steps}")
print(f"Target: Daily Returns (not absolute prices)")
print(f"\nKey Fixes Applied:")
print("1. Eliminated data leakage in features")
print("2. Predict returns instead of absolute prices")
print("3. Simplified model architecture")
print("4. Proper time series validation")
print("5. Standardized scaling approach")
print("6. Added directional accuracy metric")
print(f"\nFinal Performance:")
print(f"- Return R²: {r2:.4f}")
print(f"- Price RMSE: {price_rmse:.4f}")
print(f"- Directional Accuracy: {np.mean(np.sign(y_actual) == np.sign(y_pred)) * 100:.2f}%")
print(f"{'='*50}")













# import os
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization, Attention
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.regularizers import l1_l2
# import matplotlib.pyplot as plt
# import warnings
# warnings.filterwarnings('ignore')

# # --- 1. Enhanced Data Loading and Preprocessing ---

# data_folder = 'VI Data'
# file_names = [
#     os.path.join(data_folder, '2015-2016.csv'),
#     os.path.join(data_folder, '2016-2017.csv'),
#     os.path.join(data_folder, '2017-2018.csv'),
#     os.path.join(data_folder, '2018-2019.csv'),
#     os.path.join(data_folder, '2019-2020.csv'),
#     os.path.join(data_folder, '2020-2021.csv'),
#     os.path.join(data_folder, '2021-2022.csv'),
#     os.path.join(data_folder, '2022-2023.csv'),
#     os.path.join(data_folder, '2023-2024.csv'),
#     os.path.join(data_folder, '2024-2025.csv')
# ]

# try:
#     all_dfs = [pd.read_csv(filename) for filename in file_names]
#     data = pd.concat(all_dfs, ignore_index=True)
#     data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
#     data.sort_values('Date', inplace=True)
#     data.set_index('Date', inplace=True)
#     print(f"Data loaded successfully. Shape after loading: {data.shape}")
# except FileNotFoundError as e:
#     print(f"Error: {e}. Please make sure all CSV files are in the 'VI Data' folder.")
#     exit()

# # --- 2. Advanced Feature Engineering ---

# def convert_volume_to_float(vol_str):
#     try:
#         if isinstance(vol_str, str):
#             vol_str = vol_str.strip()
#             if vol_str.endswith('B'):
#                 return float(vol_str[:-1]) * 1_000_000_000
#             elif vol_str.endswith('M'):
#                 return float(vol_str[:-1]) * 1_000_000
#             elif vol_str.endswith('K'):
#                 return float(vol_str[:-1]) * 1_000
#         return float(vol_str)
#     except (ValueError, TypeError):
#         return np.nan

# data['Vol.'] = data['Vol.'].apply(convert_volume_to_float)

# # Clean percentage columns
# for col in ['Change %']:
#     if col in data.columns:
#         data[col] = data[col].str.replace('%', '').astype(float)

# print(f"Number of rows with invalid data to be dropped: {data.isna().any(axis=1).sum()}")
# data.dropna(inplace=True)
# print(f"Shape after cleaning: {data.shape}")

# # --- Enhanced Feature Engineering ---
# def create_technical_indicators(df):
#     """Create advanced technical indicators"""
#     df = df.copy()
    
#     # Price-based indicators
#     df['Price_MA_5'] = df['Price'].rolling(window=5).mean()
#     df['Price_MA_10'] = df['Price'].rolling(window=10).mean()
#     df['Price_MA_20'] = df['Price'].rolling(window=20).mean()
#     df['Price_MA_50'] = df['Price'].rolling(window=50).mean()
    
#     # Exponential Moving Averages
#     df['Price_EMA_12'] = df['Price'].ewm(span=12).mean()
#     df['Price_EMA_26'] = df['Price'].ewm(span=26).mean()
    
#     # MACD
#     df['MACD'] = df['Price_EMA_12'] - df['Price_EMA_26']
#     df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
#     df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
#     # RSI
#     delta = df['Price'].diff()
#     gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
#     loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
#     rs = gain / loss
#     df['RSI'] = 100 - (100 / (1 + rs))
    
#     # Bollinger Bands
#     df['BB_Middle'] = df['Price'].rolling(window=20).mean()
#     bb_std = df['Price'].rolling(window=20).std()
#     df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
#     df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
#     df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
#     df['BB_Position'] = (df['Price'] - df['BB_Lower']) / df['BB_Width']
    
#     # Volume indicators
#     df['Volume_MA_10'] = df['Vol.'].rolling(window=10).mean()
#     df['Volume_Ratio'] = df['Vol.'] / df['Volume_MA_10']
    
#     # Price momentum and volatility
#     df['Price_Return'] = df['Price'].pct_change()
#     df['Price_Volatility'] = df['Price_Return'].rolling(window=20).std()
#     df['Price_Momentum'] = df['Price'] / df['Price'].shift(10) - 1
    
#     # Support and Resistance levels
#     df['High_20'] = df['Price'].rolling(window=20).max()
#     df['Low_20'] = df['Price'].rolling(window=20).min()
#     df['Price_Position'] = (df['Price'] - df['Low_20']) / (df['High_20'] - df['Low_20'])
    
#     # Lag features
#     for lag in [1, 2, 3, 5, 10]:
#         df[f'Price_Lag_{lag}'] = df['Price'].shift(lag)
#         df[f'Volume_Lag_{lag}'] = df['Vol.'].shift(lag)
    
#     return df

# # Apply feature engineering
# print("Creating technical indicators...")
# data_enhanced = create_technical_indicators(data)
# data_enhanced.dropna(inplace=True)
# print(f"Shape after feature engineering: {data_enhanced.shape}")

# # Select enhanced features
# feature_columns = [
#     'Price', 'Vol.', 'Price_MA_5', 'Price_MA_10', 'Price_MA_20', 'Price_MA_50',
#     'Price_EMA_12', 'Price_EMA_26', 'MACD', 'MACD_Signal', 'MACD_Histogram',
#     'RSI', 'BB_Width', 'BB_Position', 'Volume_Ratio', 'Price_Return',
#     'Price_Volatility', 'Price_Momentum', 'Price_Position',
#     'Price_Lag_1', 'Price_Lag_2', 'Price_Lag_3', 'Volume_Lag_1', 'Volume_Lag_2'
# ]

# # Filter available columns
# available_features = [col for col in feature_columns if col in data_enhanced.columns]
# features = data_enhanced[available_features].values

# print(f"Using {len(available_features)} features: {available_features}")

# # --- 3. Advanced Data Splitting & Scaling ---

# # Use more sophisticated train-validation-test split
# train_size = int(len(features) * 0.7)
# val_size = int(len(features) * 0.15)
# test_size = len(features) - train_size - val_size

# train_data = features[:train_size]
# val_data = features[train_size:train_size + val_size]
# test_data = features[train_size + val_size:]

# # Use RobustScaler for better handling of outliers
# scaler = RobustScaler()
# train_data_scaled = scaler.fit_transform(train_data)
# val_data_scaled = scaler.transform(val_data)
# test_data_scaled = scaler.transform(test_data)

# # --- 4. Optimized Sequence Creation ---

# def create_sequences_advanced(data, time_step=60, future_step=1):
#     """Create sequences with multiple future predictions"""
#     X, y = [], []
#     for i in range(len(data) - time_step - future_step + 1):
#         X.append(data[i:(i + time_step)])
#         y.append(data[i + time_step:i + time_step + future_step, 0])  # Predict next 'future_step' prices
#     return np.array(X), np.array(y)

# # Optimize time_step based on data size
# optimal_time_step = min(60, len(train_data_scaled) // 20)  # Adaptive time step
# print(f"Using time_step: {optimal_time_step}")

# X_train, y_train = create_sequences_advanced(train_data_scaled, optimal_time_step)
# X_val, y_val = create_sequences_advanced(val_data_scaled, optimal_time_step)
# X_test, y_test = create_sequences_advanced(test_data_scaled, optimal_time_step)

# print(f"Training data shape: X_train={X_train.shape}, y_train={y_train.shape}")
# print(f"Validation data shape: X_val={X_val.shape}, y_val={y_val.shape}")
# print(f"Test data shape: X_test={X_test.shape}, y_test={y_test.shape}")

# if X_train.shape[0] == 0:
#     print("Error: Insufficient data for training.")
#     exit()

# # --- 5. Advanced Model Architecture ---

# def create_advanced_model(input_shape):
#     """Create an advanced LSTM model with attention and regularization"""
#     model = Sequential()
    
#     # First Bidirectional LSTM layer with more units
#     model.add(Bidirectional(LSTM(units=128, return_sequences=True, 
#                                  dropout=0.2, recurrent_dropout=0.2,
#                                  kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
#                            input_shape=input_shape))
#     model.add(BatchNormalization())
    
#     # Second Bidirectional LSTM layer
#     model.add(Bidirectional(LSTM(units=64, return_sequences=True,
#                                  dropout=0.2, recurrent_dropout=0.2,
#                                  kernel_regularizer=l1_l2(l1=0.01, l2=0.01))))
#     model.add(BatchNormalization())
    
#     # Third LSTM layer
#     model.add(LSTM(units=32, return_sequences=False,
#                    dropout=0.2, recurrent_dropout=0.2))
#     model.add(BatchNormalization())
    
#     # Dense layers with dropout
#     model.add(Dense(units=64, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))
#     model.add(Dropout(0.3))
#     model.add(Dense(units=32, activation='relu'))
#     model.add(Dropout(0.2))
#     model.add(Dense(units=1, activation='linear'))
    
#     return model

# model = create_advanced_model((X_train.shape[1], X_train.shape[2]))

# # Advanced optimizer with custom learning rate
# optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
# model.compile(optimizer=optimizer, loss='huber', metrics=['mae', 'mse'])

# model.build(input_shape=(None, X_train.shape[1], X_train.shape[2]))
# model.summary()

# # --- 6. Advanced Training with Multiple Callbacks ---

# print("\nTraining the optimized model...")

# callbacks = [
#     EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1),
#     ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7, verbose=1),
#     ModelCheckpoint('best_voda_model.h5', monitor='val_loss', save_best_only=True, verbose=1)
# ]

# history = model.fit(
#     X_train, y_train,
#     epochs=200,
#     batch_size=16,  # Smaller batch size for better convergence
#     validation_data=(X_val, y_val),
#     callbacks=callbacks,
#     verbose=1,
#     shuffle=True
# )
# print("Model training complete.")

# # --- 7. Comprehensive Evaluation ---

# def evaluate_model(model, X_test, y_test, scaler, feature_count):
#     """Comprehensive model evaluation"""
#     print("\nEvaluating the model...")
    
#     # Make predictions
#     predicted_scaled = model.predict(X_test)
    
#     # Inverse transform predictions
#     dummy_pred = np.zeros((len(predicted_scaled), feature_count))
#     dummy_pred[:, 0] = predicted_scaled.flatten()
#     predicted_prices = scaler.inverse_transform(dummy_pred)[:, 0]
    
#     # Inverse transform actual values
#     dummy_actual = np.zeros((len(y_test), feature_count))
#     dummy_actual[:, 0] = y_test.flatten()
#     actual_prices = scaler.inverse_transform(dummy_actual)[:, 0]
    
#     # Calculate metrics
#     mse = mean_squared_error(actual_prices, predicted_prices)
#     rmse = np.sqrt(mse)
#     mae = mean_absolute_error(actual_prices, predicted_prices)
#     r2 = r2_score(actual_prices, predicted_prices)
#     mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100
    
#     print(f"Model Performance Metrics:")
#     print(f"RMSE: {rmse:.4f}")
#     print(f"MAE: {mae:.4f}")
#     print(f"R² Score: {r2:.4f}")
#     print(f"MAPE: {mape:.2f}%")
    
#     return predicted_prices, actual_prices

# predicted_prices, actual_prices = evaluate_model(model, X_test, y_test, scaler, len(available_features))

# # --- 8. Enhanced Visualization ---

# def plot_results(actual_prices, predicted_prices, data_enhanced, train_size, val_size, optimal_time_step):
#     """Create comprehensive result visualization"""
#     plt.figure(figsize=(16, 10))
    
#     # Main prediction plot
#     plt.subplot(2, 2, 1)
#     plot_start_idx = train_size + val_size + optimal_time_step
#     plot_dates = data_enhanced.index[plot_start_idx:plot_start_idx + len(actual_prices)]
    
#     plt.plot(plot_dates, actual_prices, color='red', label='Actual Price', linewidth=2)
#     plt.plot(plot_dates, predicted_prices, color='blue', label='Predicted Price', linewidth=2, alpha=0.8)
#     plt.title('VODA Stock Price Prediction (Optimized Model)', fontsize=14, fontweight='bold')
#     plt.xlabel('Date')
#     plt.ylabel('Stock Price')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
    
#     # Error distribution
#     plt.subplot(2, 2, 2)
#     errors = actual_prices - predicted_prices
#     plt.hist(errors, bins=30, alpha=0.7, color='green', edgecolor='black')
#     plt.title('Prediction Error Distribution')
#     plt.xlabel('Prediction Error')
#     plt.ylabel('Frequency')
#     plt.grid(True, alpha=0.3)
    
#     # Scatter plot
#     plt.subplot(2, 2, 3)
#     plt.scatter(actual_prices, predicted_prices, alpha=0.6, color='purple')
#     plt.plot([actual_prices.min(), actual_prices.max()], 
#              [actual_prices.min(), actual_prices.max()], 'r--', lw=2)
#     plt.title('Actual vs Predicted Prices')
#     plt.xlabel('Actual Price')
#     plt.ylabel('Predicted Price')
#     plt.grid(True, alpha=0.3)
    
#     # Training history
#     plt.subplot(2, 2, 4)
#     plt.plot(history.history['loss'], label='Training Loss')
#     plt.plot(history.history['val_loss'], label='Validation Loss')
#     plt.title('Model Training History')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
    
#     plt.tight_layout()
#     plt.savefig('voda_prediction_optimized.png', dpi=300, bbox_inches='tight')
#     plt.show()
#     print("\nOptimized prediction plots saved as 'voda_prediction_optimized.png'")

# plot_results(actual_prices, predicted_prices, data_enhanced, train_size, val_size, optimal_time_step)

# # --- 9. Enhanced Future Prediction ---

# def predict_future_prices(model, data_scaled, scaler, time_step, feature_count, days_ahead=5):
#     """Predict multiple days ahead with confidence intervals"""
#     print(f"\nPredicting next {days_ahead} days...")
    
#     predictions = []
#     current_sequence = data_scaled[-time_step:].copy()
    
#     for day in range(days_ahead):
#         # Reshape for prediction
#         next_input = current_sequence.reshape(1, time_step, feature_count)
        
#         # Predict next day
#         next_price_scaled = model.predict(next_input, verbose=0)
        
#         # Inverse transform
#         dummy_array = np.zeros((1, feature_count))
#         dummy_array[:, 0] = next_price_scaled.flatten()
#         next_price = scaler.inverse_transform(dummy_array)[:, 0]
        
#         predictions.append(next_price[0])
        
#         # Update sequence for next prediction
#         # Create new row with predicted price and last known values for other features
#         new_row = current_sequence[-1].copy()
#         new_row[0] = next_price_scaled[0, 0]  # Update price
        
#         # Shift sequence and add new prediction
#         current_sequence = np.vstack([current_sequence[1:], new_row])
    
#     return predictions

# future_predictions = predict_future_prices(
#     model, train_data_scaled, scaler, optimal_time_step, 
#     len(available_features), days_ahead=5
# )

# # Display future predictions
# last_date = data_enhanced.index[-1]
# print(f"\nFuture Price Predictions:")
# print(f"Current Price: {data_enhanced['Price'].iloc[-1]:.2f}")
# print("-" * 40)

# for i, pred_price in enumerate(future_predictions, 1):
#     future_date = last_date + pd.Timedelta(days=i)
#     print(f"Day {i} ({future_date.strftime('%Y-%m-%d')}): {pred_price:.2f}")

# # Calculate prediction confidence
# recent_errors = np.abs(actual_prices[-30:] - predicted_prices[-30:])  # Last 30 predictions
# confidence_interval = np.std(recent_errors) * 1.96  # 95% confidence interval

# print(f"\n95% Confidence Interval: ±{confidence_interval:.2f}")
# print(f"Model shows {'High' if confidence_interval < data_enhanced['Price'].iloc[-1] * 0.05 else 'Moderate'} confidence in predictions")

# print(f"\nOptimization Complete! Model saved as 'best_voda_model.h5'")


























# import os
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
# from tensorflow.keras.callbacks import EarlyStopping
# import matplotlib.pyplot as plt

# # --- 1. Data Loading and Preprocessing ---

# data_folder = 'VI Data'
# file_names = [
#     os.path.join(data_folder, '2015-2016.csv'),
#     os.path.join(data_folder, '2016-2017.csv'),
#     os.path.join(data_folder, '2017-2018.csv'),
#     os.path.join(data_folder, '2018-2019.csv'),
#     os.path.join(data_folder, '2019-2020.csv'),
#     os.path.join(data_folder, '2020-2021.csv'),
#     os.path.join(data_folder, '2021-2022.csv'),
#     os.path.join(data_folder, '2022-2023.csv'),
#     os.path.join(data_folder, '2023-2024.csv'),
#     os.path.join(data_folder, '2024-2025.csv')
# ]

# try:
#     all_dfs = [pd.read_csv(filename) for filename in file_names]
#     data = pd.concat(all_dfs, ignore_index=True)
#     data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
#     data.sort_values('Date', inplace=True)
#     data.set_index('Date', inplace=True)
#     print(f"Data loaded successfully. Shape after loading: {data.shape}")
# except FileNotFoundError as e:
#     print(f"Error: {e}. Please make sure all CSV files are in the 'VI Data' folder.")
#     exit()

# # --- 2. Feature Engineering & Cleaning ---

# # NEW: More robust function to handle any conversion errors
# def convert_volume_to_float(vol_str):
#     try:
#         if isinstance(vol_str, str):
#             vol_str = vol_str.strip()
#             if vol_str.endswith('B'):
#                 return float(vol_str[:-1]) * 1_000_000_000
#             elif vol_str.endswith('M'):
#                 return float(vol_str[:-1]) * 1_000_000
#             elif vol_str.endswith('K'):
#                 return float(vol_str[:-1]) * 1_000
#         return float(vol_str)
#     except (ValueError, TypeError):
#         # If any conversion fails, return NaN so we can drop it later
#         return np.nan

# data['Vol.'] = data['Vol.'].apply(convert_volume_to_float)

# # NEW: Debugging print to see how many rows will be dropped
# print(f"Number of rows with invalid data to be dropped: {data.isna().any(axis=1).sum()}")
# data.dropna(inplace=True)
# print(f"Shape after cleaning and dropping invalid rows: {data.shape}")

# features = data[['Price', 'Vol.']].values

# # --- 3. Data Splitting & Scaling ---

# train_size = int(len(features) * 0.8)
# train_data, test_data = features[:train_size], features[train_size:]

# scaler = MinMaxScaler(feature_range=(0, 1))
# train_data_scaled = scaler.fit_transform(train_data)
# test_data_scaled = scaler.transform(test_data)

# # --- 4. Create Sequences for LSTM ---

# def create_sequences(data, time_step=60):
#     X, y = [], []
#     for i in range(len(data) - time_step):
#         X.append(data[i:(i + time_step)])
#         y.append(data[i + time_step, 0])
#     return np.array(X), np.array(y)

# time_step = 180
# X_train, y_train = create_sequences(train_data_scaled, time_step)
# X_test, y_test = create_sequences(test_data_scaled, time_step)

# # NEW: Debugging prints and check for empty data
# print(f"Shape of X_train: {X_train.shape}")
# print(f"Shape of X_test: {X_test.shape}")

# if X_train.shape[0] == 0 or X_test.shape[0] == 0:
#     print("\nError: Training or testing data has become empty. Cannot build the model.")
#     print("This might be because the dataset is too small for the chosen time_step of 180.")
#     exit()


# # --- 5. Build the Bidirectional LSTM Model ---

# model = Sequential()
# model.add(Bidirectional(LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]))))
# model.add(Dropout(0.2))
# model.add(Bidirectional(LSTM(units=64, return_sequences=False)))
# model.add(Dropout(0.2))
# model.add(Dense(units=25))
# model.add(Dense(units=1))

# model.compile(optimizer='adam', loss='mean_squared_error')

# # Build the model explicitly before calling summary
# model.build(input_shape=(None, X_train.shape[1], X_train.shape[2]))
# model.summary()  # This will now work without error

# # --- 6. Train with Early Stopping ---

# print("\nTraining the improved model...")
# early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# history = model.fit(
#     X_train, y_train,
#     epochs=100,
#     batch_size=32,
#     validation_data=(X_test, y_test),
#     callbacks=[early_stopping],
#     verbose=1
# )
# print("Model training complete.")

# # --- 7. Evaluation and Prediction ---

# print("\nEvaluating the model and making predictions...")
# predicted_prices_scaled = model.predict(X_test)

# dummy_array_for_inverse = np.zeros((len(predicted_prices_scaled), 2))
# dummy_array_for_inverse[:, 0] = predicted_prices_scaled.flatten()
# predicted_prices = scaler.inverse_transform(dummy_array_for_inverse)[:, 0]

# actual_prices = scaler.inverse_transform(test_data_scaled)[:,0][time_step:]

# # --- 8. Visualize the Results ---

# print("Visualizing the results...")
# plt.figure(figsize=(14, 7))
# plot_dates = data.index[train_size + time_step:]
# plt.plot(plot_dates, actual_prices, color='red', label='Actual VODA Stock Price')
# plt.plot(plot_dates, predicted_prices, color='blue', label='Predicted VODA Stock Price')
# plt.title('VODA Stock Price Prediction (Improved Model)')
# plt.xlabel('Date')
# plt.ylabel('Stock Price')
# plt.legend()
# plt.grid(True)
# plt.savefig('voda_prediction_improved.png')
# print("\nImproved prediction plot saved as 'voda_prediction_improved.png'")

# # --- 9. Predict the Next Day's Price with Date ---

# print("\nPredicting the next day's stock price...")
# full_data_scaled = scaler.transform(features)
# last_sequence = full_data_scaled[-time_step:]
# next_day_input = np.reshape(last_sequence, (1, time_step, 2))

# next_day_price_scaled = model.predict(next_day_input)

# dummy_for_next_day = np.zeros((1, 2))
# dummy_for_next_day[:, 0] = next_day_price_scaled.flatten()
# next_day_price = scaler.inverse_transform(dummy_for_next_day)[:, 0]

# last_date = data.index[-1]
# prediction_date = last_date + pd.Timedelta(days=1)

# print(f"\nPredicted price for {prediction_date.strftime('%Y-%m-%d')} is: {next_day_price[0]:.2f}")