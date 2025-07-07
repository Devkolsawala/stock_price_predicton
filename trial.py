# =============================================================================
# 1. IMPORT LIBRARIES
# =============================================================================
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# 2. DATA LOADING AND PREPROCESSING
# =============================================================================
# Define the folder containing the CSV files
data_folder = 'VI Data'
# List of CSV files in chronological order
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

try:
    # Load and concatenate all data files
    all_dfs = [pd.read_csv(filename) for filename in file_names]
    data = pd.concat(all_dfs, ignore_index=True)
    
    # Convert 'Date' column to datetime objects
    data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
    # Sort data by date to ensure chronological order
    data.sort_values('Date', inplace=True)
    data.set_index('Date', inplace=True)
    print(f"Data loaded successfully. Total records: {len(data)}")

except FileNotFoundError as e:
    print(f"Error: {e}. Ensure all CSV files are in the '{data_folder}' folder.")
    exit()

# Helper function to convert volume strings (e.g., '100M', '2.5B') to float
def convert_volume_to_float(vol_str):
    try:
        if isinstance(vol_str, str):
            vol_str = vol_str.strip().upper()
            if vol_str.endswith('B'):
                return float(vol_str[:-1]) * 1_000_000_000
            elif vol_str.endswith('M'):
                return float(vol_str[:-1]) * 1_000_000
            elif vol_str.endswith('K'):
                return float(vol_str[:-1]) * 1_000
        return float(vol_str)
    except (ValueError, TypeError):
        return np.nan

# Apply volume conversion and clean percentage column
data['Vol.'] = data['Vol.'].apply(convert_volume_to_float)
if 'Change %' in data.columns:
    data['Change %'] = data['Change %'].str.replace('%', '').astype(float)

# Drop any rows with missing values
data.dropna(inplace=True)
print(f"Data shape after cleaning: {data.shape}")

# =============================================================================
# 3. FEATURE ENGINEERING
# =============================================================================
def create_technical_indicators(df):
    """Adds technical indicators to the dataframe."""
    df_tech = df.copy()
    
    # Moving Averages
    df_tech['MA_20'] = df_tech['Price'].rolling(window=20).mean()
    df_tech['MA_50'] = df_tech['Price'].rolling(window=50).mean()
    
    # Relative Strength Index (RSI)
    delta = df_tech['Price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_tech['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = df_tech['Price'].ewm(span=12, adjust=False).mean()
    ema_26 = df_tech['Price'].ewm(span=26, adjust=False).mean()
    df_tech['MACD'] = ema_12 - ema_26
    
    # Bollinger Bands
    df_tech['BB_Middle'] = df_tech['Price'].rolling(window=20).mean()
    bb_std = df_tech['Price'].rolling(window=20).std()
    df_tech['BB_Upper'] = df_tech['BB_Middle'] + (bb_std * 2)
    df_tech['BB_Lower'] = df_tech['BB_Middle'] - (bb_std * 2)
    
    # Lag Features
    df_tech['Price_Lag_1'] = df_tech['Price'].shift(1)
    
    # Drop rows with NaN values created by indicators
    df_tech.dropna(inplace=True)
    return df_tech

print("Adding technical indicators...")
data_enhanced = create_technical_indicators(data)
print(f"Data shape after feature engineering: {data_enhanced.shape}")

# Define the feature set for the model
# The target variable 'Price' must be the first column for easy inverse scaling later
feature_columns = [
    'Price', 'Vol.', 'MA_20', 'MA_50', 'RSI', 'MACD', 'BB_Upper', 'BB_Lower', 'Price_Lag_1'
]
# Ensure all selected columns are in the dataframe
available_features = [col for col in feature_columns if col in data_enhanced.columns]
features_df = data_enhanced[available_features]

# =============================================================================
# 4. DATA SCALING AND SPLITTING
# =============================================================================
# Scale features to the range [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(features_df)

# Split data into training (80%) and testing (20%) sets
train_size = int(len(scaled_features) * 0.8)
train_data = scaled_features[:train_size]
test_data = scaled_features[train_size:]

# =============================================================================
# 5. CREATE SEQUENCES FOR LSTM
# =============================================================================
def create_sequences(data, time_step=60):
    """Creates input sequences (X) and output values (y) for the LSTM model."""
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step)])
        y.append(data[i + time_step, 0]) # The target is the 'Price' column (index 0)
    return np.array(X), np.array(y)

TIME_STEP = 60
X_train, y_train = create_sequences(train_data, TIME_STEP)
X_test, y_test = create_sequences(test_data, TIME_STEP)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# =============================================================================
# 6. BUILD AND TRAIN THE OPTIMIZED LSTM MODEL
# =============================================================================
def build_optimized_model(input_shape):
    """Builds, compiles, and returns the optimized LSTM model."""
    model = Sequential([
        LSTM(units=100, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=100, return_sequences=False),
        Dropout(0.2),
        Dense(units=50, activation='relu'),
        Dense(units=1) # Output layer for regression
    ])
    
    # Compile with Adam optimizer and Mean Squared Error loss
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# Build the model
model = build_optimized_model((X_train.shape[1], X_train.shape[2]))
model.summary()

# Define callbacks for training
callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ModelCheckpoint('voda_optimized_model.h5', monitor='val_loss', save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
]

print("\nTraining the optimized model...")
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.1, # Use part of training data for validation
    callbacks=callbacks,
    verbose=1
)

# =============================================================================
# 7. EVALUATE THE MODEL AND VISUALIZE RESULTS
# =============================================================================
print("\nEvaluating model performance...")
# Make predictions on the test set
predicted_scaled = model.predict(X_test)

# To inverse transform the predictions, we need to create a dummy array
# with the same shape as the number of features, then replace the first column
# with our scaled predictions.
dummy_array_pred = np.zeros((len(predicted_scaled), len(available_features)))
dummy_array_pred[:, 0] = predicted_scaled.flatten()
predicted_prices = scaler.inverse_transform(dummy_array_pred)[:, 0]

# Inverse transform the actual test prices for comparison
dummy_array_actual = np.zeros((len(y_test), len(available_features)))
dummy_array_actual[:, 0] = y_test.flatten()
actual_prices = scaler.inverse_transform(dummy_array_actual)[:, 0]

# Calculate performance metrics
rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
mae = mean_absolute_error(actual_prices, predicted_prices)
r2 = r2_score(actual_prices, predicted_prices)
mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100

print("\nModel Performance Metrics:")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

# --- Plotting the results ---
print("\nGenerating prediction plot...")
plt.style.use('default')
plt.figure(figsize=(15, 7))

# Get the dates for the x-axis of the plot
plot_dates = features_df.index[train_size + TIME_STEP:]

plt.plot(plot_dates, actual_prices, color='red', label='Actual Price', linewidth=2)
plt.plot(plot_dates, predicted_prices, color='blue', label='Predicted Price', linewidth=2, alpha=0.8)
plt.title('VODA Stock Price Prediction (Optimized Model)', fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Stock Price', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('voda_prediction_optimized.png', dpi=300)
plt.show()

print("\nOptimization complete. Model saved as 'voda_optimized_model_trial.h5'")