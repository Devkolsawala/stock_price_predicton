import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import joblib

# Fetch data
start_date = '2005-05-16'
end_date = '2025-07-11'
nifty_data = yf.download('^NSEI', start=start_date, end=end_date)

# Add technical indicators
nifty_data['MA50'] = nifty_data['Close'].rolling(50).mean()
nifty_data['MA200'] = nifty_data['Close'].rolling(200).mean()
nifty_data['EMA12'] = nifty_data['Close'].ewm(span=12, adjust=False).mean()
nifty_data['EMA26'] = nifty_data['Close'].ewm(span=26, adjust=False).mean()
nifty_data['MACD'] = nifty_data['EMA12'] - nifty_data['EMA26']
nifty_data['Signal'] = nifty_data['MACD'].ewm(span=9, adjust=False).mean()

# RSI
delta = nifty_data['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
nifty_data['RSI'] = 100 - (100 / (1 + rs))

# Drop NaN
nifty_data = nifty_data.dropna()

features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA50', 'MA200', 'MACD', 'Signal', 'RSI']
data = nifty_data[features].values

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Prepare datasets
time_step = 100
train_size = int(len(scaled_data) * 0.8)
train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

def create_dataset(dataset, time_step=100):
    X, y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), :])
        y.append(dataset[i + time_step, 3])  # Close index 3
    return np.array(X), np.array(y)

X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Build model
model = Sequential()
model.add(Bidirectional(LSTM(128, return_sequences=True, input_shape=(time_step, len(features)))))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(128)))
model.add(Dropout(0.2))
model.add(Dense(50))
model.add(Dense(1))
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.2, callbacks=[early_stop])

# Save
model.save('nifty_model.h5')
joblib.dump(scaler, 'scaler.pkl')

# Predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform
inv_train_predict = np.concatenate((np.zeros((len(train_predict), 3)), train_predict, np.zeros((len(train_predict), 6))), axis=1)
inv_train_predict = scaler.inverse_transform(inv_train_predict)[:, 3]

inv_test_predict = np.concatenate((np.zeros((len(test_predict), 3)), test_predict, np.zeros((len(test_predict), 6))), axis=1)
inv_test_predict = scaler.inverse_transform(inv_test_predict)[:, 3]

# Plot
close_prices = nifty_data['Close'].values
train_predict_plot = np.full(len(close_prices), np.nan)
train_predict_plot[time_step:time_step + len(train_predict)] = inv_train_predict

test_predict_plot = np.full(len(close_prices), np.nan)
test_predict_plot[train_size + time_step:train_size + time_step + len(test_predict)] = inv_test_predict

plt.plot(close_prices)
plt.plot(train_predict_plot, 'r')
plt.plot(test_predict_plot, 'g')
plt.show()