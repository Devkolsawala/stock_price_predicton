## code made with the help of claude


import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import joblib
import logging
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class NIFTYPredictor:
    def __init__(self, config=None):
        """
        Initialize NIFTY stock predictor with configurable parameters
        """
        # Start with default config and update with custom config
        self.config = self._default_config()
        if config:
            self.config.update(config)
        
        self.model = None
        self.scaler = None
        self.data = None
        self.features = None
        self.setup_logging()
        
    def _default_config(self):
        """Default configuration parameters"""
        return {
            'start_date': '2005-05-16',
            'end_date': '2025-07-10',
            'time_step': 100,
            'train_split': 0.8,
            'validation_split': 0.2,
            'epochs': 200,
            'batch_size': 32,
            'learning_rate': 0.001,
            'early_stopping_patience': 20,
            'lstm_units': [128, 128, 128],
            'dropout_rate': 0.2,
            'dense_units': 50,
            'model_path': 'models/nifty_model.h5',
            'scaler_path': 'models/scaler.pkl',
            'log_path': 'logs/nifty_prediction.log'
        }
    
    def setup_logging(self):
        """Setup logging configuration"""
        os.makedirs(os.path.dirname(self.config['log_path']), exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config['log_path']),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def fetch_data(self):
        """
        Fetch NIFTY data with error handling and proper column handling
        """
        try:
            self.logger.info(f"Fetching NIFTY data from {self.config['start_date']} to {self.config['end_date']}")
            data = yf.download('^NSEI', start=self.config['start_date'], end=self.config['end_date'])
            
            if data.empty:
                raise ValueError("No data fetched. Check date range and internet connection.")
            
            # Fix MultiIndex columns issue
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)
            
            # Ensure we have the required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Remove any rows with NaN values in basic OHLCV data
            data = data.dropna(subset=required_cols)
            
            self.logger.info(f"Successfully fetched {len(data)} data points")
            self.logger.info(f"Data columns: {list(data.columns)}")
            self.logger.info(f"Data shape: {data.shape}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching data: {str(e)}")
            raise
    
    def add_technical_indicators(self, data):
        """
        Add comprehensive technical indicators with proper error handling
        """
        self.logger.info("Adding technical indicators...")
        
        # Create a copy to avoid modifying original data
        data = data.copy()
        
        try:
            # Moving Averages
            data['MA50'] = data['Close'].rolling(window=50, min_periods=1).mean()
            data['MA200'] = data['Close'].rolling(window=200, min_periods=1).mean()
            
            # Exponential Moving Averages
            data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
            data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
            
            # MACD
            data['MACD'] = data['EMA12'] - data['EMA26']
            data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
            data['MACD_Histogram'] = data['MACD'] - data['Signal']
            
            # RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            data['BB_Middle'] = data['Close'].rolling(window=20, min_periods=1).mean()
            bb_std = data['Close'].rolling(window=20, min_periods=1).std()
            data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
            data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
            
            # Volume indicators
            data['Volume_MA'] = data['Volume'].rolling(window=20, min_periods=1).mean()
            data['Volume_Ratio'] = data['Volume'] / data['Volume_MA']
            
            # Price momentum
            data['Price_Change'] = data['Close'].pct_change()
            data['Price_Change_MA'] = data['Price_Change'].rolling(window=10, min_periods=1).mean()
            
            # Replace infinities and NaN values
            data = data.replace([np.inf, -np.inf], np.nan)
            
            # Fill NaN values with forward fill, then backward fill
            data = data.fillna(method='ffill').fillna(method='bfill')
            
            self.logger.info("Technical indicators added successfully")
            self.logger.info(f"Data shape after adding indicators: {data.shape}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error adding technical indicators: {str(e)}")
            raise
    
    def preprocess_data(self, data):
        """
        Preprocess data for model training
        """
        self.logger.info("Preprocessing data...")
        
        # Define features
        self.features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA50', 'MA200', 
                        'MACD', 'Signal', 'MACD_Histogram', 'RSI', 'BB_Upper', 'BB_Lower', 
                        'Volume_Ratio', 'Price_Change', 'Price_Change_MA']
        
        # Check if all features exist
        missing_features = [f for f in self.features if f not in data.columns]
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        # Drop any remaining NaN values
        data_clean = data.dropna()
        
        if len(data_clean) == 0:
            raise ValueError("No data remaining after cleaning")
        
        self.logger.info(f"Data points after cleaning: {len(data_clean)}")
        
        # Prepare feature matrix
        feature_data = data_clean[self.features].values
        
        # Check for infinite values
        if np.any(np.isinf(feature_data)):
            self.logger.warning("Infinite values detected in features")
            feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale data
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = self.scaler.fit_transform(feature_data)
        
        self.logger.info(f"Data preprocessed. Shape: {scaled_data.shape}")
        return scaled_data, data_clean
    
    def create_dataset(self, dataset, time_step):
        """
        Create time series dataset for LSTM
        """
        X, y = [], []
        for i in range(len(dataset) - time_step - 1):
            X.append(dataset[i:(i + time_step), :])
            y.append(dataset[i + time_step, 3])  # Close price index
        return np.array(X), np.array(y)
    
    def prepare_train_test_data(self, scaled_data):
        """
        Prepare training and testing datasets
        """
        self.logger.info("Preparing train/test datasets...")
        
        time_step = self.config['time_step']
        train_size = int(len(scaled_data) * self.config['train_split'])
        
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size:]
        
        X_train, y_train = self.create_dataset(train_data, time_step)
        X_test, y_test = self.create_dataset(test_data, time_step)
        
        self.logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        return X_train, y_train, X_test, y_test, train_size
    
    def build_model(self, input_shape):
        """
        Build advanced LSTM model
        """
        self.logger.info("Building LSTM model...")
        
        model = Sequential()
        
        # First LSTM layer
        model.add(Bidirectional(LSTM(self.config['lstm_units'][0], 
                                   return_sequences=True, 
                                   input_shape=input_shape)))
        model.add(Dropout(self.config['dropout_rate']))
        
        # Second LSTM layer
        model.add(Bidirectional(LSTM(self.config['lstm_units'][1], 
                                   return_sequences=True)))
        model.add(Dropout(self.config['dropout_rate']))
        
        # Third LSTM layer
        model.add(Bidirectional(LSTM(self.config['lstm_units'][2])))
        model.add(Dropout(self.config['dropout_rate']))
        
        # Dense layers
        model.add(Dense(self.config['dense_units'], activation='relu'))
        model.add(Dropout(self.config['dropout_rate']))
        model.add(Dense(1))
        
        # Compile model
        model.compile(optimizer=Adam(learning_rate=self.config['learning_rate']), 
                     loss='mean_squared_error',
                     metrics=['mae'])
        
        self.logger.info("Model built successfully")
        return model
    
    def train_model(self, X_train, y_train):
        """
        Train the model with callbacks
        """
        self.logger.info("Starting model training...")
        
        # Build model
        self.model = self.build_model((X_train.shape[1], X_train.shape[2]))
        
        # Callbacks
        early_stop = EarlyStopping(monitor='val_loss', 
                                 patience=self.config['early_stopping_patience'],
                                 restore_best_weights=True,
                                 verbose=1)
        
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                                    factor=0.5, 
                                    patience=10, 
                                    min_lr=0.0001,
                                    verbose=1)
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            validation_split=self.config['validation_split'],
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        self.logger.info("Model training completed")
        return history
    
    def evaluate_model(self, X_test, y_test, test_predictions):
        """
        Comprehensive model evaluation
        """
        self.logger.info("Evaluating model performance...")
        
        # Flatten predictions if necessary
        test_predictions = test_predictions.flatten()
        y_test = y_test.flatten()
        
        # Calculate metrics
        mse = mean_squared_error(y_test, test_predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, test_predictions)
        r2 = r2_score(y_test, test_predictions)
        
        # Calculate MAPE (avoid division by zero)
        mask = y_test != 0
        mape = np.mean(np.abs((y_test[mask] - test_predictions[mask]) / y_test[mask])) * 100
        
        # Directional accuracy
        if len(y_test) > 1:
            actual_direction = np.diff(y_test) > 0
            predicted_direction = np.diff(test_predictions) > 0
            directional_accuracy = np.mean(actual_direction == predicted_direction) * 100
        else:
            directional_accuracy = 0
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape,
            'Directional_Accuracy': directional_accuracy
        }
        
        # Log metrics
        self.logger.info("Model Performance Metrics:")
        for metric, value in metrics.items():
            self.logger.info(f"{metric}: {value:.4f}")
        
        return metrics
    
    def inverse_transform_predictions(self, predictions):
        """
        Inverse transform predictions to original scale
        """
        # Create dummy array for inverse transformation
        dummy = np.zeros((len(predictions), len(self.features)))
        dummy[:, 3] = predictions.flatten()  # Close price column
        
        # Inverse transform
        inverse_pred = self.scaler.inverse_transform(dummy)
        return inverse_pred[:, 3]
    
    def save_model(self):
        """
        Save trained model and scaler
        """
        try:
            os.makedirs(os.path.dirname(self.config['model_path']), exist_ok=True)
            
            self.model.save(self.config['model_path'])
            joblib.dump(self.scaler, self.config['scaler_path'])
            
            self.logger.info(f"Model saved to {self.config['model_path']}")
            self.logger.info(f"Scaler saved to {self.config['scaler_path']}")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self):
        """
        Load pre-trained model and scaler
        """
        try:
            self.model = load_model(self.config['model_path'])
            self.scaler = joblib.load(self.config['scaler_path'])
            
            self.logger.info("Model and scaler loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
    
    def plot_results(self, actual_prices, train_predictions, test_predictions, train_size, time_step):
        """
        Plot training and testing results
        """
        plt.figure(figsize=(15, 8))
        
        # Plot actual prices
        plt.plot(actual_prices, label='Actual Price', color='blue', alpha=0.7)
        
        # Plot training predictions
        train_plot = np.full(len(actual_prices), np.nan)
        train_plot[time_step:time_step + len(train_predictions)] = train_predictions
        plt.plot(train_plot, label='Training Predictions', color='red', alpha=0.7)
        
        # Plot test predictions
        test_plot = np.full(len(actual_prices), np.nan)
        test_plot[train_size + time_step:train_size + time_step + len(test_predictions)] = test_predictions
        plt.plot(test_plot, label='Test Predictions', color='green', alpha=0.7)
        
        plt.title('NIFTY Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def walk_forward_validation(self, scaled_data, n_splits=5):
        """
        Perform walk-forward validation
        """
        self.logger.info(f"Performing walk-forward validation with {n_splits} splits...")
        
        data_len = len(scaled_data)
        step_size = data_len // n_splits
        metrics_list = []
        
        for i in range(n_splits):
            # Define split boundaries
            train_end = (i + 1) * step_size
            test_start = train_end
            test_end = min(test_start + step_size, data_len)
            
            if test_end - test_start < self.config['time_step'] + 1:
                continue
            
            # Prepare data for this split
            train_data = scaled_data[:train_end]
            test_data = scaled_data[test_start:test_end]
            
            if len(train_data) < self.config['time_step'] + 1:
                continue
            
            X_train, y_train = self.create_dataset(train_data, self.config['time_step'])
            X_test, y_test = self.create_dataset(test_data, self.config['time_step'])
            
            if len(X_train) == 0 or len(X_test) == 0:
                continue
            
            # Train model for this split
            fold_model = self.build_model((X_train.shape[1], X_train.shape[2]))
            fold_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
            
            # Make predictions
            fold_predictions = fold_model.predict(X_test)
            
            # Evaluate
            fold_metrics = self.evaluate_model(X_test, y_test, fold_predictions)
            metrics_list.append(fold_metrics)
        
        if not metrics_list:
            self.logger.warning("No valid splits for walk-forward validation")
            return {}
        
        # Calculate average metrics
        avg_metrics = {}
        for metric in metrics_list[0].keys():
            avg_metrics[f'avg_{metric}'] = np.mean([m[metric] for m in metrics_list])
            avg_metrics[f'std_{metric}'] = np.std([m[metric] for m in metrics_list])
        
        self.logger.info("Walk-forward validation results:")
        for metric, value in avg_metrics.items():
            self.logger.info(f"{metric}: {value:.4f}")
        
        return avg_metrics
    
    def predict_future(self, days=30):
        """
        Predict future stock prices using trained model
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained or loaded. Run training first or load existing model.")
        
        self.logger.info(f"Predicting next {days} days...")
        
        # Get the last time_step data points
        last_data = self.data.iloc[-self.config['time_step']:][self.features].values
        last_scaled = self.scaler.transform(last_data)
        
        predictions = []
        current_data = last_scaled.copy()
        
        for _ in range(days):
            # Prepare input for prediction
            X_pred = current_data.reshape(1, self.config['time_step'], len(self.features))
            
            # Make prediction
            next_pred = self.model.predict(X_pred, verbose=0)
            
            # Store prediction
            predictions.append(next_pred[0, 0])
            
            # Update current_data for next prediction
            # Create new row with prediction as Close price
            new_row = current_data[-1].copy()
            new_row[3] = next_pred[0, 0]  # Close price index
            
            # Shift data and add new prediction
            current_data = np.roll(current_data, -1, axis=0)
            current_data[-1] = new_row
        
        # Inverse transform predictions
        predictions_inv = self.inverse_transform_predictions(np.array(predictions))
        
        # Create future dates
        last_date = self.data.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                   periods=days, freq='D')
        
        # Create results dataframe
        future_predictions = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Close': predictions_inv
        })
        
        self.logger.info(f"Future predictions completed for {days} days")
        return future_predictions
    
    def test_model_on_new_data(self, start_date, end_date):
        """
        Test trained model on completely new data period
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained or loaded. Run training first or load existing model.")
        
        self.logger.info(f"Testing model on new data from {start_date} to {end_date}")
        
        # Fetch new data
        new_data = yf.download('^NSEI', start=start_date, end=end_date)
        
        if new_data.empty:
            raise ValueError("No new data fetched for testing.")
        
        # Fix MultiIndex columns if present
        if isinstance(new_data.columns, pd.MultiIndex):
            new_data.columns = new_data.columns.droplevel(1)
        
        # Add technical indicators
        new_data = self.add_technical_indicators(new_data)
        new_data = new_data.dropna()
        
        # Prepare features
        new_features = new_data[self.features].values
        new_scaled = self.scaler.transform(new_features)
        
        # Create test dataset
        X_new, y_new = self.create_dataset(new_scaled, self.config['time_step'])
        
        if len(X_new) == 0:
            raise ValueError("Not enough data for testing after preprocessing")
        
        # Make predictions
        new_predictions = self.model.predict(X_new)
        
        # Inverse transform
        new_pred_inv = self.inverse_transform_predictions(new_predictions)
        
        # Get actual prices for comparison
        actual_prices = new_data['Close'].values[self.config['time_step']+1:]
        
        # Evaluate performance
        test_metrics = self.evaluate_model(X_new, y_new, new_predictions)
        
        # Create results dataframe
        test_dates = new_data.index[self.config['time_step']+1:]
        test_results = pd.DataFrame({
            'Date': test_dates,
            'Actual_Close': actual_prices,
            'Predicted_Close': new_pred_inv
        })
        
        self.logger.info("Model testing on new data completed")
        return test_results, test_metrics
    
    def plot_predictions(self, predictions_df, title="Stock Price Predictions"):
        """
        Plot predictions vs actual prices
        """
        plt.figure(figsize=(12, 6))
        
        if 'Actual_Close' in predictions_df.columns:
            plt.plot(predictions_df['Date'], predictions_df['Actual_Close'], 
                    label='Actual Price', color='blue', linewidth=2)
        
        plt.plot(predictions_df['Date'], predictions_df['Predicted_Close'], 
                label='Predicted Price', color='red', linewidth=2)
        
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def run_complete_pipeline(self):
        """
        Run the complete prediction pipeline
        """
        try:
            # Fetch and preprocess data
            raw_data = self.fetch_data()
            processed_data = self.add_technical_indicators(raw_data)
            scaled_data, self.data = self.preprocess_data(processed_data)
            
            # Prepare datasets
            X_train, y_train, X_test, y_test, train_size = self.prepare_train_test_data(scaled_data)
            
            # Train model
            history = self.train_model(X_train, y_train)
            
            # Make predictions
            train_predictions = self.model.predict(X_train)
            test_predictions = self.model.predict(X_test)
            
            # Inverse transform predictions
            train_pred_inv = self.inverse_transform_predictions(train_predictions)
            test_pred_inv = self.inverse_transform_predictions(test_predictions)
            
            # Evaluate model
            metrics = self.evaluate_model(X_test, y_test, test_predictions)
            
            # Walk-forward validation
            wf_metrics = self.walk_forward_validation(scaled_data)
            
            # Save model
            self.save_model()
            
            # Plot results
            actual_prices = self.data['Close'].values
            self.plot_results(actual_prices, train_pred_inv, test_pred_inv, train_size, self.config['time_step'])
            
            return {
                'model': self.model,
                'metrics': metrics,
                'walk_forward_metrics': wf_metrics,
                'history': history
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    # Custom configuration (optional)
    custom_config = {
        'start_date': '2005-05-16',
        'end_date': '2025-07-10',
        'time_step': 60,
        'epochs': 100,
        'batch_size': 64,
        'learning_rate': 0.0005,
    }
    
    # Initialize and run predictor
    predictor = NIFTYPredictor(custom_config)
    results = predictor.run_complete_pipeline()
    
    print("\n" + "="*50)
    print("NIFTY PREDICTION PIPELINE COMPLETED")
    print("="*50)
    print(f"Final Test Metrics: {results['metrics']}")
    print(f"Cross-validation Metrics: {results['walk_forward_metrics']}")
    
    # Additional testing examples:
    
    # Example 1: Test on new data period
    print("\n" + "="*50)
    print("TESTING ON NEW DATA PERIOD")
    print("="*50)
    try:
        # Test on a period not used in training
        test_results, test_metrics = predictor.test_model_on_new_data('2025-01-01', '2025-07-11')
        print(f"Test Results Shape: {test_results.shape}")
        print(f"Test Metrics: {test_metrics}")
        
        # Plot test results
        predictor.plot_predictions(test_results, "Model Performance on New Data")
        
    except Exception as e:
        print(f"New data testing failed: {e}")
    
    # Example 2: Predict future prices
    print("\n" + "="*50)
    print("FUTURE PRICE PREDICTIONS")
    print("="*50)
    try:
        future_predictions = predictor.predict_future(days=30)
        print(f"Future Predictions:\n{future_predictions.head(10)}")
        
        # Plot future predictions
        predictor.plot_predictions(future_predictions, "Future Price Predictions")
        
    except Exception as e:
        print(f"Future prediction failed: {e}")
    
    # Example 3: Load pre-trained model and test
    print("\n" + "="*50)
    print("LOADING PRE-TRAINED MODEL")
    print("="*50)
    try:
        # Create new predictor instance
        new_predictor = NIFTYPredictor(custom_config)
        
        # Load pre-trained model
        new_predictor.load_model()
        
        # Need to load the data and features for predictions
        raw_data = new_predictor.fetch_data()
        processed_data = new_predictor.add_technical_indicators(raw_data)
        scaled_data, new_predictor.data = new_predictor.preprocess_data(processed_data)
        
        # Test future predictions with loaded model
        future_pred = new_predictor.predict_future(days=15)
        print(f"Future predictions with loaded model:\n{future_pred.head()}")
        
    except Exception as e:
        print(f"Loading pre-trained model failed: {e}")
