import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class ModelFusion:
    def __init__(self, model_path='models/'):
        self.rf_model = joblib.load(f'{model_path}random_forest_multioutput_model.pkl')
        self.xgb_model = joblib.load(f'{model_path}xgboost_multioutput_model.pkl')
        self.scaler = joblib.load(f'{model_path}scaler.pkl')
        self.target_features = ['pH', 'TDS', 'Temperature', 'DO', 'Salinity', 'Turbidity']
        self.feature_columns = self.scaler.feature_names_in_

    def _feature_engineering(self, df):
        df = df.copy()
        
        # Ensure Date is datetime and sorted
        if 'Date' not in df.columns:
            raise ValueError("Input DataFrame must contain 'Date' column")
        
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Extract temporal features
        df['day_of_week'] = df['Date'].dt.dayofweek
        if 'month' not in df.columns:
            df['month'] = df['Date'].dt.month
        
        # Ensure we have enough data for lag features
        if len(df) < 3:
            # Pad with the first row to create minimum required history
            first_row = df.iloc[0].copy()
            padding_rows = []
            for i in range(3 - len(df)):
                padded_row = first_row.copy()
                # Adjust date backwards
                padded_row['Date'] = first_row['Date'] - pd.Timedelta(days=i+1)
                padded_row['day_of_week'] = padded_row['Date'].dt.dayofweek
                padded_row['month'] = padded_row['Date'].dt.month
                padding_rows.append(padded_row)
            
            # Add padding rows at the beginning
            if padding_rows:
                padding_df = pd.DataFrame(padding_rows)
                df = pd.concat([padding_df, df], ignore_index=True)
                df = df.sort_values('Date').reset_index(drop=True)
        
        # Create lag features with EXACT naming from training
        for col in self.target_features:
            if col in df.columns:
                df[f'{col}_t-1'] = df[col].shift(1)
                df[f'{col}_t-2'] = df[col].shift(2)
                df[f'{col}_t-3'] = df[col].shift(3)
                
                # Rolling features
                df[f'{col}_rolling_mean_3'] = df[col].rolling(window=3, min_periods=1).mean()
                df[f'{col}_rolling_std_3'] = df[col].rolling(window=3, min_periods=1).std()
        
        # Interaction features
        if 'TDS' in df.columns and 'Salinity' in df.columns:
            df['TDS_Salinity'] = df['TDS'] * df['Salinity']
        if 'DO' in df.columns and 'Temperature' in df.columns:
            df['DO_Temp_Ratio'] = df['DO'] / (df['Temperature'] + 1e-3)
        if 'Turbidity' in df.columns and 'TDS' in df.columns:
            df['Turb_TDS_Ratio'] = df['Turbidity'] / (df['TDS'] + 1e-3)
        
        # Cyclical month encoding
        if 'month' in df.columns:
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Fill any remaining NaN values with forward fill, then backward fill
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df

    def predict_future(self, historical_df, days=5, fusion_method='average', weights=None):
        df = historical_df.copy()
        
        if 'Date' not in df.columns:
            raise ValueError("Input DataFrame must contain 'Date' column")
        
        # Ensure we have the required target features
        missing_features = [col for col in self.target_features if col not in df.columns]
        if missing_features:
            raise ValueError(f"Missing required features in input data: {missing_features}")
        
        last_date = pd.to_datetime(df['Date']).max()
        predictions = []
        
        # Add controlled randomness for variation
        np.random.seed(42)
        
        for i in range(days):
            # Feature engineering for current state
            current_df = self._feature_engineering(df)
            
            # Ensure all required features are present
            missing_cols = [col for col in self.feature_columns if col not in current_df.columns]
            if missing_cols:
                raise ValueError(f"Missing features after engineering: {missing_cols}")
            
            # Get last available data point
            X = current_df[self.feature_columns].tail(1)
            
            if X.empty or X.isnull().any().any():
                # Handle case where we still have NaN values
                X = X.fillna(X.mean())
                if X.isnull().any().any():
                    # If still NaN, use median values from training data
                    for col in X.columns:
                        if X[col].isnull().any():
                            X[col] = 0  # Default fallback
            
            # Scale and predict
            try:
                scaled = self.scaler.transform(X)
                rf_pred = self.rf_model.predict(scaled)[0]
                xgb_pred = self.xgb_model.predict(scaled)[0]
            except Exception as e:
                raise ValueError(f"Prediction failed: {str(e)}. Check if input data matches training format.")
            
            # Fusion logic
            if fusion_method == 'weighted_average' and weights:
                fused_pred = weights[0] * rf_pred + weights[1] * xgb_pred
            elif fusion_method == 'max':
                fused_pred = np.maximum(rf_pred, xgb_pred)
            elif fusion_method == 'min':
                fused_pred = np.minimum(rf_pred, xgb_pred)
            else:
                fused_pred = (rf_pred + xgb_pred) / 2
            
            # Add realistic variation
            variation_factors = {
                'pH': 0.05,
                'TDS': 100,
                'Temperature': 0.3,
                'DO': 0.15,
                'Salinity': 0.5,
                'Turbidity': 2.0
            }
            
            # Create new row with predictions
            new_date = last_date + pd.Timedelta(days=i+1)
            new_row = {'Date': new_date}
            
            for idx, col in enumerate(self.target_features):
                base_value = fused_pred[idx]
                
                # Add trend-based variation
                if len(df) >= 3:
                    recent_values = df[col].tail(3).values
                    trend = np.polyfit(range(3), recent_values, 1)[0]
                    trend_factor = trend * (i + 1) * 0.3
                else:
                    trend_factor = 0
                
                # Add seasonal/cyclical variation
                seasonal_factor = 0.02 * np.sin(2 * np.pi * (i + 1) / 7) * variation_factors[col]
                
                # Add controlled random noise
                noise = np.random.normal(0, variation_factors[col] * 0.1)
                
                # Combine all factors
                predicted_value = base_value + trend_factor + seasonal_factor + noise
                
                # Apply realistic bounds
                bounds = {
                    'pH': (7.0, 9.5),
                    'TDS': (3000, 12000),
                    'Temperature': (25, 35),
                    'DO': (3.0, 8.0),
                    'Salinity': (8, 25),
                    'Turbidity': (5, 50)
                }
                
                min_val, max_val = bounds[col]
                new_row[col] = np.clip(predicted_value, min_val, max_val)
            
            # Ensure month is set correctly
            new_row['month'] = new_row['Date'].month
            
            # Append to df for recursive feature generation
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            predictions.append(new_row)
            
            # Update last_date for next iteration
            last_date = new_row['Date']
        
        return pd.DataFrame(predictions)
