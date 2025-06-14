import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class ModelFusion:
    def __init__(self, model_path='models/'):
        try:
            self.rf_model = joblib.load(f'{model_path}random_forest_multioutput_model.pkl')
            self.xgb_model = joblib.load(f'{model_path}xgboost_multioutput_model.pkl')
            self.scaler = joblib.load(f'{model_path}scaler.pkl')
        except FileNotFoundError as e:
            print(f"Model files not found: {e}")
            print("Please ensure model files are in the models/ directory")
            raise
        
        self.target_features = ['pH', 'TDS', 'Temperature', 'DO', 'Salinity', 'Turbidity']
        self.feature_columns = self.scaler.feature_names_in_

    def _validate_and_clean_data(self, df):
        """Validate and clean input data"""
        df = df.copy()
        
        # Ensure all target features exist
        for feature in self.target_features:
            if feature not in df.columns:
                print(f"Warning: {feature} not found in data")
                # Add default values if missing
                default_values = {
                    'pH': 8.0, 'TDS': 7000, 'Temperature': 30.0,
                    'DO': 5.0, 'Salinity': 15.0, 'Turbidity': 25.0
                }
                df[feature] = default_values.get(feature, 0)
        
        # Remove outliers and invalid values
        bounds = {
            'pH': (6.5, 9.5),
            'TDS': (1000, 15000),
            'Temperature': (20, 40),
            'DO': (2.0, 10.0),
            'Salinity': (5, 30),
            'Turbidity': (1, 100)
        }
        
        for feature, (min_val, max_val) in bounds.items():
            if feature in df.columns:
                df[feature] = np.clip(df[feature], min_val, max_val)
        
        return df

    def _predict_suitability(self, water_params):
        """Predict suitability based on water quality parameters"""
        ph = water_params.get('pH', 8.0)
        tds = water_params.get('TDS', 7000)
        temp = water_params.get('Temperature', 30.0)
        do = water_params.get('DO', 5.0)
        salinity = water_params.get('Salinity', 15.0)
        turbidity = water_params.get('Turbidity', 25.0)
        
        # Define optimal ranges for prawn farming
        suitable = True
        
        if ph < 7.5 or ph > 8.5:
            suitable = False
        if tds < 5000 or tds > 8000:
            suitable = False
        if temp < 28 or temp > 32:
            suitable = False
        if do < 4.0 or do > 7.0:
            suitable = False
        if salinity < 10 or salinity > 20:
            suitable = False
        if turbidity > 30:
            suitable = False
        
        return 1 if suitable else 0

    def _feature_engineering(self, df):
        df = df.copy()
        
        # Validate and clean data first
        df = self._validate_and_clean_data(df)
        
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
            first_row = df.iloc[0].copy()
            padding_rows = []
            for i in range(3 - len(df)):
                padded_row = first_row.copy()
                padded_row['Date'] = first_row['Date'] - pd.Timedelta(days=i+1)
                padded_row['day_of_week'] = padded_row['Date'].dt.dayofweek
                padded_row['month'] = padded_row['Date'].dt.month
                padding_rows.append(padded_row)

            if padding_rows:
                padding_df = pd.DataFrame(padding_rows)
                df = pd.concat([padding_df, df], ignore_index=True)
                df = df.sort_values('Date').reset_index(drop=True)

        # Create lag features
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

        # Fill any remaining NaN values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Final fallback for any remaining NaN
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64'] and df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())

        return df

    def predict_future(self, historical_df, days=5, fusion_method='average', weights=None):
        """Predict future values with enhanced error handling"""
        df = historical_df.copy()
        
        if df.empty:
            raise ValueError("Input DataFrame is empty")
        
        if 'Date' not in df.columns:
            raise ValueError("Input DataFrame must contain 'Date' column")

        # Validate that we have the required target features
        missing_features = [col for col in self.target_features if col not in df.columns]
        if missing_features:
            print(f"Warning: Missing features {missing_features} will be estimated")
            # Add missing features with default values
            for feature in missing_features:
                default_values = {
                    'pH': 8.0, 'TDS': 7000, 'Temperature': 30.0,
                    'DO': 5.0, 'Salinity': 15.0, 'Turbidity': 25.0
                }
                df[feature] = default_values.get(feature, 0)

        last_date = pd.to_datetime(df['Date']).max()
        predictions = []

        # Set random seed for reproducible variation
        np.random.seed(42)

        for i in range(days):
            try:
                # Feature engineering for current state
                current_df = self._feature_engineering(df)

                # Ensure all required features are present
                missing_cols = [col for col in self.feature_columns if col not in current_df.columns]
                if missing_cols:
                    print(f"Warning: Missing engineered features: {missing_cols}")
                    # Add missing columns with zeros or median values
                    for col in missing_cols:
                        current_df[col] = 0

                # Get last available data point
                X = current_df[self.feature_columns].tail(1)
                
                # Handle any remaining NaN values
                X = X.fillna(0)

                # Scale and predict
                scaled = self.scaler.transform(X)
                rf_pred = self.rf_model.predict(scaled)[0]
                xgb_pred = self.xgb_model.predict(scaled)[0]

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
                    'pH': 0.05, 'TDS': 100, 'Temperature': 0.3,
                    'DO': 0.15, 'Salinity': 0.5, 'Turbidity': 2.0
                }

                # Create new row with predictions
                # FIX: Use incremental date calculation
                new_date = last_date + pd.Timedelta(days=1)  # Always add 1 day from last_date
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
                        'pH': (7.0, 9.5), 'TDS': (3000, 12000), 'Temperature': (25, 35),
                        'DO': (3.0, 8.0), 'Salinity': (8, 25), 'Turbidity': (5, 50)
                    }
                    
                    min_val, max_val = bounds[col]
                    new_row[col] = np.clip(predicted_value, min_val, max_val)

                # Add other required columns
                new_row['month'] = new_date.month
                new_row['Pond No'] = df['Pond No'].iloc[-1] if 'Pond No' in df.columns else 0
                
                # FIX: Predict suitability based on water quality parameters
                predicted_suitability = self._predict_suitability(new_row)
                new_row['suitability'] = predicted_suitability

                # Append to df for recursive feature generation
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                predictions.append(new_row)

                # FIX: Update last_date for next iteration to ensure consecutive dates
                last_date = new_date

            except Exception as e:
                print(f"Error in prediction step {i+1}: {e}")
                # Create a fallback prediction based on last known values
                new_date = last_date + pd.Timedelta(days=1)  # FIX: Consecutive fallback dates
                new_row = {'Date': new_date}
                
                for col in self.target_features:
                    if col in df.columns:
                        new_row[col] = df[col].iloc[-1]  # Use last known value
                
                new_row['month'] = new_date.month
                new_row['Pond No'] = df['Pond No'].iloc[-1] if 'Pond No' in df.columns else 0
                new_row['suitability'] = self._predict_suitability(new_row)
                
                predictions.append(new_row)
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                last_date = new_date  # FIX: Update last_date even in error case

        return pd.DataFrame(predictions)
