import pandas as pd
import streamlit as st
import numpy as np
from datetime import datetime, timedelta

class FirebaseHandler:
    def __init__(self):
        self.db_ref = None
        self.firebase_available = False
        self._initialize_firebase()

    def _initialize_firebase(self):
        try:
            import firebase_admin
            from firebase_admin import credentials, db

            if not firebase_admin._apps:
                cred = credentials.Certificate(dict(st.secrets["firebase_credentials"]))
                firebase_admin.initialize_app(cred, {
                    'databaseURL': st.secrets["firebase_database_url"]
                })

            self.db_ref = db.reference('/')
            self.firebase_available = True
        except Exception as e:
            print(f"Firebase initialization failed: {e}")
            self.firebase_available = False

    def _estimate_do_salinity(self, df):
        """
        Estimate DO and Salinity based on available parameters using empirical relationships
        """
        df = df.copy()
        
        # Estimate Dissolved Oxygen (DO) based on Temperature, pH, and TDS
        # DO decreases with temperature and TDS, increases with pH (within limits)
        df['DO'] = (
            8.5 - (df['Temperature'] - 25) * 0.15 +  # Temperature effect
            (df['pH'] - 7.5) * 0.3 -                 # pH effect
            (df['TDS'] / 1000) * 0.2 +               # TDS effect
            np.random.normal(0, 0.3, len(df))        # Natural variation
        )
        
        # Clamp DO to realistic range for aquaculture
        df['DO'] = np.clip(df['DO'], 3.0, 8.0)
        
        # Estimate Salinity based on TDS and Temperature
        # Salinity is roughly related to TDS but not linearly
        df['Salinity'] = (
            (df['TDS'] / 500) +                      # Base relationship
            (df['Temperature'] - 30) * 0.1 +         # Temperature effect
            np.random.normal(0, 1.5, len(df))        # Natural variation
        )
        
        # Clamp Salinity to realistic range for prawn farming
        df['Salinity'] = np.clip(df['Salinity'], 8, 25)
        
        return df

    def fetch_and_process_iot_data(self, days_back=30):
        """
        Fetch IoT data from Firebase and process it for ML model
        """
        if not self.firebase_available:
            print("Firebase not available, using local CSV")
            return self._load_local_fallback()

        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Fetch data from Firebase
            sensor_data = self.db_ref.child('sensor_data').order_by_key().get()
            
            if not sensor_data:
                print("No data found in Firebase, using local CSV")
                return self._load_local_fallback()

            # Process Firebase data
            data_list = []
            for timestamp_key, value in sensor_data.items():
                if isinstance(value, dict):
                    # Convert timestamp to datetime
                    try:
                        # Assuming timestamp_key is in format like "2024-12-15_14-30-00"
                        if '_' in timestamp_key:
                            date_str = timestamp_key.replace('_', ' ').replace('-', ':')
                            date_obj = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
                        else:
                            # If it's a Unix timestamp
                            date_obj = datetime.fromtimestamp(int(timestamp_key))
                        
                        # Filter by date range
                        if start_date <= date_obj <= end_date:
                            record = {
                                'Date': date_obj,
                                'pH': value.get('pH', 8.0),
                                'TDS': value.get('TDS', 7000),
                                'Temperature': value.get('Temperature', 30.0),
                                'Turbidity': value.get('Turbidity', 25.0),
                                'suitability': value.get('suitability', 1),
                                'Pond No': value.get('pond_no', 0)
                            }
                            data_list.append(record)
                    except (ValueError, TypeError) as e:
                        print(f"Error parsing timestamp {timestamp_key}: {e}")
                        continue

            if not data_list:
                print("No valid data found in date range, using local CSV")
                return self._load_local_fallback()

            # Create DataFrame
            df = pd.DataFrame(data_list)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date').reset_index(drop=True)
            
            # Add month column
            df['month'] = df['Date'].dt.month
            
            # Estimate missing DO and Salinity
            df = self._estimate_do_salinity(df)
            
            # Save to CSV for backup and model training
            self._save_to_csv(df)
            
            return df

        except Exception as e:
            print(f"Error fetching Firebase data: {e}")
            return self._load_local_fallback()

    def _load_local_fallback(self):
        """Load local CSV as fallback"""
        try:
            df = pd.read_csv('sample_prawn_farm_data.csv')
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Add missing columns if they don't exist
            if 'suitability' not in df.columns:
                df['suitability'] = 1  # Default to suitable
            
            return df
        except Exception as e:
            print(f"Error loading local CSV: {e}")
            return pd.DataFrame()

    def _save_to_csv(self, df, filename='updated_prawn_farm_data.csv'):
        """Save processed data to CSV"""
        try:
            # Select columns in the right order
            columns_order = ['Date', 'Pond No', 'pH', 'TDS', 'Temperature', 
                           'Turbidity', 'DO', 'Salinity', 'suitability', 'month']
            
            # Only include columns that exist
            available_columns = [col for col in columns_order if col in df.columns]
            df_to_save = df[available_columns].copy()
            
            df_to_save.to_csv(filename, index=False)
            print(f"Data saved to {filename}")
            return True
        except Exception as e:
            print(f"Error saving to CSV: {e}")
            return False

    def get_latest_data(self, days_back=7):
        """Get latest data for dashboard display"""
        df = self.fetch_and_process_iot_data(days_back)
        return df.tail(days_back * 24) if not df.empty else df
