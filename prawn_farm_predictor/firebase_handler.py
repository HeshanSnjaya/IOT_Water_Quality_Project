import pandas as pd
import streamlit as st
import numpy as np
from datetime import datetime, timedelta
import os

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

    def _estimate_do_salinity(self, ph, tds, temperature, turbidity):
        """Estimate DO and Salinity for a single record"""
        # Use safe values for estimation if sensors are faulty
        safe_ph = max(6.5, min(ph, 9.5)) if ph > 0 else 7.5
        safe_tds = max(1000, tds) if tds > 0 else 7000
        safe_temp = max(20, min(temperature, 40)) if temperature > 0 else 30
        
        # Estimate Dissolved Oxygen (DO)
        do = (
            8.5 - (safe_temp - 25) * 0.15 +  # Temperature effect
            (safe_ph - 7.5) * 0.3 -          # pH effect
            (safe_tds / 1000) * 0.2 +        # TDS effect
            np.random.normal(0, 0.3)         # Natural variation
        )
        do = np.clip(do, 3.0, 8.0)
        
        # Estimate Salinity
        salinity = (
            (safe_tds / 500) +               # Base relationship
            (safe_temp - 30) * 0.1 +        # Temperature effect
            np.random.normal(0, 1.5)         # Natural variation
        )
        salinity = np.clip(salinity, 8, 25)
        
        return do, salinity

    def fetch_and_append_iot_data(self):
        """Fetch latest IoT data and append to CSV"""
        if not self.firebase_available:
            print("Firebase not available, using local CSV")
            return self._load_local_fallback()

        try:
            # Get data from Firebase 'test' node
            sensor_data = self.db_ref.child('test').get()
            
            if not sensor_data:
                print("No data found in Firebase, using local CSV")
                return self._load_local_fallback()

            print(f"Raw Firebase data: {sensor_data}")
            
            # Process single sensor reading - handle keys with colons
            if isinstance(sensor_data, dict):
                # Handle both formats: with and without colons
                ph_value = float(sensor_data.get('Ph:', sensor_data.get('ph', sensor_data.get('pH', 8.0))))
                tds_value = float(sensor_data.get('TDS:', sensor_data.get('TDS', 7000)))
                temp_value = float(sensor_data.get('Temperature:', sensor_data.get('Temperature', 30.0)))
                turbidity_value = float(sensor_data.get('Turbidity:', sensor_data.get('Turbidity', 25.0)))
                suitability_value = int(sensor_data.get('Suitability:', sensor_data.get('suitability', 1)))
                
                print(f"Extracted values - pH: {ph_value}, TDS: {tds_value}, Temp: {temp_value}, Turbidity: {turbidity_value}, Suitability: {suitability_value}")
                
                # DON'T APPLY BOUNDS - Keep actual sensor readings for accurate monitoring
                # Only handle completely invalid readings (negative values, etc.)
                if ph_value < 0:
                    ph_value = 7.5  # Default only if completely invalid
                if tds_value < 0:
                    tds_value = 7000  # Default only if completely invalid
                if temp_value < 0:
                    temp_value = 30.0  # Default only if completely invalid
                if turbidity_value < 0:
                    turbidity_value = 25.0  # Default only if completely invalid
                
                # Estimate missing parameters using safe values
                do_value, salinity_value = self._estimate_do_salinity(
                    ph_value, tds_value, temp_value, turbidity_value
                )
                
                # Create new record with ACTUAL sensor values
                new_record = {
                    'Date': datetime.now(),
                    'Pond No': sensor_data.get('pond_no', 0),
                    'pH': ph_value,  # Keep actual pH reading
                    'TDS': tds_value,  # Keep actual TDS reading
                    'Temperature': temp_value,  # Keep actual temperature reading
                    'Turbidity': turbidity_value,  # Keep actual turbidity reading
                    'DO': do_value,
                    'Salinity': salinity_value,
                    'suitability': suitability_value,  # Keep original IoT suitability
                    'month': datetime.now().month
                }
                
                print(f"Created record with actual sensor values - pH: {ph_value}, TDS: {tds_value}, Suitability: {suitability_value}")
                
                # Append to existing CSV
                success = self._append_to_csv(new_record)
                
                if success:
                    print("Successfully appended new data to CSV")
                    return self._load_updated_csv()
                else:
                    print("Failed to append data, using existing CSV")
                    return self._load_local_fallback()
                
            print("Invalid data structure, using local CSV")
            return self._load_local_fallback()

        except Exception as e:
            print(f"Error fetching Firebase data: {e}")
            import traceback
            traceback.print_exc()
            return self._load_local_fallback()

    def _append_to_csv(self, new_record, filename='updated_prawn_farm_data.csv'):
        """Append single record to CSV file"""
        try:
            # Create DataFrame from single record
            new_df = pd.DataFrame([new_record])
            
            # Check if CSV exists
            if os.path.exists(filename):
                # Append to existing file
                new_df.to_csv(filename, mode='a', header=False, index=False)
                print(f"Appended new record to {filename}")
            else:
                # Create new file with header
                columns_order = ['Date', 'Pond No', 'pH', 'TDS', 'Temperature', 
                               'Turbidity', 'DO', 'Salinity', 'suitability', 'month']
                new_df = new_df[columns_order]
                new_df.to_csv(filename, mode='w', header=True, index=False)
                print(f"Created new file {filename} with first record")
            
            return True
            
        except Exception as e:
            print(f"Error appending to CSV: {e}")
            return False

    def _load_updated_csv(self, filename='updated_prawn_farm_data.csv'):
        """Load the updated CSV file"""
        try:
            if os.path.exists(filename):
                df = pd.read_csv(filename)
                df['Date'] = pd.to_datetime(df['Date'])
                print(f"Loaded updated CSV with {len(df)} records")
                return df
            else:
                return self._load_local_fallback()
        except Exception as e:
            print(f"Error loading updated CSV: {e}")
            return self._load_local_fallback()

    def _load_local_fallback(self):
        """Load local CSV as fallback"""
        try:
            # Try updated CSV first, then sample CSV
            if os.path.exists('updated_prawn_farm_data.csv'):
                df = pd.read_csv('updated_prawn_farm_data.csv')
                df['Date'] = pd.to_datetime(df['Date'])
                print("Loaded existing updated_prawn_farm_data.csv")
            else:
                df = pd.read_csv('sample_prawn_farm_data.csv')
                df['Date'] = pd.to_datetime(df['Date'])
                print("Loaded sample_prawn_farm_data.csv")
            
            if 'suitability' not in df.columns:
                df['suitability'] = 1
            
            return df
        except Exception as e:
            print(f"Error loading local CSV: {e}")
            return pd.DataFrame()

    def fetch_and_process_iot_data(self, days_back=30):
        """Main method for fetching and processing IoT data"""
        return self.fetch_and_append_iot_data()

    def get_latest_data(self, days_back=7):
        """Get latest data for dashboard display"""
        df = self.fetch_and_append_iot_data()
        return df.tail(days_back) if not df.empty else df

    def get_current_csv_status(self):
        """Get status of CSV files"""
        status = {}
        
        if os.path.exists('updated_prawn_farm_data.csv'):
            df = pd.read_csv('updated_prawn_farm_data.csv')
            status['updated_csv'] = {
                'exists': True,
                'records': len(df),
                'last_update': df['Date'].iloc[-1] if not df.empty else 'Unknown'
            }
        else:
            status['updated_csv'] = {'exists': False}
        
        if os.path.exists('sample_prawn_farm_data.csv'):
            df = pd.read_csv('sample_prawn_farm_data.csv')
            status['sample_csv'] = {
                'exists': True,
                'records': len(df)
            }
        else:
            status['sample_csv'] = {'exists': False}
        
        return status
