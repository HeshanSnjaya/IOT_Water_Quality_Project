import pandas as pd
import streamlit as st

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
            self.firebase_available = False

    def get_latest_data(self, days_back=7):
        if not self.firebase_available:
            return pd.DataFrame()
        try:
            sensor_data = self.db_ref.child('sensor_data').order_by_key().limit_to_last(days_back*24).get()
            if not sensor_data:
                return pd.DataFrame()
            data_list = []
            for key, value in sensor_data.items():
                if isinstance(value, dict):
                    value['timestamp'] = key
                    data_list.append(value)
            df = pd.DataFrame(data_list)
            if 'timestamp' in df.columns:
                df['Date'] = pd.to_datetime(df['timestamp'])
                df['month'] = df['Date'].dt.month
            return df
        except Exception as e:
            return pd.DataFrame()
