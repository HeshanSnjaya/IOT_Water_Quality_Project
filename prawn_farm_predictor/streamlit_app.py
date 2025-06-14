import streamlit as st
import pandas as pd
import plotly.express as px
from model_fusion import ModelFusion
from firebase_handler import FirebaseHandler

st.set_page_config(
    page_title="Prawn Farm Water Quality Predictor",
    page_icon="🦐",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .prediction-card {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 0.8rem;
        margin: 1rem 0;
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_fusion():
    return ModelFusion()

@st.cache_resource
def get_firebase():
    return FirebaseHandler()

def load_local_csv():
    df = pd.read_csv('sample_prawn_farm_data.csv')
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['month'] = df['Date'].dt.month
    return df

def display_metrics(latest_row):
    st.subheader("📊 Current Water Quality Parameters")
    cols = st.columns(3)
    metrics = [
        ("pH", "pH"),
        ("TDS", "TDS (ppm)"),
        ("Temperature", "Temp (°C)"),
        ("DO", "DO (mg/L)"),
        ("Salinity", "Salinity (ppt)"),
        ("Turbidity", "Turbidity (NTU)")
    ]
    for i, (col, label) in enumerate(metrics):
        if col in latest_row:
            with cols[i % 3]:
                st.metric(label=label, value=f"{latest_row[col]:.2f}")

def main():
    st.markdown('<h1 class="main-header">🦐 Prawn Farm Water Quality Predictor</h1>', unsafe_allow_html=True)
    fusion = get_fusion()
    firebase = get_firebase()

    # Sidebar
    with st.sidebar:
        st.header("🔧 Control Panel")
        data_source = st.radio("Data Source", ["Firebase", "Local CSV"])
        days_history = st.slider("Days of history", 1, 30, 7)
        prediction_days = st.slider("Prediction days", 1, 10, 5)
        fusion_method = st.selectbox("Fusion Method", ["average", "weighted_average", "max", "min"])
        
        if fusion_method == "weighted_average":
            rf_weight = st.slider("Random Forest Weight", 0.0, 1.0, 0.5, 0.05)
            weights = [rf_weight, 1 - rf_weight]
        else:
            weights = None

        if st.button("🚀 Load Models", use_container_width=True):
            fusion = get_fusion()
            st.success("Models loaded successfully!")

        st.markdown("---")
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()

    # Data Loading
    if data_source == "Firebase":
        df = firebase.get_latest_data(days_history)
        if df.empty:
            st.warning("No Firebase data found, using local CSV instead.")
            df = load_local_csv()
    else:
        df = load_local_csv()

    # Main Content
    col1, col2 = st.columns([2, 1])
    with col1:
        if not df.empty:
            latest_row = df.iloc[-1]
            display_metrics(latest_row)
            
            st.subheader(f"🔮 {prediction_days}-Day Predictions")
            try:
                future_df = fusion.predict_future(df, days=prediction_days, 
                                                 fusion_method=fusion_method, 
                                                 weights=weights)
                
                if not future_df.empty:
                    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                    
                    # Display predictions
                    st.write("**Future Predictions:**")
                    display_df = future_df.set_index('Date').applymap(lambda x: f"{x:.2f}")
                    st.dataframe(display_df, use_container_width=True)
                    
                    # Download button
                    csv = future_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download predictions as CSV",
                        data=csv,
                        file_name='prawn_farm_predictions.csv',
                        mime='text/csv'
                    )
                    
                    # Visualization
                    st.subheader("📈 Prediction Trends")
                    melted = future_df.melt(id_vars='Date', var_name='Parameter', value_name='Value')
                    fig = px.line(melted, x='Date', y='Value', color='Parameter', 
                                markers=True, title=f"{prediction_days}-Day Water Quality Forecast")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.warning("Could not generate predictions")
                    
            except Exception as e:
                st.error(f"Prediction error: {e}")

    with col2:
        st.subheader("📈 Historical Trends")
        if not df.empty and "Date" in df.columns:
            trend_col = st.selectbox("Select Parameter", 
                                    ["Temperature", "pH", "DO", "TDS", "Salinity", "Turbidity"])
            if trend_col in df.columns:
                fig = px.line(df.tail(30), x="Date", y=trend_col, 
                             title=f"Historical {trend_col} Trend")
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
