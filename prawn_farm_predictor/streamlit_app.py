import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from model_fusion import ModelFusion
from firebase_handler import FirebaseHandler
from notifications import check_suitability_and_notify, check_prediction_suitability

st.set_page_config(
    page_title="Prawn Farm Water Quality Predictor",
    page_icon="🦐",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def get_fusion():
    try:
        return ModelFusion()
    except Exception as e:
        st.error(f"Error loading ML models: {e}")
        return None

@st.cache_resource
def get_firebase():
    return FirebaseHandler()

def display_metrics(latest_row):
    st.subheader("📊 Current Water Quality Parameters")
    cols = st.columns(3)
    
    metrics = [
        ("pH", "pH", "🔬"),
        ("TDS", "TDS (ppm)", "💧"),
        ("Temperature", "Temp (°C)", "🌡️"),
        ("DO", "DO (mg/L)", "🫧"),
        ("Salinity", "Salinity (ppt)", "🧂"),
        ("Turbidity", "Turbidity (NTU)", "🌊")
    ]
    
    for i, (col, label, icon) in enumerate(metrics):
        if col in latest_row:
            with cols[i % 3]:
                value = latest_row[col]
                status = get_parameter_status(col, value)
                st.metric(
                    label=f"{icon} {label}", 
                    value=f"{value:.2f}",
                    help=f"Status: {status}"
                )

def get_parameter_status(parameter, value):
    """Get status indicator for water quality parameters"""
    ranges = {
        'pH': (7.5, 8.5, 'Optimal'),
        'TDS': (5000, 8000, 'Good'),
        'Temperature': (28, 32, 'Ideal'),
        'DO': (4.0, 7.0, 'Good'),
        'Salinity': (10, 20, 'Suitable'),
        'Turbidity': (10, 30, 'Acceptable')
    }
    
    if parameter in ranges:
        min_val, max_val, status = ranges[parameter]
        if min_val <= value <= max_val:
            return f"✅ {status}"
        elif value < min_val:
            return "⚠️ Low"
        else:
            return "⚠️ High"
    return "ℹ️ Unknown"

def plot_predictions(historical_df, predictions_df):
    """Create comprehensive prediction plots"""
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=['pH', 'TDS', 'Temperature', 'DO', 'Salinity', 'Turbidity'],
        vertical_spacing=0.08
    )
    
    parameters = ['pH', 'TDS', 'Temperature', 'DO', 'Salinity', 'Turbidity']
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
    
    for i, (param, color) in enumerate(zip(parameters, colors)):
        row = (i // 2) + 1
        col = (i % 2) + 1
        
        if param in historical_df.columns:
            # Historical data
            fig.add_trace(
                go.Scatter(
                    x=historical_df['Date'],
                    y=historical_df[param],
                    mode='lines+markers',
                    name=f'{param} (Historical)',
                    line=dict(color=color),
                    showlegend=(i == 0)
                ),
                row=row, col=col
            )
            
            # Predictions
            fig.add_trace(
                go.Scatter(
                    x=predictions_df['Date'],
                    y=predictions_df[param],
                    mode='lines+markers',
                    name=f'{param} (Predicted)',
                    line=dict(color=color, dash='dash'),
                    showlegend=(i == 0)
                ),
                row=row, col=col
            )
    
    fig.update_layout(
        height=800,
        title_text="Water Quality Parameters: Historical vs Predicted",
        showlegend=True
    )
    
    return fig

def main():
    st.title("🦐 Prawn Farm Water Quality Predictor")
    st.markdown("Real-time IoT data integration with ML-powered predictions")
    
    # Initialize handlers
    firebase_handler = get_firebase()
    fusion_model = get_fusion()
    
    if fusion_model is None:
        st.error("ML models not available. Please check model files.")
        return
    
    # Sidebar controls
    st.sidebar.header("⚙️ Configuration")
    
    # Notification settings
    st.sidebar.subheader("🔔 Notification Settings")
    enable_notifications = st.sidebar.checkbox("Enable Notifications", value=True)
    auto_check_suitability = st.sidebar.checkbox("Auto-check Suitability", value=True)
    
    # Data source selection
    data_source = st.sidebar.radio(
        "Data Source",
        ["Firebase IoT Data", "Local CSV File"],
        help="Choose between live IoT data from Firebase or local CSV file"
    )
    
    # Prediction settings
    st.sidebar.subheader("Prediction Settings")
    prediction_days = st.sidebar.slider("Prediction Days", 1, 10, 5)
    fusion_method = st.sidebar.selectbox(
        "Fusion Method",
        ["average", "weighted_average", "max", "min"]
    )
    
    if fusion_method == "weighted_average":
        rf_weight = st.sidebar.slider("Random Forest Weight", 0.0, 1.0, 0.6)
        xgb_weight = 1.0 - rf_weight
        weights = [rf_weight, xgb_weight]
        st.sidebar.write(f"XGBoost Weight: {xgb_weight:.1f}")
    else:
        weights = None
    
    # Data loading and processing
    st.header("📊 Data Loading")
    
    with st.spinner("Loading and processing data..."):
        if data_source == "Firebase IoT Data":
            days_back = st.sidebar.slider("Days of Historical Data", 7, 60, 30)
            df = firebase_handler.fetch_and_process_iot_data(days_back)
            
            if df.empty:
                st.warning("No data available from Firebase. Using local CSV as fallback.")
                df = pd.read_csv('sample_prawn_farm_data.csv')
                df['Date'] = pd.to_datetime(df['Date'])
                if 'month' not in df.columns:
                    df['month'] = df['Date'].dt.month
        else:
            df = pd.read_csv('sample_prawn_farm_data.csv')
            df['Date'] = pd.to_datetime(df['Date'])
            if 'month' not in df.columns:
                df['month'] = df['Date'].dt.month
    
    if df.empty:
        st.error("No data available for processing.")
        return
    
    # Display data info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Date Range", f"{df['Date'].min().date()} to {df['Date'].max().date()}")
    with col3:
        st.metric("Data Source", data_source)
    
    # Current parameters and suitability check
    if not df.empty:
        latest_row = df.iloc[-1]
        display_metrics(latest_row)
        
        # Suitability notification check
        if enable_notifications and auto_check_suitability:
            check_suitability_and_notify(latest_row)
        
        # Manual suitability check button
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔍 Check Current Suitability", type="secondary"):
                if enable_notifications:
                    check_suitability_and_notify(latest_row)
                else:
                    suitability = int(latest_row.get('suitability', 1))
                    if suitability == 0:
                        st.error("🚨 Current water conditions are NOT suitable for prawn farming!")
                    else:
                        st.success("✅ Current water conditions are suitable for prawn farming!")
        
        # Show data quality indicators
        st.subheader("📈 Data Quality")
        if data_source == "Firebase IoT Data":
            estimated_params = ['DO', 'Salinity']
            st.info(f"📡 **IoT Measured**: pH, TDS, Temperature, Turbidity")
            st.info(f"🧮 **ML Estimated**: {', '.join(estimated_params)}")
        
        # Data preview
        with st.expander("📋 View Recent Data"):
            st.dataframe(df.tail(10))
    
    # Predictions
    st.header("🔮 Future Predictions")
    
    if st.button("Generate Predictions", type="primary"):
        try:
            with st.spinner(f"Generating {prediction_days}-day predictions..."):
                predictions = fusion_model.predict_future(
                    df, 
                    days=prediction_days, 
                    fusion_method=fusion_method,
                    weights=weights
                )
            
            if not predictions.empty:
                st.success(f"✅ Successfully generated {len(predictions)} days of predictions!")
                
                # Check prediction suitability and notify
                if enable_notifications:
                    check_prediction_suitability(predictions)
                
                # Display prediction metrics
                st.subheader("📊 Predicted Values")
                
                # Show predictions in a nice format
                pred_display = predictions[['Date'] + fusion_model.target_features + ['suitability']].copy()
                pred_display['Date'] = pred_display['Date'].dt.strftime('%Y-%m-%d')
                
                st.dataframe(pred_display, use_container_width=True)
                
                # Visualization
                st.subheader("📈 Prediction Visualization")
                
                # Get recent historical data for comparison
                recent_df = df.tail(14)  # Last 2 weeks
                
                fig = plot_predictions(recent_df, predictions)
                st.plotly_chart(fig, use_container_width=True)
                
                # Suitability analysis
                st.subheader("🎯 Suitability Analysis")
                suitable_days = predictions['suitability'].sum() if 'suitability' in predictions.columns else prediction_days
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Suitable Days", f"{suitable_days}/{prediction_days}")
                with col2:
                    suitability_pct = (suitable_days / prediction_days) * 100
                    st.metric("Suitability %", f"{suitability_pct:.1f}%")
                
                # Download predictions
                csv = predictions.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions CSV",
                    data=csv,
                    file_name=f"prawn_farm_predictions_{prediction_days}days.csv",
                    mime="text/csv"
                )
                
            else:
                st.error("Failed to generate predictions. Please check your data and model.")
                
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.exception(e)
    
    # System status
    st.sidebar.header("🔧 System Status")
    st.sidebar.success("✅ ML Models Loaded" if fusion_model else "❌ ML Models Error")
    st.sidebar.success("✅ Firebase Connected" if firebase_handler.firebase_available else "⚠️ Firebase Offline")
    
    if enable_notifications:
        st.sidebar.success("🔔 Notifications Enabled")
    else:
        st.sidebar.info("🔕 Notifications Disabled")

if __name__ == "__main__":
    main()
