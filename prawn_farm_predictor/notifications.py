import streamlit as st

def check_suitability_and_notify(latest_data):
    """Check water quality suitability and show appropriate notifications"""
    if 'suitability' not in latest_data:
        return
    
    suitability = int(latest_data['suitability'])
    
    if suitability == 0:
        # Not suitable - show error notification
        st.error("🚨 **WATER QUALITY ALERT**: Current IoT sensor data indicates water conditions are NOT SUITABLE for prawn farming!")
        
        # Show specific parameter issues with sensor fault detection
        issues = []
        sensor_faults = []
        
        if 'pH' in latest_data:
            ph = latest_data['pH']
            if ph < 6.0:
                sensor_faults.append(f"pH: {ph:.2f} (CRITICALLY LOW - Possible sensor fault)")
            elif ph < 7.5 or ph > 8.5:
                issues.append(f"pH: {ph:.2f} (Optimal: 7.5-8.5)")
        
        if 'TDS' in latest_data:
            tds = latest_data['TDS']
            if tds < 100:
                sensor_faults.append(f"TDS: {tds:.0f} ppm (CRITICALLY LOW - Possible sensor fault)")
            elif tds < 5000 or tds > 8000:
                issues.append(f"TDS: {tds:.0f} ppm (Optimal: 5000-8000)")
        
        if 'Temperature' in latest_data:
            temp = latest_data['Temperature']
            if temp < 15 or temp > 45:
                sensor_faults.append(f"Temperature: {temp:.1f}°C (EXTREME VALUE - Check sensor)")
            elif temp < 28 or temp > 32:
                issues.append(f"Temperature: {temp:.1f}°C (Optimal: 28-32°C)")
        
        if 'Turbidity' in latest_data:
            turb = latest_data['Turbidity']
            if turb > 1000:
                sensor_faults.append(f"Turbidity: {turb:.1f} NTU (EXTREMELY HIGH - Check sensor/water)")
            elif turb > 30:
                issues.append(f"Turbidity: {turb:.1f} NTU (Should be < 30)")
        
        if 'DO' in latest_data:
            do = latest_data['DO']
            if do < 4.0 or do > 7.0:
                issues.append(f"DO: {do:.2f} mg/L (Optimal: 4.0-7.0)")
        
        if 'Salinity' in latest_data:
            sal = latest_data['Salinity']
            if sal < 10 or sal > 20:
                issues.append(f"Salinity: {sal:.1f} ppt (Optimal: 10-20)")
        
        # Show sensor faults first (critical)
        if sensor_faults:
            st.error("**🔧 SENSOR FAULTS DETECTED:**")
            for fault in sensor_faults:
                st.write(f"• {fault}")
            st.warning("**⚠️ Check sensor calibration and connections immediately!**")
        
        # Show parameter issues
        if issues:
            st.warning("**Parameter Issues Detected:**")
            for issue in issues:
                st.write(f"• {issue}")
        
        # Show recommendations
        st.info("**Recommended Actions:**")
        if sensor_faults:
            st.write("• **URGENT**: Inspect and recalibrate faulty sensors")
            st.write("• Verify sensor connections and power supply")
            st.write("• Check for physical damage to sensors")
        st.write("• Check water filtration systems")
        st.write("• Monitor aeration equipment")
        st.write("• Consider water treatment measures")
        st.write("• Consult aquaculture specialist if issues persist")
    
    elif suitability == 1:
        # Suitable - show success notification
        st.success("✅ **WATER QUALITY STATUS**: Current IoT sensor data indicates water conditions are SUITABLE for prawn farming!")

def check_prediction_suitability(predictions_df):
    """Check future predictions suitability and notify"""
    if predictions_df.empty or 'suitability' not in predictions_df.columns:
        return
    
    unsuitable_days = len(predictions_df[predictions_df['suitability'] == 0])
    total_days = len(predictions_df)
    
    if unsuitable_days > 0:
        st.warning(f"⚠️ **PREDICTION ALERT**: Forecast shows {unsuitable_days} out of {total_days} days may have unsuitable water conditions. Plan corrective actions!")
        
        # Show which days are problematic
        unsuitable_dates = predictions_df[predictions_df['suitability'] == 0]['Date'].dt.strftime('%Y-%m-%d').tolist()
        st.write(f"**Problematic dates**: {', '.join(unsuitable_dates)}")
    else:
        st.success(f"✅ **PREDICTION STATUS**: All {total_days} forecasted days show suitable water conditions!")
