import streamlit as st
import pandas as pd
import joblib
import os

# -----------------------------------
# PAGE SETTINGS
# -----------------------------------
st.set_page_config(
    page_title="Incubator Quality Control System",
    page_icon="🩺",
    layout="wide"
)

# -----------------------------------
# CUSTOM CSS
# -----------------------------------
st.markdown("""
<style>
body {background-color: #f4f7fb;}
.main-title {
    background: linear-gradient(90deg,#0f4c81,#1f77b4);
    padding: 25px; border-radius: 18px; color: white;
    text-align: center; margin-bottom: 25px;
}
.section-box {
    background-color: white; padding: 25px;
    border-radius: 18px; box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}
.normal-box {background-color: #d4edda; color: #155724; padding: 20px; border-radius: 12px; font-size: 22px; font-weight: bold;}
.warning-box {background-color: #fff3cd; color: #856404; padding: 20px; border-radius: 12px; font-size: 22px; font-weight: bold;}
.failure-box {background-color: #f8d7da; color: #721c24; padding: 20px; border-radius: 12px; font-size: 22px; font-weight: bold;}
.note-box {background-color: #eef4ff; padding: 15px; border-radius: 12px; margin-top: 15px;}
.range-box {background-color: #f5f5f5; padding: 12px; border-radius: 10px; margin-bottom: 10px;}
div.stButton > button {background-color: #0f4c81; color: white; border-radius: 10px; height: 50px; width: 100%; font-size: 18px; font-weight: bold; border: none;}
</style>
""", unsafe_allow_html=True)

# -----------------------------------
# LOAD MODEL
# -----------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "..", "model", "incubator_qc_model.pkl")
model = joblib.load(model_path)

# -----------------------------------
# HEADER
# -----------------------------------
st.markdown("""
<div class="main-title">
<h1>🩺 Incubator Quality Control System</h1>
<p>Machine Learning-Based QC Decision Support</p>
</div>
""", unsafe_allow_html=True)

st.info("Enter incubator measured values and set values to calculate errors and evaluate QC status.")
# -----------------------------------
# SIDEBAR
# -----------------------------------

st.sidebar.title("About This System")

st.sidebar.write("""
This application is designed to evaluate the quality and operational performance 
of neonatal incubators using machine learning and engineering quality-control principles.
""")

st.sidebar.info("""
The system compares:
- Set values from the incubator
- Measured values from testing equipment

Then calculates:
- Temperature error
- Humidity error
- Oxygen error

to determine the incubator quality status.
""")

st.sidebar.write("### Intended Users")

st.sidebar.write("""
- Biomedical Engineers
- Hospitals
- NICU Technicians
- Government Inspection Teams
- Medical Equipment Quality Control Departments
""")

st.sidebar.write("### Quality Classification")

st.sidebar.write("""
✅ Normal Quality  
⚠ Warning Condition  
❌ Quality Failure
""")
# -----------------------------------
# FUNCTIONS
# -----------------------------------
def calculate_error(set_value, measured_value):
    """Calculate percentage error between set and measured values"""
    return abs((measured_value - set_value) / set_value) * 100

def get_result(temp_error, humidity_error, oxygen_error, noise_level, device_age, maintenance_days, repair_history):
    """QC decision based on errors and device info"""
    score = 0
    if temp_error > 1: score += 1
    if humidity_error > 10: score += 1
    if oxygen_error > 5: score += 1
    if noise_level > 60: score += 1
    if device_age > 7: score += 1
    if maintenance_days > 90: score += 1
    if repair_history > 3: score += 1

    if score <= 1:
        return "Normal"
    elif score <= 3:
        return "Warning"
    else:
        return "Failure"

def suggest_parts_to_check(temp_error, humidity_error, oxygen_error, noise_level, device_age, maintenance_days, repair_history):
    """Suggest components to inspect based on errors"""
    parts = []
    if temp_error > 1: parts.append("Temperature sensor or heater system")
    if humidity_error > 10: parts.append("Humidity sensor or humidification system")
    if oxygen_error > 5: parts.append("Oxygen sensor or gas system")
    if noise_level > 60: parts.append("Fans, airflow system, or filters")
    if device_age > 7 or repair_history > 3: parts.append("Aging components or frequent repairs")
    if maintenance_days > 90: parts.append("Maintenance overdue: check calibration and filters")
    if not parts: parts.append("No urgent inspection required")
    return parts

# -----------------------------------
# LAYOUT
# -----------------------------------
col1, col2 = st.columns([1,1])

with col1:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("Enter Set & Measured Values")

    st.markdown("### Temperature")
    set_temp = st.number_input("Set Temperature (°C)", value=36.5, step=0.1)
    measured_temp = st.number_input("Measured Temperature (°C)", value=36.6, step=0.1)
    st.markdown('<div class="range-box">Recommended temp error: < 1%</div>', unsafe_allow_html=True)

    st.markdown("### Humidity")
    set_humidity = st.number_input("Set Humidity (%)", value=60.0, step=1.0)
    measured_humidity = st.number_input("Measured Humidity (%)", value=59.0, step=1.0)
    st.markdown('<div class="range-box">Recommended humidity error: < 10%</div>', unsafe_allow_html=True)

    st.markdown("### Oxygen")
    set_oxygen = st.number_input("Set Oxygen (%)", value=21.0, step=1.0)
    measured_oxygen = st.number_input("Measured Oxygen (%)", value=21.0, step=1.0)
    st.markdown('<div class="range-box">Recommended oxygen error: < 5%</div>', unsafe_allow_html=True)

    st.markdown("### Noise Level")
    noise_level = st.number_input("Noise Level (dB)", value=45, step=1)
    st.markdown('<div class="range-box">Recommended noise: < 60 dB</div>', unsafe_allow_html=True)

    st.markdown("### Device Information")
    device_age = st.number_input("Device Age (years)", value=0, step=1)
    maintenance_days = st.number_input("Days Since Last Maintenance", value=0, step=1)
    repair_history = st.number_input("Previous Repairs", value=0, step=1)

    st.markdown("""
    <div class="note-box">
    <b>Note:</b> For new devices, set Device Age, Maintenance Days, and Repair History to 0.
    </div>
    """, unsafe_allow_html=True)

    check_button = st.button("Run QC Test")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    st.subheader("QC Results")

    if check_button:
        temp_error = calculate_error(set_temp, measured_temp)
        humidity_error = calculate_error(set_humidity, measured_humidity)
        oxygen_error = calculate_error(set_oxygen, measured_oxygen)

        # Prepare ML input with exact feature names
        input_data = pd.DataFrame([{
            "temp_error": temp_error,
            "humidity_error": humidity_error,
            "oxygen_error": oxygen_error,
            "noise_level": noise_level,
            "device_age": device_age,
            "last_maintenance_days": maintenance_days,
            "repair_history": repair_history
        }])

        # Predict using ML model
        prediction = model.predict(input_data)[0]
        result = get_result(temp_error, humidity_error, oxygen_error, noise_level, device_age, maintenance_days, repair_history)
        parts_to_check = suggest_parts_to_check(temp_error, humidity_error, oxygen_error, noise_level, device_age, maintenance_days, repair_history)

        st.write("### Calculated Errors")
        st.write(f"Temperature Error: {temp_error:.2f}%")
        st.write(f"Humidity Error: {humidity_error:.2f}%")
        st.write(f"Oxygen Error: {oxygen_error:.2f}%")

        st.write("---")

        if result == "Normal":
            st.markdown('<div class="normal-box">✅ NORMAL QUALITY</div>', unsafe_allow_html=True)
            st.success("The incubator is within acceptable QC limits.")
        elif result == "Warning":
            st.markdown('<div class="warning-box">⚠ WARNING CONDITION</div>', unsafe_allow_html=True)
            st.warning("Inspection or recalibration recommended.")
        else:
            st.markdown('<div class="failure-box">❌ QUALITY FAILURE</div>', unsafe_allow_html=True)
            st.error("Incubator failed QC testing.")

        st.markdown('<div class="note-box"><b>Parts to inspect:</b></div>', unsafe_allow_html=True)
        for part in parts_to_check:
            st.write(f"- {part}")

    else:
        st.info("Enter values and click 'Run QC Test' to see results.")

    st.markdown('</div>', unsafe_allow_html=True)