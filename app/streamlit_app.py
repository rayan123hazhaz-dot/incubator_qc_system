import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
import html


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Neonatal Incubator Acceptance Testing",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "incubator_qc_model.pkl"


# ============================================================
# ACCEPTANCE CRITERIA
# ============================================================
# These are the criteria used by the application.
# Change ONLY these values if your approved project criteria
# are changed later.

TEMP_ERROR_LIMIT = 1.0          # %
HUMIDITY_ERROR_LIMIT = 10.0     # %
OXYGEN_ERROR_LIMIT = 5.0        # %

AIRFLOW_MIN = 0.10              # m/s
AIRFLOW_MAX = 0.35              # m/s

NOISE_MAX = 60.0                # dB(A)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

.main-title {
    background: linear-gradient(90deg, #0f4c81, #1f77b4);
    padding: 30px;
    border-radius: 18px;
    color: white;
    text-align: center;
    margin-bottom: 25px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.15);
}

.main-title h1 {
    margin-bottom: 8px;
}

.main-title p {
    font-size: 18px;
    margin: 0;
}

.section-card {
    background-color: white;
    padding: 24px;
    border-radius: 16px;
    box-shadow: 0 3px 12px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.info-card {
    background-color: #eef5fb;
    padding: 20px;
    border-radius: 14px;
    border-left: 5px solid #1f77b4;
    margin: 15px 0;
}

.instrument-card {
    background-color: #f8fafc;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #e1e7ef;
    margin-bottom: 18px;
}

.normal-box {
    background-color: #d4edda;
    color: #155724;
    padding: 20px;
    border-radius: 14px;
    font-size: 21px;
    font-weight: bold;
    margin: 12px 0;
}

.warning-box {
    background-color: #fff3cd;
    color: #856404;
    padding: 20px;
    border-radius: 14px;
    font-size: 21px;
    font-weight: bold;
    margin: 12px 0;
}

.failure-box {
    background-color: #f8d7da;
    color: #721c24;
    padding: 20px;
    border-radius: 14px;
    font-size: 21px;
    font-weight: bold;
    margin: 12px 0;
}

.pass-box {
    background-color: #e8f5e9;
    color: #1b5e20;
    padding: 15px;
    border-radius: 12px;
    margin: 10px 0;
}

.fail-box {
    background-color: #ffebee;
    color: #b71c1c;
    padding: 15px;
    border-radius: 12px;
    margin: 10px 0;
}

.recommend-box {
    background-color: #eef4ff;
    color: #17365d;
    padding: 20px;
    border-radius: 14px;
    border-left: 5px solid #1f77b4;
    margin-top: 15px;
}

.report-box {
    background-color: #f8f9fa;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #dee2e6;
    margin-top: 15px;
}

div.stButton > button {
    border-radius: 10px;
    min-height: 48px;
    font-weight: bold;
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        return None

    return joblib.load(MODEL_PATH)


model = load_model()


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Home"

if "test_results" not in st.session_state:
    st.session_state.test_results = None


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_error(set_value, measured_value):

    if set_value == 0:
        return 0.0

    return abs(measured_value - set_value) / abs(set_value) * 100


def temperature_status(error):

    if error <= TEMP_ERROR_LIMIT:
        return "PASS"

    return "FAIL"


def humidity_status(error):

    if error <= HUMIDITY_ERROR_LIMIT:
        return "PASS"

    return "FAIL"


def oxygen_status(error):

    if error <= OXYGEN_ERROR_LIMIT:
        return "PASS"

    return "FAIL"


def airflow_status(value):

    if AIRFLOW_MIN <= value <= AIRFLOW_MAX:
        return "PASS"

    return "FAIL"


def noise_status(value):

    if value <= NOISE_MAX:
        return "PASS"

    return "FAIL"


def generate_recommendations(
    temp_status,
    humidity_status_value,
    oxygen_status_value,
    airflow_status_value,
    noise_status_value
):

    recommendations = []

    if temp_status == "FAIL":
        recommendations.append(
            "Inspect the temperature sensor, heating element, "
            "temperature-control system, and calibration."
        )

    if humidity_status_value == "FAIL":
        recommendations.append(
            "Inspect the humidity sensor, humidification system, "
            "water reservoir, and humidity-control circuit."
        )

    if oxygen_status_value == "FAIL":
        recommendations.append(
            "Inspect the oxygen sensor/analyzer interface, oxygen-control "
            "system, oxygen supply, and calibration."
        )

    if airflow_status_value == "FAIL":
        recommendations.append(
            "Inspect the circulation fan, airflow pathway, filters, "
            "air-distribution system, and airflow sensor."
        )

    if noise_status_value == "FAIL":
        recommendations.append(
            "Inspect the circulation fan, motors, alarms, mechanical "
            "vibration, and loose components."
        )

    if not recommendations:
        recommendations.append(
            "No specific parameter requires urgent inspection. "
            "Continue routine monitoring and scheduled preventive maintenance."
        )

    return recommendations


def get_ml_prediction(
    temp_error,
    humidity_error,
    oxygen_error,
    noise_level,
    airflow,
    device_age,
    last_maintenance_days,
    repair_history
):

    if model is None:
        return None, "Model file not found."

    try:

        model_input = pd.DataFrame([{
            "temp_error": temp_error,
            "humidity_error": humidity_error,
            "oxygen_error": oxygen_error,
            "noise_level": noise_level,
            "airflow": airflow,
            "device_age": device_age,
            "last_maintenance_days": last_maintenance_days,
            "repair_history": repair_history
        }])

        prediction = model.predict(model_input)[0]

        return prediction, None

    except Exception as error:

        return None, str(error)


def ml_label(prediction):

    if prediction is None:
        return "Unavailable"

    if prediction == 0:
        return "Accepted Pattern"

    if prediction == 1:
        return "Needs Inspection Pattern"

    return "Rejected Pattern"


def final_decision(
    temp_status,
    humidity_status_value,
    oxygen_status_value,
    airflow_status_value,
    noise_status_value
):

    statuses = [
        temp_status,
        humidity_status_value,
        oxygen_status_value,
        airflow_status_value,
        noise_status_value
    ]

    failed = statuses.count("FAIL")

    if failed == 0:
        return "ACCEPTED"

    if failed <= 2:
        return "NEEDS INSPECTION"

    return "REJECTED / UNSATISFACTORY"


def status_html(status):

    if status == "PASS":

        return (
            '<span style="color:#1b5e20;font-weight:bold;">'
            '✓ PASS'
            '</span>'
        )

    return (
        '<span style="color:#b71c1c;font-weight:bold;">'
        '✕ FAIL'
        '</span>'
    )


# ============================================================
# REPORT GENERATION
# ============================================================

def generate_report_html(results):

    rows = ""

    for item in results["measurements"]:

        rows += f"""
        <tr>
            <td>{html.escape(item["parameter"])}</td>
            <td>{item["set_value"]}</td>
            <td>{item["measured_value"]}</td>
            <td>{item["error"]}</td>
            <td>{html.escape(item["unit"])}</td>
            <td>{html.escape(item["criterion"])}</td>
            <td>{item["status"]}</td>
        </tr>
        """

    recommendations = ""

    for recommendation in results["recommendations"]:

        recommendations += (
            f"<li>{html.escape(recommendation)}</li>"
        )

    report = f"""
<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<title>Neonatal Incubator Acceptance Test Report</title>

<style>

body {{
    font-family: Arial, sans-serif;
    margin: 40px;
    color: #222;
}}

.header {{
    text-align: center;
    border-bottom: 3px solid #0f4c81;
    padding-bottom: 20px;
}}

h1 {{
    color: #0f4c81;
}}

h2 {{
    color: #0f4c81;
    margin-top: 30px;
}}

.info {{
    background: #eef5fb;
    padding: 15px;
    border-radius: 10px;
}}

table {{
    width: 100%;
    border-collapse: collapse;
    margin-top: 15px;
}}

th, td {{
    border: 1px solid #ccc;
    padding: 9px;
    text-align: center;
}}

th {{
    background: #0f4c81;
    color: white;
}}

.accepted {{
    background: #d4edda;
    color: #155724;
    padding: 18px;
    font-size: 22px;
    font-weight: bold;
    border-radius: 10px;
}}

.warning {{
    background: #fff3cd;
    color: #856404;
    padding: 18px;
    font-size: 22px;
    font-weight: bold;
    border-radius: 10px;
}}

.rejected {{
    background: #f8d7da;
    color: #721c24;
    padding: 18px;
    font-size: 22px;
    font-weight: bold;
    border-radius: 10px;
}}

.recommendations {{
    background: #eef4ff;
    padding: 20px;
    border-radius: 10px;
}}

.footer {{
    margin-top: 40px;
    border-top: 1px solid #ccc;
    padding-top: 15px;
}}

</style>

</head>

<body>

<div class="header">

<h1>Neonatal Incubator</h1>

<h2>Acceptance Testing and Performance Evaluation</h2>

<p>Machine Learning-Based Decision Support System</p>

</div>

<h2>Test Information</h2>

<div class="info">

<p><b>Date:</b> {results["date"]}</p>

<p><b>Device Age:</b> {results["device_age"]} years</p>

<p><b>Days Since Last Maintenance:</b>
{results["maintenance_days"]}</p>

<p><b>Previous Repairs:</b>
{results["repair_history"]}</p>

</div>

<h2>Measurement Results</h2>

<table>

<tr>

<th>Parameter</th>
<th>Set Value</th>
<th>Measured Value</th>
<th>Error</th>
<th>Unit</th>
<th>Acceptance Criterion</th>
<th>Status</th>

</tr>

{rows}

</table>


<h2>Machine Learning Assessment</h2>

<div class="info">

<p><b>ML Result:</b>
{html.escape(results["ml_label"])}</p>

</div>


<h2>Final Acceptance Decision</h2>

<div class="{results["decision_class"]}">

{html.escape(results["decision"])}

</div>


<h2>Recommended Inspection Areas</h2>

<div class="recommendations">

<ul>

{recommendations}

</ul>

</div>


<h2>Test Instruments</h2>

<ul>

<li>Digital Thermometer and Thermocouples — Temperature</li>

<li>Calibrated Hygrometer — Humidity</li>

<li>Oxygen Analyzer — Oxygen Concentration</li>

<li>Anemometer / Airflow Meter — Airflow</li>

<li>Sound Level Meter — Acoustic Noise</li>

</ul>


<div class="footer">

<p><b>Biomedical Engineer:</b>
____________________________</p>

<p><b>Signature:</b>
____________________________</p>

<p><b>Date:</b>
____________________________</p>

<br>

<p>
This report is a decision-support output and does not replace
calibrated reference instruments, applicable standards,
manufacturer requirements, or professional biomedical engineering
judgment.
</p>

</div>

</body>

</html>
"""

    return report


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("## 🩺 Incubator System")

st.sidebar.markdown(
    """
This application supports biomedical engineers in the
acceptance testing and performance evaluation of neonatal
incubators.
"""
)

st.sidebar.markdown("---")

if st.sidebar.button("🏠 About the System", use_container_width=True):
    st.session_state.page = "Home"

if st.sidebar.button("🧪 Acceptance Test", use_container_width=True):
    st.session_state.page = "Test"

if st.sidebar.button("📚 Learn", use_container_width=True):
    st.session_state.page = "Learn"


# ============================================================
# HOME PAGE
# ============================================================

if st.session_state.page == "Home":

    st.markdown(
        """
        <div class="main-title">

        <h1>
        🩺 Neonatal Incubator
        </h1>

        <h2>
        Acceptance Testing and Performance Evaluation
        </h2>

        <p>
        Machine Learning-Based Decision Support System
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-card">

        <h2>About the System</h2>

        <p>
        This application is designed to support biomedical engineers
        during acceptance testing and performance evaluation of
        neonatal incubators.
        </p>

        <p>
        The system allows the engineer to enter values obtained from
        the incubator and compare them with measurements obtained from
        reference instruments.
        </p>

        <p>
        Measurement errors are calculated automatically and the
        measured parameters are evaluated against predefined
        acceptance criteria.
        </p>

        <p>
        Machine learning is incorporated as a decision-support
        component to provide an additional equipment-condition
        assessment.
        </p>

        <p>
        The system does not replace calibrated reference instruments,
        applicable standards, manufacturer requirements, or
        professional biomedical engineering judgment.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            <div class="section-card">

            <h2>🧪 Acceptance Testing</h2>

            <p>
            Perform an incubator performance test using temperature,
            humidity, oxygen, airflow, and noise measurements.
            </p>

            <p>
            The system calculates measurement errors, evaluates
            acceptance criteria, provides machine-learning support,
            and identifies recommended inspection areas.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button(
            "🧪 Start Acceptance Test",
            use_container_width=True
        ):

            st.session_state.page = "Test"
            st.rerun()

    with col2:

        st.markdown(
            """
            <div class="section-card">

            <h2>📚 Learn</h2>

            <p>
            Learn about neonatal incubators, their components,
            performance parameters, and the reference instruments
            used during testing.
            </p>

            <p>
            Each instrument is linked to the parameter it is used
            to evaluate.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

        if st.button(
            "📚 Open Learning Section",
            use_container_width=True
        ):

            st.session_state.page = "Learn"
            st.rerun()


# ============================================================
# TEST PAGE
# ============================================================

elif st.session_state.page == "Test":

    st.markdown(
        """
        <div class="main-title">

        <h1>🧪 Acceptance Testing</h1>

        <p>
        Neonatal Incubator Performance Evaluation
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.info(
        "Enter the set values displayed by the incubator and the "
        "corresponding measurements obtained from calibrated reference "
        "instruments."
    )

    # --------------------------------------------------------
    # DEVICE INFORMATION
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-card">',
        unsafe_allow_html=True
    )

    st.subheader("Device Information")

    info1, info2, info3 = st.columns(3)

    with info1:

        device_age = st.number_input(
            "Device Age (years)",
            min_value=0,
            value=0,
            step=1
        )

    with info2:

        maintenance_days = st.number_input(
            "Days Since Last Maintenance",
            min_value=0,
            value=0,
            step=1
        )

    with info3:

        repair_history = st.number_input(
            "Previous Repairs",
            min_value=0,
            value=0,
            step=1
        )

    st.caption(
        "For a newly purchased device, enter 0 for device age, "
        "days since maintenance, and previous repairs."
    )

    st.markdown("</div>", unsafe_allow_html=True)


    # --------------------------------------------------------
    # MEASUREMENTS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-card">',
        unsafe_allow_html=True
    )

    st.subheader("1. Temperature Test")

    col1, col2 = st.columns(2)

    with col1:

        temperature_set = st.number_input(
            "Set Temperature (°C)",
            min_value=0.0,
            value=36.5,
            step=0.1
        )

    with col2:

        temperature_measured = st.number_input(
            "Measured Temperature (°C)",
            min_value=0.0,
            value=36.5,
            step=0.1
        )

    st.info(
        "Instrument: Digital Thermometer + Thermocouples"
    )

    st.caption(
        "Temperature error acceptance limit: "
        f"≤ {TEMP_ERROR_LIMIT:.2f}%"
    )

    st.markdown("</div>", unsafe_allow_html=True)


    st.markdown(
        '<div class="section-card">',
        unsafe_allow_html=True
    )

    st.subheader("2. Humidity Test")

    col1, col2 = st.columns(2)

    with col1:

        humidity_set = st.number_input(
            "Set Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=60.0,
            step=1.0
        )

    with col2:

        humidity_measured = st.number_input(
            "Measured Humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=60.0,
            step=1.0
        )

    st.info(
        "Instrument: Calibrated Hygrometer"
    )

    st.caption(
        "Humidity error acceptance limit: "
        f"≤ {HUMIDITY_ERROR_LIMIT:.2f}%"
    )

    st.markdown("</div>", unsafe_allow_html=True)


    st.markdown(
        '<div class="section-card">',
        unsafe_allow_html=True
    )

    st.subheader("3. Oxygen Concentration Test")

    col1, col2 = st.columns(2)

    with col1:

        oxygen_set = st.number_input(
            "Set Oxygen (%)",
            min_value=0.0,
            max_value=100.0,
            value=21.0,
            step=1.0
        )

    with col2:

        oxygen_measured = st.number_input(
            "Measured Oxygen (%)",
            min_value=0.0,
            max_value=100.0,
            value=21.0,
            step=1.0
        )

    st.info(
        "Instrument: Oxygen Analyzer"
    )

    st.caption(
        "Oxygen error acceptance limit: "
        f"≤ {OXYGEN_ERROR_LIMIT:.2f}%"
    )

    st.markdown("</div>", unsafe_allow_html=True)


    st.markdown(
        '<div class="section-card">',
        unsafe_allow_html=True
    )

    st.subheader("4. Airflow Test")

    airflow = st.number_input(
        "Measured Airflow (m/s)",
        min_value=0.0,
        value=0.20,
        step=0.01
    )

    st.info(
        "Instrument: Calibrated Airflow Meter / Anemometer"
    )

    st.caption(
        f"Acceptance range: {AIRFLOW_MIN:.2f} – "
        f"{AIRFLOW_MAX:.2f} m/s"
    )

    st.markdown("</div>", unsafe_allow_html=True)


    st.markdown(
        '<div class="section-card">',
        unsafe_allow_html=True
    )

    st.subheader("5. Noise Level Test")

    noise_level = st.number_input(
        "Measured Noise Level (dB(A))",
        min_value=0.0,
        value=45.0,
        step=1.0
    )

    st.info(
        "Instrument: Sound Level Meter"
    )

    st.caption(
        f"Maximum accepted noise level: ≤ {NOISE_MAX:.0f} dB(A)"
    )

    st.markdown("</div>", unsafe_allow_html=True)


    # --------------------------------------------------------
    # RUN TEST
    # --------------------------------------------------------

    if st.button(
        "🔍 RUN ACCEPTANCE TEST",
        use_container_width=True
    ):

        # Calculate errors

        temp_error = calculate_error(
            temperature_set,
            temperature_measured
        )

        humidity_error = calculate_error(
            humidity_set,
            humidity_measured
        )

        oxygen_error = calculate_error(
            oxygen_set,
            oxygen_measured
        )


        # Status

        temp_status = temperature_status(temp_error)

        humidity_status_value = humidity_status(
            humidity_error
        )

        oxygen_status_value = oxygen_status(
            oxygen_error
        )

        airflow_status_value = airflow_status(
            airflow
        )

        noise_status_value = noise_status(
            noise_level
        )


        # Recommendations

        recommendations = generate_recommendations(
            temp_status,
            humidity_status_value,
            oxygen_status_value,
            airflow_status_value,
            noise_status_value
        )


        # Final decision

        decision = final_decision(
            temp_status,
            humidity_status_value,
            oxygen_status_value,
            airflow_status_value,
            noise_status_value
        )


        # ML

        ml_prediction, ml_error = get_ml_prediction(
            temp_error,
            humidity_error,
            oxygen_error,
            noise_level,
            airflow,
            device_age,
            maintenance_days,
            repair_history
        )

        ml_result = ml_label(ml_prediction)


        # Save results

        measurements = [

            {
                "parameter": "Temperature",
                "set_value": f"{temperature_set:.2f}",
                "measured_value": f"{temperature_measured:.2f}",
                "error": f"{temp_error:.2f}%",
                "unit": "°C",
                "criterion": f"Error ≤ {TEMP_ERROR_LIMIT:.2f}%",
                "status": temp_status
            },

            {
                "parameter": "Humidity",
                "set_value": f"{humidity_set:.2f}",
                "measured_value": f"{humidity_measured:.2f}",
                "error": f"{humidity_error:.2f}%",
                "unit": "%",
                "criterion": f"Error ≤ {HUMIDITY_ERROR_LIMIT:.2f}%",
                "status": humidity_status_value
            },

            {
                "parameter": "Oxygen",
                "set_value": f"{oxygen_set:.2f}",
                "measured_value": f"{oxygen_measured:.2f}",
                "error": f"{oxygen_error:.2f}%",
                "unit": "%",
                "criterion": f"Error ≤ {OXYGEN_ERROR_LIMIT:.2f}%",
                "status": oxygen_status_value
            },

            {
                "parameter": "Airflow",
                "set_value": "N/A",
                "measured_value": f"{airflow:.2f}",
                "error": "N/A",
                "unit": "m/s",
                "criterion": (
                    f"{AIRFLOW_MIN:.2f}–"
                    f"{AIRFLOW_MAX:.2f} m/s"
                ),
                "status": airflow_status_value
            },

            {
                "parameter": "Noise Level",
                "set_value": "N/A",
                "measured_value": f"{noise_level:.1f}",
                "error": "N/A",
                "unit": "dB(A)",
                "criterion": f"≤ {NOISE_MAX:.0f} dB(A)",
                "status": noise_status_value
            }
        ]


        if decision == "ACCEPTED":
            decision_class = "accepted"

        elif decision == "NEEDS INSPECTION":
            decision_class = "warning"

        else:
            decision_class = "rejected"


        st.session_state.test_results = {

            "date": datetime.now().strftime(
                "%Y-%m-%d %H:%M"
            ),

            "device_age": device_age,

            "maintenance_days": maintenance_days,

            "repair_history": repair_history,

            "measurements": measurements,

            "recommendations": recommendations,

            "decision": decision,

            "decision_class": decision_class,

            "ml_label": ml_result,

            "ml_error": ml_error
        }


    # --------------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------------

    if st.session_state.test_results is not None:

        results = st.session_state.test_results

        st.markdown("---")

        st.markdown(
            """
            <div class="main-title">

            <h1>📊 Test Results</h1>

            <p>
            Acceptance Testing and Performance Evaluation
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )


        # Results table

        result_df = pd.DataFrame(
            results["measurements"]
        )

        st.dataframe(
            result_df,
            use_container_width=True,
            hide_index=True
        )


        # Final decision

        st.subheader("Final Acceptance Decision")

        if results["decision"] == "ACCEPTED":

            st.markdown(
                '<div class="normal-box">'
                '✓ ACCEPTED<br>'
                '<span style="font-size:15px;">'
                'All tested parameters are within the '
                'defined acceptance criteria.'
                '</span>'
                '</div>',
                unsafe_allow_html=True
            )

        elif results["decision"] == "NEEDS INSPECTION":

            st.markdown(
                '<div class="warning-box">'
                '⚠ NEEDS INSPECTION<br>'
                '<span style="font-size:15px;">'
                'One or more parameters require engineering '
                'inspection before final approval.'
                '</span>'
                '</div>',
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                '<div class="failure-box">'
                '✕ REJECTED / UNSATISFACTORY<br>'
                '<span style="font-size:15px;">'
                'Multiple performance parameters exceed '
                'the defined acceptance criteria.'
                '</span>'
                '</div>',
                unsafe_allow_html=True
            )


        # ----------------------------------------------------
        # RECOMMENDED INSPECTION AREAS
        # ----------------------------------------------------

        st.subheader("🔧 Recommended Inspection Areas")

        st.markdown(
            '<div class="recommend-box">',
            unsafe_allow_html=True
        )

        for recommendation in results["recommendations"]:

            st.markdown(
                f"• {recommendation}"
            )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # ML RESULT
        # ----------------------------------------------------

        st.subheader("🤖 Machine Learning Assessment")

        if results["ml_error"] is None:

            st.info(
                f"Machine Learning Result: "
                f"**{results['ml_label']}**"
            )

        else:

            st.warning(
                "Machine learning supporting analysis could "
                "not be completed."
            )

            st.caption(
                results["ml_error"]
            )


        # ----------------------------------------------------
        # TEST INSTRUMENTS SUMMARY
        # ----------------------------------------------------

        st.subheader("🧪 Test Instruments Used")

        instruments_df = pd.DataFrame({

            "Test Parameter": [
                "Temperature",
                "Humidity",
                "Oxygen Concentration",
                "Airflow",
                "Noise Level"
            ],

            "Reference Instrument": [
                "Digital Thermometer + Thermocouples",
                "Calibrated Hygrometer",
                "Oxygen Analyzer",
                "Calibrated Airflow Meter / Anemometer",
                "Sound Level Meter"
            ]

        })

        st.dataframe(
            instruments_df,
            use_container_width=True,
            hide_index=True
        )


        # ----------------------------------------------------
        # REPORT
        # ----------------------------------------------------

        st.subheader("🖨️ Test Report")

        report_html = generate_report_html(
            results
        )

        st.download_button(
            label="🖨️ Download / Print Test Report",
            data=report_html,
            file_name=(
                "neonatal_incubator_acceptance_test.html"
            ),
            mime="text/html",
            use_container_width=True
        )

        st.caption(
            "Open the downloaded report in your browser and "
            "use Ctrl + P to print or save it as PDF."
        )


# ============================================================
# LEARN PAGE
# ============================================================

elif st.session_state.page == "Learn":

    st.markdown(
        """
        <div class="main-title">

        <h1>📚 Learn</h1>

        <p>
        Neonatal Incubators and Performance-Test Instruments
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # INCUBATOR BASICS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-card">',
        unsafe_allow_html=True
    )

    st.subheader("👶 What is a Neonatal Incubator?")

    st.write(
        """
        A neonatal incubator is a medical device designed to provide
        a controlled environment for premature and critically ill
        newborn infants.

        The incubator helps maintain appropriate environmental
        conditions by controlling parameters such as temperature,
        humidity, airflow, and, when available, oxygen concentration.

        Because premature infants have limited thermoregulatory
        ability, accurate environmental control is essential for
        patient safety.
        """
    )

    st.markdown("</div>", unsafe_allow_html=True)


    # --------------------------------------------------------
    # COMPONENTS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-card">',
        unsafe_allow_html=True
    )

    st.subheader("⚙️ Main Incubator Components")

    components = {

        "Incubator Chamber":
        "Transparent enclosure that protects the infant and helps maintain the controlled environment.",

        "Heating System":
        "Provides the heat required to maintain the selected temperature.",

        "Temperature Sensors":
        "Measure internal temperature and provide feedback to the controller.",

        "Air Circulation Fan":
        "Distributes conditioned air throughout the chamber.",

        "Humidity Control System":
        "Maintains the required relative humidity inside the chamber.",

        "Oxygen Supply System":
        "Provides controlled oxygen enrichment in models equipped with oxygen control.",

        "Air Filtration System":
        "Filters incoming air before it enters the chamber.",

        "Alarm System":
        "Provides visual and audible warnings when abnormal conditions occur.",

        "Control Panel":
        "Allows the user to configure and monitor operating parameters."
    }


    for name, description in components.items():

        with st.expander(name):

            st.write(description)

    st.markdown("</div>", unsafe_allow_html=True)


    # --------------------------------------------------------
    # INSTRUMENTS
    # --------------------------------------------------------

    st.subheader("🧪 Instruments Used During Testing")

    st.info(
        "Instrument photographs can be added to the corresponding "
        "cards later by placing the image files in the project's "
        "assets/instruments folder."
    )


    instruments = [

        (
            "🌡️",
            "Digital Thermometer + Thermocouples",
            "Temperature",
            "Measures temperature at different locations inside the incubator chamber.",
            "Used to compare the incubator temperature indication with an independent reference measurement."
        ),

        (
            "💧",
            "Calibrated Hygrometer",
            "Humidity",
            "Measures relative humidity inside the infant compartment.",
            "The measured humidity is compared with the incubator's displayed or selected humidity value."
        ),

        (
            "🫁",
            "Oxygen Analyzer",
            "Oxygen Concentration",
            "Measures oxygen concentration as a percentage.",
            "Used when the incubator includes an oxygen-enrichment system."
        ),

        (
            "🌬️",
            "Airflow Meter / Anemometer",
            "Airflow",
            "Measures air velocity inside the incubator chamber.",
            "Used to evaluate air circulation and airflow performance."
        ),

        (
            "🔊",
            "Sound Level Meter",
            "Noise Level",
            "Measures acoustic noise, normally expressed in dB(A).",
            "Used to evaluate noise produced by fans, motors, alarms, vibration, and other sources."
        ),

        (
            "⚡",
            "Electrical Safety Analyzer",
            "Electrical Safety",
            "Used to evaluate electrical safety parameters such as leakage current and protective earth resistance.",
            "Important during acceptance testing, recurrent testing, and after major repairs."
        ),

        (
            "📈",
            "Data Logger",
            "Continuous Monitoring",
            "Records measurements continuously over a selected period.",
            "Useful for evaluating temperature stability, humidity stability, fluctuations, and recovery behavior."
        ),

        (
            "🧪",
            "Neonatal Incubator Analyzer",
            "Multiple Parameters",
            "Integrated test equipment capable of measuring several incubator performance parameters.",
            "Can reduce the number of separate instruments required during comprehensive testing."
        )
    ]


    for icon, name, parameter, description, usage in instruments:

        st.markdown(
            f"""
            <div class="instrument-card">

            <h3>{icon} {name}</h3>

            <p>
            <b>Used for:</b> {parameter}
            </p>

            <p>
            <b>What does it measure?</b>
            {description}
            </p>

            <p>
            <b>How is it used?</b>
            {usage}
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # TEST PARAMETERS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-card">',
        unsafe_allow_html=True
    )

    st.subheader("📏 Performance Parameters")

    parameter_info = {

        "Temperature":
        "The thermal environment provided by the incubator.",

        "Humidity":
        "The moisture level maintained inside the incubator chamber.",

        "Oxygen Concentration":
        "The oxygen percentage inside incubators equipped with oxygen enrichment.",

        "Airflow":
        "The velocity of air circulating inside the chamber.",

        "Noise":
        "The acoustic environment produced by fans, motors, alarms, vibration, and other sources.",

        "Alarm Function":
        "Ensures abnormal operating conditions produce appropriate warnings.",

        "Electrical Safety":
        "Verifies that the incubator does not present unacceptable electrical hazards."
    }


    for parameter, explanation in parameter_info.items():

        with st.expander(parameter):

            st.write(explanation)

    st.markdown("</div>", unsafe_allow_html=True)


    # --------------------------------------------------------
    # DISCLAIMER
    # --------------------------------------------------------

    st.warning(
        "The learning information and application output are intended "
        "to support biomedical engineering assessment. They do not "
        "replace manufacturer instructions, applicable standards, "
        "calibrated reference instruments, or professional engineering "
        "judgment."
    )
