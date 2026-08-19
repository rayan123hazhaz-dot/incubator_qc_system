import streamlit as st
import pandas as pd
import joblib
import os


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Neonatal Incubator Acceptance Testing",
    page_icon="🩺",
    layout="wide"
)


# ============================================================
# CSS DESIGN
# ============================================================

st.markdown("""
<style>

.main-title {

background: linear-gradient(90deg,#0f4c81,#1f77b4);
padding:30px;
border-radius:20px;
color:white;
text-align:center;
box-shadow:0 4px 12px rgba(0,0,0,0.15);

}


.section-box {

background:white;
padding:20px;
border-radius:15px;
box-shadow:0 3px 10px rgba(0,0,0,0.08);
margin-bottom:20px;

}


.instrument-box {

background:#eef4ff;
padding:12px;
border-left:5px solid #1f77b4;
border-radius:10px;
margin-top:10px;

}


.criteria-box {

background:#f7f9fc;
padding:12px;
border-radius:10px;
margin-top:10px;

}


.note-box {

background:#fff8e1;
padding:15px;
border-radius:12px;

}


.accepted {

background:#d4edda;
color:#155724;
padding:20px;
border-radius:12px;
font-size:22px;
font-weight:bold;

}


.warning {

background:#fff3cd;
color:#856404;
padding:20px;
border-radius:12px;
font-size:22px;
font-weight:bold;

}


.rejected {

background:#f8d7da;
color:#721c24;
padding:20px;
border-radius:12px;
font-size:22px;
font-weight:bold;

}


.error-box {

background:#f9fafc;
padding:12px;
border-left:5px solid #0f4c81;
border-radius:10px;
margin-bottom:10px;

}


</style>

""", unsafe_allow_html=True)



# ============================================================
# LOAD MODEL
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


MODEL_PATH = os.path.join(
    BASE_DIR,
    "..",
    "model",
    "incubator_qc_model.pkl"
)


try:

    model = joblib.load(MODEL_PATH)

except Exception as e:

    st.error("Model loading failed")
    st.write(e)
    st.stop()



# ============================================================
# FUNCTIONS
# ============================================================


def calculate_error(set_value, measured_value):

    if set_value == 0:
        return 0

    return abs(
        (measured_value-set_value)/set_value
    )*100



# ------------------------------------------------------------
# FINAL ENGINEERING ACCEPTANCE DECISION
# ------------------------------------------------------------

def acceptance_decision(
    temp_error,
    humidity_error,
    oxygen_error,
    airflow,
    noise
):

    problems=[]


    if temp_error >= 1:

        problems.append(
            "Temperature error exceeds acceptance limit"
        )


    if humidity_error >= 10:

        problems.append(
            "Humidity error exceeds acceptance limit"
        )


    if oxygen_error >= 5:

        problems.append(
            "Oxygen concentration error exceeds acceptance limit"
        )


    if airflow < 0.05 or airflow > 0.138:

        problems.append(
            "Airflow is outside acceptable range"
        )


    if noise >= 60:

        problems.append(
            "Noise level exceeds acceptable limit"
        )



    if len(problems)==0:

        return (
            "Accepted",
            "All measured parameters are within acceptance criteria.",
            problems
        )


    elif len(problems)<=2:

        return (
            "Needs Inspection",
            "Some parameters require verification or adjustment.",
            problems
        )


    else:

        return (
            "Rejected / Unsatisfactory",
            "Multiple performance parameters are outside limits.",
            problems
        )



def ml_description(value):

    if value==0:

        return "Model classification: Accepted"

    elif value==1:

        return "Model classification: Needs Inspection"

    else:

        return "Model classification: Rejected"



# ============================================================
# HEADER
# ============================================================


st.markdown("""
<div class="main-title">

<h1>
Acceptance Testing and Performance Evaluation of Neonatal Incubators Using Machine Learning
</h1>

<p>
Biomedical Engineering Decision Support System
</p>

</div>

""",unsafe_allow_html=True)



st.info(
"""
This system evaluates neonatal incubator performance by comparing
incubator settings with measurements obtained from calibrated test
instruments.

The calculated errors are compared with acceptance criteria, while
machine learning provides additional decision support.
"""
)



# ============================================================
# SIDEBAR
# ============================================================


st.sidebar.title("About The System")


st.sidebar.write(
"""
This application is designed for:

• Biomedical Engineers

• Clinical Engineering Departments

• Hospital Inspection Teams

• Medical Device Evaluation Personnel


It can be used for:

✓ Newly purchased incubators (acceptance testing)

✓ Incubators already in clinical service (performance evaluation)
"""
)


st.sidebar.write(
"""
Assessment output:

✅ Accepted

⚠️ Needs Inspection

❌ Rejected
"""
)



# ============================================================
# MAIN COLUMNS
# ============================================================


left,right = st.columns([1,1])



# ============================================================
# LEFT SIDE INPUTS
# ============================================================


with left:


    st.header("Incubator Test Measurements")



    # ---------------- TEMPERATURE ----------------


    st.markdown(
    '<div class="section-box">',
    unsafe_allow_html=True
    )


    st.subheader("🌡 Temperature Test")


    set_temp = st.number_input(
        "Temperature Set Value (°C)",
        value=36.5
    )


    measured_temp = st.number_input(
        "Measured Temperature (°C)",
        value=36.5
    )


    st.markdown(
    """
    <div class="instrument-box">

    <b>Test Instrument:</b><br>

    Calibrated incubator analyzer or temperature analyzer
    with temperature probes.

    </div>
    """,
    unsafe_allow_html=True
    )


    st.markdown(
    """
    <div class="criteria-box">

    Acceptance criterion:
    Temperature error < 1%

    </div>
    """,
    unsafe_allow_html=True
    )


    st.markdown("</div>",unsafe_allow_html=True)





    # ---------------- HUMIDITY ----------------


    st.markdown(
    '<div class="section-box">',
    unsafe_allow_html=True
    )


    st.subheader("💧 Humidity Test")


    set_humidity = st.number_input(
        "Humidity Set Value (%)",
        value=60.0
    )


    measured_humidity = st.number_input(
        "Measured Humidity (%)",
        value=60.0
    )


    st.markdown(
    """
    <div class="instrument-box">

    <b>Test Instrument:</b><br>

    Calibrated humidity meter or incubator analyzer
    with humidity sensor.

    </div>
    """,
    unsafe_allow_html=True
    )


    st.markdown(
    """
    <div class="criteria-box">

    Acceptance criterion:
    Humidity error < 10%

    </div>
    """,
    unsafe_allow_html=True
    )


    st.markdown("</div>",unsafe_allow_html=True)
        # ---------------- OXYGEN ----------------


    st.markdown(
    '<div class="section-box">',
    unsafe_allow_html=True
    )


    st.subheader("🫁 Oxygen Concentration Test")


    set_oxygen = st.number_input(
        "Oxygen Set Value (%)",
        value=21.0
    )


    measured_oxygen = st.number_input(
        "Measured Oxygen Concentration (%)",
        value=21.0
    )


    st.markdown(
    """
    <div class="instrument-box">

    <b>Test Instrument:</b><br>

    Calibrated oxygen analyzer for incubator oxygen
    concentration measurement.

    </div>
    """,
    unsafe_allow_html=True
    )


    st.markdown(
    """
    <div class="criteria-box">

    Acceptance criterion:
    Oxygen error < 5%

    </div>
    """,
    unsafe_allow_html=True
    )


    st.markdown("</div>",unsafe_allow_html=True)



    # ---------------- AIRFLOW ----------------


    st.markdown(
    '<div class="section-box">',
    unsafe_allow_html=True
    )


    st.subheader("💨 Airflow Test")


    airflow = st.number_input(
        "Measured Airflow Velocity (m/s)",
        value=0.100,
        step=0.001,
        format="%.3f"
    )


    st.markdown(
    """
    <div class="instrument-box">

    <b>Test Instrument:</b><br>

    Calibrated anemometer or incubator analyzer
    with airflow measurement capability.

    </div>
    """,
    unsafe_allow_html=True
    )


    st.markdown(
    """
    <div class="criteria-box">

    Acceptance range:
    0.05 – 0.138 m/s

    </div>
    """,
    unsafe_allow_html=True
    )


    st.markdown("</div>",unsafe_allow_html=True)




    # ---------------- NOISE ----------------


    st.markdown(
    '<div class="section-box">',
    unsafe_allow_html=True
    )


    st.subheader("🔊 Noise Test")


    noise = st.number_input(
        "Measured Noise Level (dB)",
        value=45.0
    )


    st.markdown(
    """
    <div class="instrument-box">

    <b>Test Instrument:</b><br>

    Calibrated sound level meter.

    </div>
    """,
    unsafe_allow_html=True
    )


    st.markdown(
    """
    <div class="criteria-box">

    Acceptance criterion:
    Noise level < 60 dB

    </div>
    """,
    unsafe_allow_html=True
    )


    st.markdown("</div>",unsafe_allow_html=True)




    # ---------------- DEVICE INFORMATION ----------------


    st.markdown(
    '<div class="section-box">',
    unsafe_allow_html=True
    )


    st.subheader("📋 Device History Information")


    st.caption(
        "Source: Hospital asset records and biomedical engineering maintenance records."
    )


    device_age = st.number_input(
        "Device Age (years)",
        min_value=0,
        value=0
    )


    maintenance_days = st.number_input(
        "Days Since Last Maintenance",
        min_value=0,
        value=0
    )


    repair_history = st.number_input(
        "Previous Repairs Count",
        min_value=0,
        value=0
    )


    st.markdown(
    """
    <div class="note-box">

    <b>For a new incubator during acceptance testing:</b>

    <ul>
    <li>Device age = 0</li>
    <li>Maintenance days = 0</li>
    <li>Repair history = 0</li>
    </ul>

    For devices already in hospitals, enter the values
    from biomedical engineering records.

    </div>
    """,
    unsafe_allow_html=True
    )


    st.markdown("</div>",unsafe_allow_html=True)




    # ---------------- BUTTON ----------------


    run_test = st.button(
        "Run Acceptance Test"
    )
    # ============================================================
# RESULTS SECTION
# ============================================================


with right:

    st.header("Assessment Results")


    if run_test:


        # ====================================================
        # CALCULATE ERRORS
        # ====================================================


        temp_error = calculate_error(
            set_temp,
            measured_temp
        )


        humidity_error = calculate_error(
            set_humidity,
            measured_humidity
        )


        oxygen_error = calculate_error(
            set_oxygen,
            measured_oxygen
        )



        # ====================================================
        # DISPLAY CALCULATED ERRORS
        # ====================================================


        st.subheader("Calculated Measurement Errors")


        st.markdown(
        f"""
        <div class="error-box">

        🌡 Temperature Error:
        <b>{temp_error:.2f}%</b>

        </div>
        """,
        unsafe_allow_html=True
        )


        st.markdown(
        f"""
        <div class="error-box">

        💧 Humidity Error:
        <b>{humidity_error:.2f}%</b>

        </div>
        """,
        unsafe_allow_html=True
        )


        st.markdown(
        f"""
        <div class="error-box">

        🫁 Oxygen Error:
        <b>{oxygen_error:.2f}%</b>

        </div>
        """,
        unsafe_allow_html=True
        )


        st.markdown(
        f"""
        <div class="error-box">

        💨 Airflow:
        <b>{airflow:.3f} m/s</b>

        </div>
        """,
        unsafe_allow_html=True
        )


        st.markdown(
        f"""
        <div class="error-box">

        🔊 Noise:
        <b>{noise:.1f} dB</b>

        </div>
        """,
        unsafe_allow_html=True
        )



        # ====================================================
        # MACHINE LEARNING SUPPORTING ANALYSIS
        # ====================================================


        st.divider()

        st.subheader(
            "Machine Learning Supporting Analysis"
        )


        model_input = pd.DataFrame([{

            "temp_error": temp_error,

            "humidity_error": humidity_error,

            "oxygen_error": oxygen_error,

            "noise_level": noise,

            "airflow": airflow,

            "device_age": device_age,

            "last_maintenance_days": maintenance_days,

            "repair_history": repair_history

        }])


        try:

            ml_prediction = model.predict(
                model_input
            )[0]


            st.write(
                f"Model output value: **{ml_prediction}**"
            )


            if ml_prediction == 0:

                st.info(
                    "Machine learning interpretation: "
                    "Accepted pattern"
                )


            elif ml_prediction == 1:

                st.info(
                    "Machine learning interpretation: "
                    "Needs inspection pattern"
                )


            else:

                st.info(
                    "Machine learning interpretation: "
                    "Rejected pattern"
                )


        except Exception as e:

            st.warning(
                "Machine learning prediction could not be completed."
            )

            st.write(e)




        # ====================================================
        # FINAL ENGINEERING DECISION
        # ====================================================


        final_result, message, problems = acceptance_decision(

            temp_error,

            humidity_error,

            oxygen_error,

            airflow,

            noise

        )



        st.divider()


        st.subheader(
            "Final Acceptance Decision"
        )



        if final_result == "Accepted":


            st.markdown(
            """
            <div class="accepted">

            ✅ ACCEPTED

            </div>
            """,
            unsafe_allow_html=True
            )


            st.success(
                message
            )



        elif final_result == "Needs Inspection":


            st.markdown(
            """
            <div class="warning">

            ⚠️ NEEDS INSPECTION

            </div>
            """,
            unsafe_allow_html=True
            )


            st.warning(
                message
            )



        else:


            st.markdown(
            """
            <div class="rejected">

            ❌ REJECTED / UNSATISFACTORY

            </div>
            """,
            unsafe_allow_html=True
            )


            st.error(
                message
            )



        # ====================================================
        # INSPECTION RECOMMENDATIONS
        # ====================================================


        st.divider()


        st.subheader(
            "Recommended Inspection Areas"
        )


        if len(problems) == 0:


            st.write(
                "No parameter exceeded acceptance criteria."
            )


        else:


            for item in problems:

                st.write(
                    "• " + item
                )



        # ====================================================
        # REPORT SUMMARY
        # ====================================================


        st.divider()


        st.subheader(
            "Assessment Summary"
        )


        summary = pd.DataFrame({

            "Parameter":[

                "Temperature Error",

                "Humidity Error",

                "Oxygen Error",

                "Airflow",

                "Noise Level",

                "Device Age",

                "Maintenance Days",

                "Repair History"

            ],

            "Value":[

                f"{temp_error:.2f}%",

                f"{humidity_error:.2f}%",

                f"{oxygen_error:.2f}%",

                f"{airflow:.3f} m/s",

                f"{noise:.1f} dB",

                device_age,

                maintenance_days,

                repair_history

            ]

        })


        st.dataframe(
            summary,
            use_container_width=True
        )



    else:


        st.info(
            "Enter test values and press 'Run Acceptance Test' "
            "to evaluate the incubator."
        )



# ============================================================
# FOOTER
# ============================================================


st.divider()


st.caption(
"""
Acceptance Testing and Performance Evaluation of Neonatal Incubators
Using Machine Learning

Biomedical Engineering Graduation Project
"""
)
