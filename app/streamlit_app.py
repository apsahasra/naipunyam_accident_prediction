import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import base64
import os
import time
from PIL import Image


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Road Accident Severity Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ==========================================================
# LOAD MODELS
# ==========================================================

@st.cache_resource
def load_resources():
    model = joblib.load("../models/xgboost.pkl")
    scaler = joblib.load("../models/scaler.pkl")
    features = joblib.load("../models/features.pkl")
    importance_df = joblib.load("../models/feature_importance.pkl")

    return model, scaler, features, importance_df


model, scaler, features, importance_df = load_resources()


# ==========================================================
# SESSION STATE
# ==========================================================

if "page" not in st.session_state:
    st.session_state.page = "home"


# ==========================================================
# IMAGE TO BASE64
# ==========================================================

def get_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# ========================================================== 
# BACKGROUND STYLING 
# ========================================================== 
def set_background(image_path, opacity=0.88):
    encoded = get_base64(image_path)
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(
                rgba(255,255,255,{opacity}),
                rgba(255,255,255,{opacity})
            ), url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        .stSelectbox label, .stSlider label, .stNumberInput label {{
            color: #1F2D3D !important;
            font-weight: 700 !important;
        }}
        </style>
        """, 
        unsafe_allow_html=True
    )


# ==========================================================
# GLOBAL STYLING
# ==========================================================

def apply_global_css():

    st.markdown(
        """
        <style>
        /* Selected value */

        div[data-baseweb="select"] span {
            color: white !important;
        }

        /* Dropdown arrow */

        div[data-baseweb="select"] svg {
            fill: white !important;
        }

        /* Dropdown menu background */

        div[role="listbox"] {
            background-color: #232734 !important;
        }

        /* Dropdown options */

        div[role="option"] {
            background-color: #232734 !important;
            color: white !important;
        }

        /* Text inside options */

        div[role="option"] * {
            color: white !important;
            opacity: 1 !important;
        }

        /* Hover */

        div[role="option"]:hover {
            background-color: #40465A !important;
        }
        
        /* Number input visibility */
        input {
            color: #1F2D3D !important;
            background-color: rgba(255,255,255,0.92) !important;
        }
        
        /* Slider labels */
        .stSlider * {
            color: #1F2D3D !important;
        }
        
        /* Remove Streamlit menu */
        #MainMenu {
            visibility: hidden;
        }

        footer {
            visibility: hidden;
        }

        header {
            visibility: hidden;
        }


        /* General text */
        html, body, [class*="css"] {
            color: #1F2D3D !important;
        }

        p, label {
            color: #1F2D3D !important;
        }

        /* Main page title */
        h1 {
            color: white;
            text-align: center;
            font-weight: 700;

            text-shadow:
                2px 2px 10px rgba(0,0,0,0.65);
        }

        /* Section headings */

        h2, h3 {
            color: #0F172A !important;

            font-weight: 700 !important;

            text-shadow:
                0px 0px 4px rgba(255,255,255,0.85);
        }
        
        /* Streamlit generated headings */

        [data-testid="stMarkdownContainer"] h1,
        [data-testid="stMarkdownContainer"] h2,
        [data-testid="stMarkdownContainer"] h3,
        [data-testid="stMarkdownContainer"] h4 {
            color: #0F172A !important;

            font-weight: 700 !important;

            text-shadow:
                0px 0px 4px rgba(255,255,255,0.85);
        }

        /* Cards */
        .info-card {
            background-color: rgba(234,249,240,0.98);
            padding: 25px;
            border-radius: 18px;
            border-left: 8px solid #56B8C5;
            margin-bottom: 18px;
            color: #355B64;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        }


        /* Severity cards */
        .severity-card {
            background-color: rgba(255,255,255,0.92);
            padding: 18px;
            border-radius: 18px;
            text-align: center;
            margin: 8px;
            box-shadow: 0 3px 12px rgba(0,0,0,0.08);
        }


        /* Prediction result */
        .result-card {
            padding: 25px;
            border-radius: 20px;
            text-align: center;
            color: white;
            font-size: 28px;
            font-weight: bold;
            margin-top: 20px;
            margin-bottom: 20px;
        }


        /* Start button */
        div.stButton > button:first-child {

            background-color: #56B8C5;
            color: white;

            border: none;

            border-radius: 15px;

            height: 55px;

            width: 100%;

            font-size: 20px;

            font-weight: bold;

            transition: 0.3s;
        }


        div.stButton > button:first-child:hover {

            background-color: #449AA6;

            color: white;

            transform: scale(1.03);
        }


        /* Input labels */
        label {
            color: #355B64 !important;
            font-weight: 600;
        }


        /* Back button */
        .back-btn {

            background-color: rgba(86,184,197,0.92);

            width: 45px;

            height: 45px;

            border-radius: 50%;

            display: flex;

            align-items: center;

            justify-content: center;

            color: white;

            font-size: 22px;

            margin-bottom: 20px;
        }


        </style>
        """,
        unsafe_allow_html=True
    )


apply_global_css()


# ==========================================================
# CHART COLOR THEMES
# ==========================================================

severity_colors = {
    "Minor Injury": "#57C58A",
    "Moderate Injury": "#F0C75E",
    "Serious Injury": "#F29A4B",
    "Fatal": "#D95A5A"
}


severity_mapping = {
    0: "Minor Injury",
    1: "Moderate Injury",
    2: "Serious Injury",
    3: "Fatal"
}


severity_emojis = {
    0: "🟢",
    1: "🟡",
    2: "🟠",
    3: "🔴"
}

# ==========================================================
# HOME PAGE
# ==========================================================

if st.session_state.page == "home":

    set_background("images/home.jpg", opacity=0.35)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <h1>
            🚗 Road Accident Severity Prediction System
        </h1>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="
            text-align:center;
            font-size:22px;
            color:white;
            text-shadow:1px 1px 6px rgba(0,0,0,0.6);
            font-weight:500;
            margin-bottom:30px;
        ">
        Predict the severity of road accidents using machine learning
        models trained on real-world accident data.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # =======================
    # WHAT THE APP DOES
    # =======================

    st.markdown(
        """
        <div class="info-card">
        <h3>🔍 What can this system do?</h3>

        • Predict accident severity using environmental conditions.<br>

        • Analyze how different factors influence accident outcomes.<br>

        • Display prediction confidence levels.<br>

        • Highlight the most important contributing features.
        </div>
        """,
        unsafe_allow_html=True
    )

    # =======================
    # HOW TO USE
    # =======================

    st.markdown(
        """
        <div class="info-card">
        <h3>🛠️ How to use this application</h3>

        1. Click <b>Start Prediction</b> below.<br>

        2. Enter accident conditions such as weather and time.<br>

        3. Press the prediction button.<br>

        4. Review severity level, confidence scores,
           and feature analysis.
        </div>
        """,
        unsafe_allow_html=True
    )

    # =======================
    # SEVERITY LEVELS
    # =======================

    st.markdown(
        """
        <h2 style="text-align:center;">
            Severity Levels
        </h2>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            """
            <div class="severity-card">
            <h2>🟢</h2>
            <h4>Minor Injury</h4>
            <p>Low impact accidents with minimal injuries.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="severity-card">
            <h2>🟡</h2>
            <h4>Moderate Injury</h4>
            <p>Accidents requiring medical attention.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <div class="severity-card">
            <h2>🟠</h2>
            <h4>Serious Injury</h4>
            <p>Severe accidents with major consequences.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            """
            <div class="severity-card">
            <h2>🔴</h2>
            <h4>Fatal</h4>
            <p>Critical incidents resulting in fatalities.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br><br>", unsafe_allow_html=True)

    # =======================
    # DATASET / MODEL INFO
    # =======================

    st.markdown(
        """
        <div class="info-card">
        <h3>📊 Project Information</h3>

        <b>Dataset:</b> US Accidents Dataset (2016–2023)<br><br>

        <b>Models Evaluated:</b><br>
        • Logistic Regression<br>
        • Random Forest<br>
        • XGBoost<br><br>

        <b>Best Performing Model:</b> XGBoost
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # =======================
    # START BUTTON
    # =======================

    left, center, right = st.columns([2, 3, 2])

    with center:

        if st.button(
            "🚀 START PREDICTION",
            use_container_width=True
        ):

            st.session_state.page = "prediction"

            st.rerun()
            

# ==========================================================
# PREDICTION PAGE (PART 3A)
# ==========================================================

elif st.session_state.page == "prediction":

    set_background("images/prediction.jpg", opacity=0.40)

    # =======================
    # BACK BUTTON
    # =======================

    col_back, col_title = st.columns([1, 12])

    with col_back:

        if st.button("◀", key="back_button"):

            st.session_state.page = "home"

            st.rerun()

    with col_title:

        st.markdown(
            """
            <h1>
                🚦 Prediction & Analysis
            </h1>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div style="
            text-align:center;
            font-size:20px;
            color:#355B64;
            margin-bottom:30px;
        ">
        Adjust accident conditions below and let the AI model
        estimate the severity level.
        </div>
        """,
        unsafe_allow_html=True
    )

    # =======================
    # INPUT AREA
    # =======================

    left_col, right_col = st.columns([2, 1])

    with left_col:
        st.markdown(
            """
            <div style="
                background-color:rgba(255,255,255,0.88);
                padding:25px;
                border-radius:20px;
                box-shadow:0 6px 18px rgba(0,0,0,0.15);
            ">
            """,
            unsafe_allow_html=True
        )

        st.subheader("📝 Accident Information")

        temperature = st.number_input(
            "Temperature (°F)",
            min_value=-50.0,
            max_value=150.0,
            value=70.0
        )

        humidity = st.slider(
            "Humidity (%)",
            0,
            100,
            50
        )

        visibility = st.number_input(
            "Visibility (miles)",
            min_value=0.0,
            max_value=50.0,
            value=10.0
        )

        wind_speed = st.number_input(
            "Wind Speed (mph)",
            min_value=0.0,
            max_value=100.0,
            value=5.0
        )

        hour = st.slider(
            "Hour of the Day",
            0,
            23,
            12
        )

        day_of_week = st.selectbox(
            "Day of Week",
            [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday"
            ]
        )

        day_map = {
            "Monday": 0,
            "Tuesday": 1,
            "Wednesday": 2,
            "Thursday": 3,
            "Friday": 4,
            "Saturday": 5,
            "Sunday": 6
        }

        month = st.slider(
            "Month",
            1,
            12,
            6
        )

        duration = st.number_input(
            "Estimated Accident Duration (minutes)",
            min_value=0.0,
            max_value=1440.0,
            value=30.0
        )

        is_rush_hour = st.selectbox(
            "Rush Hour?",
            ["No", "Yes"]
        )

        is_night = st.selectbox(
            "Night Time?",
            ["No", "Yes"]
        )
        
        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

    with right_col:

        st.markdown("### 💡 Safety Insights")

        st.markdown(
            """
            <div class="info-card">
            🚗 Rush hour traffic can increase accident severity.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="info-card">
            🌫️ Poor visibility conditions often lead to more serious accidents.
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="info-card">
            🛡️ Safer environmental conditions reduce accident risks.
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    predict_col1, predict_col2, predict_col3 = st.columns([2, 3, 2])

    prediction_made = False

    with predict_col2:

        predict_clicked = st.button(
            "🔮 Predict Severity",
            use_container_width=True
        )

    if predict_clicked:

        with st.spinner(
            "Analyzing accident conditions..."
        ):

            time.sleep(2)

            input_df = pd.DataFrame(
                [[
                    temperature,
                    humidity,
                    visibility,
                    wind_speed,
                    hour,
                    day_map[day_of_week],
                    month,
                    duration,
                    1 if is_rush_hour == "Yes" else 0,
                    1 if is_night == "Yes" else 0
                ]],
                columns=features
            )

            input_scaled = scaler.transform(input_df)

            prediction = model.predict(
                input_scaled
            )[0]

            probabilities = model.predict_proba(
                input_scaled
            )[0]

            prediction_made = True

            severity_text = severity_mapping[prediction]

            severity_color = severity_colors[
                severity_text
            ]

            severity_icon = severity_emojis[
                prediction
            ]

            st.markdown(
                f"""
                <div class="result-card"
                style="
                    background-color:{severity_color};
                    color:white;
                    text-shadow:1px 1px 5px rgba(0,0,0,0.4);
                ">

                {severity_icon}<br><br>

                Predicted Severity:<br>

                {severity_text}

                </div>
                """,
                unsafe_allow_html=True
            )
            
            # ==========================================
            # CONFIDENCE CHART
            # ==========================================

            st.markdown("## 📊 Prediction Confidence")

            confidence_df = pd.DataFrame({
                "Severity": [
                    "Minor Injury",
                    "Moderate Injury",
                    "Serious Injury",
                    "Fatal"
                ],
                "Probability": probabilities
            })

            fig = go.Figure()

            fig.add_trace(
                go.Bar(
                    x=confidence_df["Severity"],
                    y=confidence_df["Probability"],
                    marker_color=[
                        "#57C58A",
                        "#F0C75E",
                        "#F29A4B",
                        "#D95A5A"
                    ],
                    text=[
                        f"{p:.2%}"
                        for p in confidence_df["Probability"]
                    ],
                    textposition="auto"
                )
            )

            fig.update_layout(
                #title="Prediction Confidence by Severity Level",

                plot_bgcolor="rgba(255,255,255,0.75)",

                paper_bgcolor="rgba(255,255,255,0.55)",

                height=500,

                font=dict(
                    color="#000000",
                    size=18
                ),

                xaxis=dict(
                    title="Severity Class",
                    title_font=dict(size=18,color="#0F172A"),
                    tickfont=dict(size=16,color="#0F172A"),
                    showgrid=False
                ),

                yaxis=dict(
                    title="Probability",
                    title_font=dict(size=18,color="#0F172A"),
                    tickfont=dict(size=16,color="#0F172A"),
                    showgrid=True,
                    gridcolor="rgba(0,0,0,0.12)"
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )
            
            st.caption(
                "Higher probability indicates greater confidence in the predicted severity class."
            )

            st.markdown("<br>", unsafe_allow_html=True)

            # ==========================================
            # FEATURE IMPORTANCE
            # ==========================================

            st.markdown(
                "## 🔍 Important Factors Influencing Prediction"
            )

            top_features = (
                importance_df
                .sort_values(
                    "Importance",
                    ascending=False
                )
                .head(10)
            )

            feature_fig = px.bar(
                top_features,
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale=[
                    "#56B8C5",
                    "#7CCED8",
                    "#D5AE52"
                ]
            )

            feature_fig.update_layout(
                plot_bgcolor="rgba(255,255,255,0.75)",

                paper_bgcolor="rgba(255,255,255,0.55)",

                height=650,

                font=dict(
                    color="#0F172A",
                    size=16
                ),

                xaxis=dict(
                    title="Importance Score",
                    title_font=dict(size=18,color="#0F172A"),
                    tickfont=dict(size=15,color="#0F172A"),
                    showgrid=True,
                    gridcolor="rgba(0,0,0,0.12)"
                ),

                yaxis=dict(
                    title="Features",
                    title_font=dict(size=18,color="#0F172A"),
                    tickfont=dict(size=15,color="#0F172A"),
                    categoryorder="total ascending"
                )
            )
            st.plotly_chart(
                feature_fig,
                use_container_width=True
            )
            
            st.caption(
                "Features with larger importance scores have a stronger influence on the XGBoost prediction."
            )

            st.markdown("<br>", unsafe_allow_html=True)

            # ==========================================
            # FEATURE TABLE
            # ==========================================

            st.markdown(
                "### 📋 Feature Importance Table"
            )

            st.markdown(
                """
                <div style="
                    background-color:rgba(255,255,255,0.90);
                    padding:15px;
                    border-radius:20px;
                ">
                """,
                unsafe_allow_html=True
            )

            st.dataframe(
                top_features,
                use_container_width=True
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True
            )

            st.markdown("<br>", unsafe_allow_html=True)

            # ==========================================
            # PROJECT SUMMARY CARD
            # ==========================================

            st.markdown(
                """
                <div class="info-card">

                <h3>🤖 AI Model Information</h3>

                <b>Dataset:</b>
                US Accidents Dataset (2016–2023)
                <br><br>

                <b>Algorithm:</b>
                XGBoost Classifier
                <br><br>

                <b>Prediction Type:</b>
                Multi-Class Classification
                <br><br>

                <b>Classes:</b>
                Minor, Moderate, Serious, Fatal

                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("<br>", unsafe_allow_html=True)

            # ==========================================
            # RECOMMENDATION CARD
            # ==========================================

            recommendation_color = severity_color

            st.markdown(
                f"""
                <div style="
                    background-color:{recommendation_color};
                    color:white;
                    padding:20px;
                    border-radius:18px;
                    text-align:center;
                    font-size:18px;
                    font-weight:600;
                    margin-bottom:20px;
                ">

                Safety Recommendation

                <br><br>

                Monitor weather, visibility,
                traffic density and time-of-day
                conditions to reduce accident risk.

                </div>
                """,
                unsafe_allow_html=True
            )

    # ==========================================
    # FOOTER
    # ==========================================

    st.markdown("<br><br>", unsafe_allow_html=True)

    st.markdown(
        """
        <hr>

        <div style="
            text-align:center;
            color:#355B64;
            font-size:15px;
            padding:15px;
        ">

        🚗 Road Accident Severity Prediction System

        <br>

        Machine Learning • XGBoost • Streamlit

        </div>
        """,
        unsafe_allow_html=True
    )