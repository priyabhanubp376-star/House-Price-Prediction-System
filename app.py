
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "final_house_price_model.pkl"
FIGURE_DIR = ROOT / "reports" / "figures"
DATA_PATH = ROOT / "data" / "raw" / "california_housing.csv"

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>
    .main {
        background-color: #f7f9fc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .main-title {
        font-size: 34px;
        font-weight: 700;
        color: #172554;
    }

    .subtitle {
        color: #64748b;
        font-size: 16px;
    }

    .section-title {
        font-size: 23px;
        font-weight: 650;
        color: #1e293b;
    }

    div[data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #e2e8f0;
        padding: 18px;
        border-radius: 12px;
    }

    div.stButton > button {
        width: 100%;
        border-radius: 8px;
        height: 48px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)

model = load_model()
df = load_dataset()

# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.title("🏠 HousePredict")

    st.caption("Machine Learning Application")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Dashboard",
            "Price Prediction",
            "Model Insights",
            "Project Information"
        ]
    )

    st.divider()

    st.markdown("### Project Details")

    st.write("**Domain:** Data Science")

    st.write("**Algorithm:** Random Forest")

    st.write("**Dataset:** California Housing")

    st.write("**Records:** 20,640")

    st.divider()

    st.caption("Built using Python and Streamlit")

# ==========================================
# DASHBOARD PAGE
# ==========================================

if page == "Dashboard":

    st.markdown(
        '<div class="main-title">House Price Prediction System</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">An end-to-end Machine Learning application '
        'for estimating California district-level house values.</div>',
        unsafe_allow_html=True
    )

    st.write("")

    st.info(
        "This model predicts district-level median house values "
        "using the California Housing dataset. It does not estimate "
        "individual property prices."
    )

    st.markdown('<div class="section-title">Dataset Overview</div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Records", "20,640")
    c2.metric("Input Features", "8")
    c3.metric("Missing Values", "0")
    c4.metric("ML Algorithm", "Random Forest")

    st.divider()

    st.markdown('<div class="section-title">Model Performance</div>',
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    c1.metric("R² Score", "0.8073")
    c2.metric("Mean Absolute Error", "$33,310")
    c3.metric("Root Mean Squared Error", "$49,910")

    st.caption(
        "Performance metrics are from the final evaluation. "
        "Errors are approximate USD equivalents."
    )

    st.divider()

    st.markdown('<div class="section-title">Exploratory Data Analysis</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**House Price Distribution**")

        st.image(
            str(FIGURE_DIR / "house_price_distribution.png"),
            width="stretch"
        )

    with col2:
        st.markdown("**Feature Correlation Heatmap**")

        st.image(
            str(FIGURE_DIR / "correlation_heatmap.png"),
            width="stretch"
        )

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Geographical Analysis**")

        st.image(
            str(FIGURE_DIR / "geographical_analysis.png"),
            width="stretch"
        )

    with col4:
        st.markdown("**Outlier Analysis**")

        st.image(
            str(FIGURE_DIR / "outlier_analysis.png"),
            width="stretch"
        )

# ==========================================
# PRICE PREDICTION PAGE
# ==========================================

elif page == "Price Prediction":

    st.title("🏡 House Price Prediction")

    st.write(
        "Enter housing and geographical details to estimate "
        "the median house value."
    )

    st.warning(
        "Use California district-level feature values. "
        "This model is not designed for Indian property locations."
    )

    with st.form("prediction_form"):

        st.subheader("Housing Information")

        col1, col2 = st.columns(2)

        with col1:

            med_inc = st.number_input(
                "Median Income (in $10,000s)",
                min_value=0.0,
                max_value=20.0,
                value=3.87
            )

            house_age = st.number_input(
                "Median House Age",
                min_value=1.0,
                max_value=60.0,
                value=28.0
            )

            ave_rooms = st.number_input(
                "Average Rooms",
                min_value=1.0,
                max_value=142.0,
                value=5.4
            )

            ave_bedrms = st.number_input(
                "Average Bedrooms",
                min_value=0.5,
                max_value=35.0,
                value=1.1
            )

        with col2:

            population = st.number_input(
                "Population",
                min_value=1.0,
                max_value=36000.0,
                value=1425.0
            )

            ave_occup = st.number_input(
                "Average Occupancy",
                min_value=0.5,
                max_value=1250.0,
                value=3.0
            )

            latitude = st.number_input(
                "Latitude",
                min_value=32.0,
                max_value=42.0,
                value=35.6
            )

            longitude = st.number_input(
                "Longitude",
                min_value=-125.0,
                max_value=-114.0,
                value=-119.6
            )

        submitted = st.form_submit_button(
            "Predict House Price",
            type="primary"
        )

    if submitted:

        input_data = pd.DataFrame([{
            "MedInc": med_inc,
            "HouseAge": house_age,
            "AveRooms": ave_rooms,
            "AveBedrms": ave_bedrms,
            "Population": population,
            "AveOccup": ave_occup,
            "Latitude": latitude,
            "Longitude": longitude
        }])

        prediction = model.predict(input_data)[0]

        predicted_price = prediction * 100000

        st.success("Prediction generated successfully!")

        st.metric(
            "Estimated Median House Value",
            f"${predicted_price:,.0f}"
        )

        st.caption(
            "Estimated district-level median value in US dollars."
        )

        result = input_data.copy()

        result["Predicted_Median_House_Value_USD"] = predicted_price

        csv = result.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Prediction Result",
            data=csv,
            file_name="house_price_prediction.csv",
            mime="text/csv"
        )

# ==========================================
# MODEL INSIGHTS PAGE
# ==========================================

elif page == "Model Insights":

    st.title("📊 Model Insights")

    st.write(
        "Understand model predictions and evaluation results."
    )

    st.subheader("Feature Importance")

    st.image(
        str(FIGURE_DIR / "feature_importance.png"),
        width="stretch"
    )

    st.subheader("Actual vs Predicted Values")

    st.image(
        str(FIGURE_DIR / "actual_vs_predicted.png"),
        width="stretch"
    )

    st.subheader("Residual Analysis")

    st.image(
        str(FIGURE_DIR / "residual_analysis.png"),
        width="stretch"
    )

    st.info(
        "Feature importance represents the model's relative "
        "importance scores. It does not establish causation."
    )

# ==========================================
# PROJECT INFORMATION PAGE
# ==========================================

elif page == "Project Information":

    st.title("📌 Project Information")

    st.subheader("Project Overview")

    st.write(
        "The House Price Prediction System is an end-to-end "
        "Machine Learning project that estimates California "
        "district-level median house values."
    )

    st.subheader("Technology Stack")

    st.markdown("""
    - Python
    - Pandas and NumPy
    - Matplotlib and Seaborn
    - Scikit-learn
    - Random Forest Regressor
    - Streamlit
    - Joblib
    """)

    st.subheader("Machine Learning Workflow")

    st.markdown("""
    1. Dataset collection
    2. Exploratory Data Analysis
    3. Data preprocessing
    4. Model training
    5. Cross-validation
    6. Hyperparameter tuning
    7. Model evaluation
    8. Streamlit deployment interface
    """)

    st.subheader("Important Dataset Limitation")

    st.write(
        "The California Housing dataset contains aggregated "
        "census block group information. The target is expressed "
        "in units of $100,000 and is capped near 5.0. "
        "Predictions should be treated as educational estimates, "
        "not real-time market valuations."
    )

# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "House Price Prediction System | Python • Scikit-learn • Streamlit"
)
