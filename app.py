import streamlit as st
import numpy as np
import pickle
import gzip

# -------------------------
# Load the Best Model
# -------------------------
@st.cache_resource
def load_model():
    with gzip.open('best_regression_model.pkl.gz', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

# -------------------------
# Inject Stylish CSS
# -------------------------
st.markdown("""
    <style>
    /* App overall background */
    .stApp {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: #ffffff;
        font-family: 'Poppins', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Main container styling */
    .css-18e3th9 {
        background-color: rgba(255, 255, 255, 0.95);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }

    /* Header text styling */
    h1 {
        color: Black !important;
        font-size: 36px;
        font-weight: 700;
    }

    /* Subheader styling */
    .stMarkdown p {
        color: #333333 !important;
        font-size: 18px;
    }

    /* Input fields styling */
    .stNumberInput>div>div>input {
        border: 2px solid #4a148c;
        border-radius: 12px;
        padding: 0.6rem;
        font-size: 16px;
    }
    .stNumberInput>label {
        font-weight: 500;
        color: #4a148c !important;
    }

    /* Button styling */
    .stButton button {
        background: linear-gradient(45deg, #ff6ec4, #7873f5);
        color: white;
        border: none;
        padding: 0.8rem 1.8rem;
        font-size: 18px;
        font-weight: bold;
        border-radius: 12px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background: linear-gradient(45deg, #7873f5, #ff6ec4);
        transform: scale(1.05);
    }

    /* Success message styling */
    .stAlert {
        border-radius: 12px;
        font-size: 18px;
        color: #2e7d32 !important;
    }

    </style>
""", unsafe_allow_html=True)

# -------------------------
# App Title and Description
# -------------------------
st.markdown("<h1 style='text-align: center;'>🏠 California Housing Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter the details below to predict the housing price in California.</p>", unsafe_allow_html=True)

# -------------------------
# Input Fields in Two Columns
# -------------------------
feature_names = [
    'Median Income', 'House Age', 'Average Rooms', 'Average Bedrooms', 'Population', 'Average Occupation', 'Latitude', 'Longitude'
]

col1, col2 = st.columns(2)
input_data = []

for i, feature in enumerate(feature_names):
    with col1 if i % 2 == 0 else col2:
        value = st.number_input(f"{feature}", format="%.3f")
        input_data.append(value)

# -------------------------
# Prediction Button
# -------------------------
if st.button("Predict"):
    X_input = np.array(input_data).reshape(1, -1)
    prediction = model.predict(X_input)[0]
    st.success(f"🏡 Predicted Housing Price: ${prediction * 100000:.2f}")
