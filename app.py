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
# Inject Custom CSS for Styling
# -------------------------
st.markdown(
    """
    <style>
    /* App background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa, #c3cfe2);
        color: #333333;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    /* Container styling */
    .css-18e3th9 {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    /* Input fields styling */
    .stNumberInput>div>div>input {
        border: 2px solid #4CAF50;
        border-radius: 8px;
        padding: 0.5rem;
    }
    /* Button styling */
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 0.7rem 1.5rem;
        text-align: center;
        font-size: 18px;
        border-radius: 10px;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    /* Success message styling */
    .stAlert {
        border-radius: 10px;
        font-size: 18px;
    }
    </style>
    """, unsafe_allow_html=True
)

# -------------------------
# App Title with Style
# -------------------------
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🏠 California Housing Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px; color: #555;'>Enter the details below to get your predicted housing price!</p>", unsafe_allow_html=True)

# -------------------------
# Input Fields in Two Columns
# -------------------------
feature_names = [
    'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude'
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
