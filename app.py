import streamlit as st
import numpy as np
import pickle
import gzip



st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5f5f5;
        color: #333333;
    }
    .css-18e3th9 {
        padding: 2rem;
        border-radius: 10px;
        background-color: #ffffff;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        text-align: center;
        font-size: 16px;
        border-radius: 5px;
        cursor: pointer;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    </style>
    """, unsafe_allow_html=True
)

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
# App Title
# -------------------------
st.title("🏠 California Housing Price Prediction")
st.write("Enter the details below to predict the housing price.")

# -------------------------
# Input Fields
# -------------------------
# List of feature names in the California dataset
feature_names = [
    'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude'
]

input_data = []
for feature in feature_names:
    value = st.number_input(f"{feature}", format="%.3f")
    input_data.append(value)

# -------------------------
# Prediction Button
# -------------------------
if st.button("Predict"):
    X_input = np.array(input_data).reshape(1, -1)
    prediction = model.predict(X_input)[0]
    st.success(f"🏡 Predicted Housing Price: ${prediction * 100000:.2f}")
