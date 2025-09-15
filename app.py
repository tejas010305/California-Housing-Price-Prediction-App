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
