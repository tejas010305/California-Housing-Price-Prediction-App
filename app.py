import streamlit as st
import numpy as np
import pickle
import gzip

# -------------------------
# Load the Best Model
# -------------------------
@st.cache_resource
def load_model():
    """Load the pre-trained model from a gzipped pickle file."""
    with gzip.open('best_regression_model.pkl.gz', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

# -------------------------
# Inject Stylish CSS
# -------------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;700&display=swap');

    /* App overall background */
    .stApp {
        background-color: #f0f2f6;
        font-family: 'Poppins', sans-serif;
    }

    /* Main container styling */
    .css-18e3th9 {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.05);
    }

    /* Navbar styling */
    .navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 2rem;
        background-color: #ffffff;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: -2rem -2rem 2rem -2rem; /* Extend to edges of the container */
    }
    .navbar-brand {
        font-size: 24px;
        font-weight: bold;
        color: #333;
    }

    /* Header text styling */
    h1 {
        color: #2c3e50 !important;
        font-size: 32px;
        font-weight: 700;
        text-align: center;
    }

    /* Subheader styling */
    .stMarkdown p {
        color: #555 !important;
        font-size: 16px;
        text-align: center;
    }

    /* Input fields styling */
    .stNumberInput>div>div>input {
        border: 2px solid #e0e0e0;
        border-radius: 12px;
        padding: 0.6rem;
        font-size: 16px;
        transition: border-color 0.3s ease;
    }
    .stNumberInput>div>div>input:focus {
        border-color: #b39ddb;
    }
    .stNumberInput>label {
        font-weight: 500;
        color: #333 !important;
    }

    /* Button styling with gradient */
    .stButton button {
        background: linear-gradient(45deg, #8e44ad, #9b59b6);
        color: #ffffff;
        border: none;
        padding: 0.8rem 1.8rem;
        font-size: 18px;
        font-weight: bold;
        border-radius: 12px;
        cursor: pointer;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton button:hover {
        background: linear-gradient(45deg, #9b59b6, #8e44ad);
        transform: scale(1.03);
        box-shadow: 0 4px 15px rgba(142, 68, 173, 0.3);
    }

    /* Success message styling */
    .stAlert {
        border-radius: 12px;
        font-size: 18px;
        text-align: center;
        background-color: #d4edda !important;
        color: #155724 !important;
        border-color: #c3e6cb !important;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 2rem 0 1rem 0;
        color: #aaa;
        font-size: 14px;
    }
    .footer a {
        color: #8e44ad;
        text-decoration: none;
    }
    .footer a:hover {
        text-decoration: underline;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------
# Navigation Bar
# -------------------------
st.markdown("""
<nav class="navbar">
  <div class="navbar-brand">🏠 PricePredict CA</div>
</nav>
""", unsafe_allow_html=True)


# -------------------------
# App Title and Description
# -------------------------
st.markdown("<h1>California Housing Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p>Enter the details below to get an estimated housing price in California.</p>", unsafe_allow_html=True)
st.write("---") # Adds a horizontal line for separation

# -------------------------
# Input Fields in Two Columns
# -------------------------
feature_names = [
    'Median Income', 'House Age', 'Average Rooms', 'Average Bedrooms', 'Population', 'Average Occupation', 'Latitude', 'Longitude'
]

col1, col2 = st.columns(2)
input_data = []

# Create input fields dynamically
for i, feature in enumerate(feature_names):
    # Use a key for each input to ensure they are unique
    key = f"input_{i}"
    # Alternate between columns for a balanced layout
    container = col1 if i % 2 == 0 else col2
    with container:
        value = st.number_input(f"{feature}", format="%.3f", key=key)
        input_data.append(value)

# Add some vertical space before the button
st.write("") 
st.write("") 

# -------------------------
# Prediction Button
# -------------------------
if st.button("Predict Price"):
    X_input = np.array(input_data).reshape(1, -1)
    prediction = model.predict(X_input)[0]
    # Display prediction in a formatted success box
    st.success(f"Predicted Housing Price: ${prediction * 100000:,.2f}")

# -------------------------
# Footer
# -------------------------
st.markdown("""
<div class="footer">
  <p>Built by Tejas | <a href="mailto:tejas@example.com">tejas@example.com</a></p>
</div>
""", unsafe_allow_html=True)
