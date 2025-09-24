import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model

# --- Theme Selector ---
theme = st.sidebar.radio("Select Theme 🌗", ["Light", "Dark"])
if theme == "Dark":
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #121212;
            color: #FFFFFF;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 18px;
            height: 50px;
            width: 100%;
            border-radius: 10px;
        }
        .stSlider>div>div>div>div {
            color: #4CAF50;
        }
        </style>
        """, unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #f5f5f5;
            color: #000000;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 18px;
            height: 50px;
            width: 100%;
            border-radius: 10px;
        }
        .stSlider>div>div>div>div {
            color: #4CAF50;
        }
        </style>
        """, unsafe_allow_html=True
    )

# --- Load Model ---
try:
    model = load_model('ann_model.h5', compile=False)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {str(e)}")
    st.info("Trying alternative loading method...")
    try:
        model = tf.keras.models.load_model('ann_model.h5', compile=False)
        st.success("✅ Model loaded with alternative method!")
    except Exception as e2:
        st.error(f"❌ Failed to load model: {str(e2)}")
        st.stop()

# --- Load Preprocessors ---
try:
    with open('label_encoder_gender.pkl', 'rb') as f:
        label_encoder_gender = pickle.load(f)
    with open('onehot_encoder_geography.pkl', 'rb') as f:
        onehot_encoder_geography = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    st.success("✅ Preprocessors loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading preprocessors: {str(e)}")
    st.stop()

# --- App Title ---
st.title("📊 Customer Churn Prediction")
st.markdown("Predict whether a customer is likely to churn using our trained ANN model.")
st.markdown("---")

# --- Input Columns ---
col1, col2 = st.columns(2)
with col1:
    geography = st.selectbox('🌍 Geography', onehot_encoder_geography.categories_[0])
    gender = st.selectbox('👤 Gender', ['Male', 'Female'])
    age = st.slider('🎂 Age', 18, 92, 30)
    balance = st.number_input('💰 Balance', min_value=0.0, value=50000.0, step=1000.0)
    credit_score = st.number_input('📝 Credit Score', min_value=300, max_value=850, value=650)

with col2:
    estimated_salary = st.number_input('💵 Estimated Salary', min_value=0.0, value=50000.0, step=1000.0)
    tenure = st.slider('🕒 Tenure', 0, 10, 3)
    num_of_products = st.slider('📦 Number of Products', 1, 4, 1)
    has_cr_card = st.selectbox('💳 Has Credit Card', [0, 1])
    is_active_member = st.selectbox('✅ Is Active Member', [0, 1])

st.markdown("---")

# --- Prepare Input Data ---
try:
    gender_encoded = label_encoder_gender.transform([gender])[0]
except:
    gender_encoded = 1 if gender == "Male" else 0
    st.warning("⚠️ Using manual gender encoding as fallback")

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [gender_encoded],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

geo_encoded = onehot_encoder_geography.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geography.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
input_data_scaled = scaler.transform(input_data)

# --- Prediction ---
if st.button('Predict Churn'):
    try:
        prediction = model.predict(input_data_scaled)
        prediction_proba = prediction[0][0]

        st.markdown(f"### 🔮 Churn Probability: **{prediction_proba:.2f}**")
        if prediction_proba > 0.5:
            st.error("⚠️ The customer is likely to churn.")
        else:
            st.success("✅ The customer is not likely to churn.")
    except Exception as e:
        st.error(f"❌ Error making prediction: {str(e)}")
        st.error("Please check if the model and preprocessors are compatible.")
