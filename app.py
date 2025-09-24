import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
import pickle
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

##Load the ANN train model and pickle files
try:
    # Try loading with compile=False to avoid compilation issues
    model = load_model('ann_model.h5', compile=False)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.info("Trying alternative loading method...")
    try:
        # Alternative loading method
        model = tf.keras.models.load_model('ann_model.h5', compile=False)
        st.success("Model loaded with alternative method!")
    except Exception as e2:
        st.error(f"Failed to load model: {str(e2)}")
        st.stop()

try:
    with open('label_encoder_gender.pkl', 'rb') as f:
        label_encoder_gender = pickle.load(f)
    with open('onehot_encoder_geography.pkl', 'rb') as f:
        onehot_encoder_geography = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    st.success("Preprocessors loaded successfully!")
except Exception as e:
    st.error(f"Error loading preprocessors: {str(e)}")
    st.stop()
    


## streamlit app
st.title('Customer Churn PRediction')

# User input
geography = st.selectbox('Geography', onehot_encoder_geography.categories_[0])
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
# Try to use the label encoder, fall back to manual encoding if it fails
try:
    gender_encoded = label_encoder_gender.transform([gender])[0]
except:
    # Manual fallback encoding - adjust this based on your training data
    # Common encoding: Female=0, Male=1
    gender_encoded = 1 if gender == "Male" else 0
    st.warning("Using manual gender encoding as fallback")

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

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geography.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geography.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict churn
if st.button('Predict Churn'):
    try:
        prediction = model.predict(input_data_scaled)
        prediction_proba = prediction[0][0]

        st.write(f'Churn Probability: {prediction_proba:.2f}')

        if prediction_proba > 0.5:
            st.write('The customer is likely to churn.')
        else:
            st.write('The customer is not likely to churn.')
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.error("Please check if the model and preprocessors are compatible.")
