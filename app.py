import streamlit as st
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = tf.keras.models.load_model('measles_diagnosis_model.keras')

# Load label encoders
le_diagnosis = LabelEncoder()
le_sorethroat = LabelEncoder()

# Load the original data to fit the label encoders
data = pd.read_csv('training.csv')
le_diagnosis.fit(data['diagnosis'])
le_sorethroat.fit(data['sorethroat'])

# Streamlit app
st.title("Measles vs. German Measles Diagnosis")

# User input
st.header("Input Patient Data")

incubation = st.number_input('Incubation period (days)', min_value=0, max_value=20, value=10)
sorethroat = st.selectbox('Sore Throat', ('Yes', 'No'))
temperature = st.number_input('Temperature (°C)', min_value=30.0, max_value=45.0, value=36.5)

# Encode sorethroat input
sorethroat_encoded = le_sorethroat.transform([sorethroat[0]])[0]

# Prepare the data for prediction
input_data = pd.DataFrame({
    'incubation': [incubation],
    'sorethroat': [sorethroat_encoded],
    'temperature': [temperature]
})

# Predict the diagnosis
prediction = model.predict(input_data)
predicted_class = (prediction > 0.5).astype(int).flatten()
predicted_label = le_diagnosis.inverse_transform(predicted_class)[0]

# Display the result
st.header("Diagnosis Prediction")
st.write(f"The predicted diagnosis is: **{predicted_label}**")
