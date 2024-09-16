import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np

# Load the trained model
model = joblib.load('predictive_maintenance_model.pkl')

# Define the input features
features = ['Type', 'Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']

def predict_failure(input_data):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data], columns=features)
    # Predict using the loaded model
    prediction = model.predict(input_df)
    return prediction[0]

# Streamlit app
st.title('Predictive Maintenance - Failure Prediction')
st.write('Enter sensor readings and operating conditions to predict equipment failure.')

# Create input fields for each feature
sensor1 = st.number_input('Type', min_value=0.0, max_value=1000.0, value=0.0)
sensor2 = st.number_input('Air Temperature [K]', min_value=0.0, max_value=1000.0, value=0.0)
sensor3 = st.number_input('Process Temperature [K]', min_value=0.0, max_value=1000.0, value=0.0)
operating_condition1 = st.number_input('Rotational Speed [rpm]', min_value=0.0, max_value=1000.0, value=0.0)
operating_condition2 = st.number_input('Torque [Nm]', min_value=0.0, max_value=1000.0, value=0.0)
operating_condition3 = st.number_input('Tool wear [min]', min_value=0.0, max_value=1000.0, value=0.0)

# When the button is clicked
if st.button('Predict Failure'):
    # Create input data dictionary
    input_data = [sensor1, sensor2, sensor3, operating_condition1, operating_condition2, operating_condition3]
    # Predict
    result = predict_failure(input_data)

    # Display result
    if result == 1:
        st.error('Warning: Equipment is likely to fail!')
    else:
        st.success('Equipment is operating normally.')

        # Plot sample data (dummy data for visualization purposes)
    sample_data = pd.DataFrame({
        'time': np.arange(1, 101),
        'sensor1': np.random.rand(100) * 100,
        'sensor2': np.random.rand(100) * 100,
        'sensor3': np.random.rand(100) * 100,
        'operating_condition1': np.random.rand(100) * 100,
        'operating_condition2': np.random.rand(100) * 100,
    })

    # Line plot of sensor readings over time
    st.subheader('Sensor Readings Over Time')
    fig, ax = plt.subplots()
    ax.plot(sample_data['time'], sample_data['sensor1'], label='Sensor 1')
    ax.plot(sample_data['time'], sample_data['sensor2'], label='Sensor 2')
    ax.plot(sample_data['time'], sample_data['sensor3'], label='Sensor 3')
    ax.set_xlabel('Time')
    ax.set_ylabel('Sensor Value')
    ax.legend()
    st.pyplot(fig)

    # Scatter plot of sensor1 vs sensor2
    st.subheader('Sensor 1 vs Sensor 2')
    fig, ax = plt.subplots()
    ax.scatter(sample_data['sensor1'], sample_data['sensor2'])
    ax.set_xlabel('Sensor 1')
    ax.set_ylabel('Sensor 2')
    st.pyplot(fig)

    # Histogram of sensor1
    st.subheader('Distribution of Sensor 1 Readings')
    fig, ax = plt.subplots()
    ax.hist(sample_data['sensor1'], bins=20, color='blue', alpha=0.7)
    ax.set_xlabel('Sensor 1 Value')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    # Interactive plot using Plotly
    st.subheader('Interactive Scatter Plot (Sensor 1 vs Sensor 3)')
    fig = px.scatter(sample_data, x='sensor1', y='sensor3', title='Sensor 1 vs Sensor 3')
    st.plotly_chart(fig)
