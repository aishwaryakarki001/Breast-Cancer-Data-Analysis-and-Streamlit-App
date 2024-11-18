import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

# Load the saved model and selected features
model = joblib.load('./Model/breast_cancer_model.pkl')
selected_features = joblib.load('./Model/selected_features.pkl')

# Set up the Streamlit app
st.title("Breast Cancer Prediction App")
st.write("This app predicts whether your have breast cancer or not.")

# Load and preprocess the dataset for feature range reference
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Standardize the features
scaler = StandardScaler()
df[data.feature_names] = scaler.fit_transform(df[data.feature_names])

# Sidebar for user inputs
st.sidebar.header("Input Parameters")
def user_input_features():
    # Allow user input only for selected features
    inputs = {feature: st.sidebar.slider(feature, 
                                         float(df[feature].min()), 
                                         float(df[feature].max()), 
                                         float(df[feature].mean())) 
              for feature in selected_features}
    return pd.DataFrame(inputs, index=[0])

# Collect user input
input_df = user_input_features()

# Make predictions using the loaded model
prediction = model.predict(input_df)

# Display prediction results
st.write("Prediction:", "Breast Cancer" if prediction[0] else "Not a Breast Cancer")
