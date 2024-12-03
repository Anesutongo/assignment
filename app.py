import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('svm_model.joblib')

def predict_wine_type(fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol):
    # Create input DataFrame with all 11 features
    input_data = pd.DataFrame([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]],
                              columns=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'])
    # Make prediction
    prediction = model.predict(input_data)
    return prediction[0]

def main():
    st.title('Wine Type Prediction')
    st.write('Enter the following details to predict the wine type:')

    # Input fields
    fixed_acidity = st.number_input("Fixed Acidity", value=7.0)
    volatile_acidity = st.number_input("Volatile Acidity", value=0.5)
    citric_acid = st.number_input("Citric Acid", value=0.3)
    residual_sugar = st.number_input("Residual Sugar", value=2.0)
    chlorides = st.number_input("Chlorides", value=0.05)
    free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", value=30.0)
    total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", value=150.0)
    density = st.number_input("Density", value=0.995)
    pH = st.number_input("pH", value=3.3)
    sulphates = st.number_input("Sulphates", value=0.6)
    alcohol = st.number_input("Alcohol", value=10.0)

    # Predict Wine type
    if st.button("Predict"):
        prediction = predict_wine_type(fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol)
        st.success(f"The predicted wine type is: {prediction}")

if __name__ == '__main__':
    main()
