import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Title
st.title("CGPA vs Package Predictor")

# Load dataset
df = pd.read_csv("placement.csv")

st.subheader("Dataset Preview")
st.write(df.head())

# Features and target
X = df[['CGPA']]
y = df['Package']

# Train model
model = LinearRegression()
model.fit(X, y)

# Show coefficients
st.subheader("Model Details")
st.write("Slope (m):", model.coef_[0])
st.write("Intercept (c):", model.intercept_)

# User input
st.subheader("Predict Package")
cgpa_input = st.number_input("Enter CGPA", min_value=0.0, max_value=10.0, step=0.1)

if st.button("Predict"):
    prediction = model.predict([[cgpa_input]])
    st.success(f"Estimated Package: {prediction[0]:.2f} LPA")

# Plot
st.subheader("Graph: CGPA vs Package")

fig, ax = plt.subplots()
ax.scatter(X, y)
ax.plot(X, model.predict(X))
ax.set_xlabel("CGPA")
ax.set_ylabel("Package (LPA)")
ax.set_title("Regression Line")

st.pyplot(fig)