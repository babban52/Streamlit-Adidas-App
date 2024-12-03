import streamlit as st
import pandas as pd
import numpy as np
import datetime
import pickle
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(layout="wide")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)

# Load model from pickle file
with open('https://github.com/babban52/Streamlit-Adidas-App/raw/trialmodel.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the data
df = pd.read_excel("https://github.com/babban52/Streamlit-Adidas-App/raw/trial/Adidas.xlsx")

# Dashboard Title and Logo
image = Image.open('adidas-logo.jpg')
col1, col2 = st.columns([0.1, 0.9])
with col1:
    st.image(image, width=100)

html_title = """
    <style>
    .title-test {
    font-weight:bold;
    padding:5px;
    border-radius:6px;
    }
    </style>
    <center><h1 class="title-test">Adidas Interactive Sales Dashboard</h1></center>"""
with col2:
    st.markdown(html_title, unsafe_allow_html=True)

# Display data summary
if st.checkbox("Show Data Summary"):
    st.write(df.describe())

# Sidebar: Feature selection and prediction target
st.sidebar.header("Prediction Options")
features = st.sidebar.multiselect(
    "Select Features", options=['PriceperUnit', 'UnitsSold', 'OperatingMargin'], 
    default=['PriceperUnit', 'UnitsSold', 'OperatingMargin']
)
target = st.sidebar.selectbox("Select Target", options=['TotalSales', 'OperatingProfit'])

# Prepare features and model input
st.subheader("Make a Prediction")
input_data = {}
for feature in features:
    input_data[feature] = st.number_input(f"{feature}", value=float(df[feature].mean()))
input_df = pd.DataFrame([input_data])

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    st.write(f"### Predicted {target}: {prediction:.2f}")

# Visualization: Actual vs Predicted
if "y_test" in locals():  # Ensure y_test exists from the trained model
    y_pred = model.predict(X_test)  # Reuse the existing X_test data if available
    scatter_fig = px.scatter(
        x=y_test, y=y_pred, labels={'x': 'Actual', 'y': 'Predicted'},
        title="Actual vs Predicted Values"
    )
    scatter_fig.add_shape(
        type="line", x0=y_test.min(), x1=y_test.max(), y0=y_test.min(), y1=y_test.max(),
        line=dict(color="red", dash="dash")
    )
    st.plotly_chart(scatter_fig, use_container_width=True)

# Display model coefficients
if hasattr(model, 'coef_'):
    coefficients = pd.DataFrame({"Feature": features, "Coefficient": model.coef_})
    coeff_fig = px.bar(
        coefficients, x="Feature", y="Coefficient", title="Feature Importance",
        labels={"Coefficient": "Impact on Target"}
    )
    st.plotly_chart(coeff_fig, use_container_width=True)

# Raw Data Download
if st.checkbox("Show Raw Data"):
    st.write(df)
    st.download_button(
        label="Download Data as CSV",
        data=df.to_csv().encode("utf-8"),
        file_name='AdidasData.csv',
        mime='text/csv'
    )
