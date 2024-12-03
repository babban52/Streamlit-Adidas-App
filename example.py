import streamlit as st
import pandas as pd
import numpy as np
import datetime
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Set page configuration
st.set_page_config(page_title="Adidas Sales Dashboard", layout="wide")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)

# Load the data
data_url = "Adidas.xlsx"
df = pd.read_excel(data_url)

# Header and Title
image = Image.open('adidas-logo.jpg')
col1, col2 = st.columns([0.1, 0.9])
with col1:
    st.image(image, width=100)
with col2:
    st.markdown("""
    <style>
        .title-test {
            font-weight: bold;
            font-size: 2rem;
            padding: 5px;
            border-radius: 6px;
        }
    </style>
    <center><h1 class="title-test">Adidas Interactive Sales Dashboard</h1></center>
    """, unsafe_allow_html=True)

# Show the last updated time
st.sidebar.write(f"**Last updated:** {datetime.datetime.now().strftime('%d %B %Y')}")

# Preprocessing
# Encode all categorical variables using LabelEncoder except the SalesMethod column
categorical_cols = df.select_dtypes(include=['object']).columns
label_encoders = {}

for col in categorical_cols:
    if col != 'SalesMethod':  # Exclude 'SalesMethod' for one-hot encoding later
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le  # Store the encoder for future use

# One-hot encode the SalesMethod column
df = pd.get_dummies(df, columns=['SalesMethod'], drop_first=True)

# Feature scaling
scale_data = True  # Ensure this is set as True if you want to apply scaling
if scale_data:
    scaler = StandardScaler()
    numeric_cols = ['PriceperUnit', 'UnitsSold', 'OperatingProfit', 'OperatingMargin']
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Visualizations and Model Training

# Retailer-wise Sales Visualization
col3, col4 = st.columns(2)
with col3:
    st.subheader("Total Sales by Retailer")
    retailer_fig = px.bar(
        df, x="Retailer", y="TotalSales",
        labels={"TotalSales": "Total Sales ($)"},
        title="Retailer-wise Total Sales",
        template="gridon", height=500
    )
    st.plotly_chart(retailer_fig, use_container_width=True)

# Convert the `InvoiceDate` column to datetime if not already
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# Extract the year from the `InvoiceDate` column
df["Year"] = df["InvoiceDate"].dt.year

# Streamlit multiselect to select two years for comparison
selected_years = st.multiselect(
    "Select Two Years to Compare Sales",
    options=sorted(df["Year"].unique()),
    default=sorted(df["Year"].unique())[:2],
    format_func=lambda x: f"Year {x}",
)

# Ensure exactly two years are selected
if len(selected_years) != 2:
    st.warning("Please select exactly two years to compare sales.")
else:
    # Filter the dataframe for the selected years
    filtered_df = df[df["Year"].isin(selected_years)]

    # Create the Month_Year column for grouping
    filtered_df["Month_Year"] = filtered_df["InvoiceDate"].dt.strftime("%b")

    # Group by Month_Year and Year to calculate monthly sales
    monthly_sales = filtered_df.groupby(["Year", "Month_Year"])["TotalSales"].sum().reset_index()

    # Sort by Month_Year for proper chronological order
    monthly_sales["Month_Year"] = pd.Categorical(
        monthly_sales["Month_Year"], 
        categories=pd.date_range(start="2023-01-01", end="2023-12-31", freq="M").strftime("%b"),
        ordered=True
    )
    monthly_sales = monthly_sales.sort_values(by=["Year", "Month_Year"])

    # Create the line plot with separate lines for each year
    time_fig = px.line(
        monthly_sales,
        x="Month_Year",
        y="TotalSales",
        color="Year",
        labels={"TotalSales": "Total Sales ($)", "Month_Year": "Month"},
        title=f"Monthly Sales Trends Comparison: {selected_years[0]} vs {selected_years[1]}",
        template="plotly_white",
    )

    # Display the plot in Streamlit
    with col4:
        st.subheader("Sales Over Time Comparison")
        st.plotly_chart(time_fig, use_container_width=True)

# State-wise Sales and Units Sold Visualization
st.subheader("State-wise Sales and Units Sold")
state_data = df.groupby("State")[["TotalSales", "UnitsSold"]].sum().reset_index()
state_fig = go.Figure()
state_fig.add_trace(go.Bar(x=state_data["State"], y=state_data["TotalSales"], name="Total Sales"))
state_fig.add_trace(go.Scatter(x=state_data["State"], y=state_data["UnitsSold"], mode="lines+markers", name="Units Sold", yaxis="y2"))
state_fig.update_layout(
    title="Sales and Units Sold by State",
    xaxis_title="State",
    yaxis=dict(title="Total Sales"),
    yaxis2=dict(title="Units Sold", overlaying="y", side="right"),
    legend=dict(x=1, y=1.1),
    template="plotly_white"
)
st.plotly_chart(state_fig, use_container_width=True)

# Treemap Visualization
st.subheader("Sales Distribution by Region and City")
treemap_data = df.groupby(["Region", "City"])["TotalSales"].sum().reset_index()
treemap_fig = px.treemap(
    treemap_data, path=["Region", "City"], values="TotalSales",
    color="TotalSales", hover_data=["TotalSales"],
    color_continuous_scale="Blues",
    title="Sales by Region and City"
)
st.plotly_chart(treemap_fig, use_container_width=True)

# Raw Data Section
if st.checkbox("Show Raw Data"):
    st.write(df)

# Model Training and Prediction
st.sidebar.header("Model Training")
features = st.sidebar.multiselect(
    "Select Features", options=df.select_dtypes(include=[np.number]).columns,
    default=["PriceperUnit", "UnitsSold", "OperatingMargin"]
)
target = st.sidebar.selectbox("Select Target", options=["TotalSales", "OperatingProfit"])

# Data Splitting and Scaling
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Model Performance
st.subheader("Model Performance")
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
st.write(f"**Mean Squared Error:** {mse:.2f}")
st.write(f"**R-squared Score:** {r2:.2f}")

# Prediction Visualization
st.subheader("Predictions: Actual vs Predicted")
pred_fig = px.scatter(
    x=y_test, y=y_pred,
    labels={"x": "Actual", "y": "Predicted"},
    title="Actual vs Predicted Sales"
)
pred_fig.add_shape(
    type="line", x0=y_test.min(), x1=y_test.max(), y0=y_test.min(), y1=y_test.max(),
    line=dict(color="red", dash="dash")
)
st.plotly_chart(pred_fig, use_container_width=True)

# Feature importance (coefficients)
coefficients = pd.DataFrame({"Feature": features, "Coefficient": model.coef_})
coeff_fig = px.bar(
    coefficients, x="Feature", y="Coefficient", title="Feature Importance",
    labels={"Coefficient": "Impact on Target"}
)
st.plotly_chart(coeff_fig, use_container_width=True)

# Custom Prediction Form
st.subheader("Make a Prediction")
user_input = {feature: st.number_input(f"{feature}", value=float(df[feature].mean())) for feature in features}
if st.button("Predict"):
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]
    st.write(f"### Predicted {target}: ${prediction:,.2f}")
