import streamlit as st
import pandas as pd
import numpy as np
import datetime
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Set page configuration
st.set_page_config(layout="wide")
st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)

# Load the data
df = pd.read_excel("https://github.com/babban52/Streamlit-Adidas-App/raw/main/Adidas.xlsx")

st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)
image = Image.open('adidas-logo.jpg')

col1, col2 = st.columns([0.1,0.9])
with col1:
    st.image(image,width=100)

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

col3, col4, col5 = st.columns([0.1,0.45,0.45])
with col3:
    box_date = str(datetime.datetime.now().strftime("%d %B %Y"))
    st.write(f"Last updated by:  \n {box_date}")

with col4:
    fig = px.bar(df, x = "Retailer", y = "TotalSales", labels={"TotalSales" : "Total Sales {$}"},
                 title = "Total Sales by Retailer", hover_data=["TotalSales"],
                 template="gridon",height=500)
    st.plotly_chart(fig,use_container_width=True)

_, view1, dwn1, view2, dwn2 = st.columns([0.15,0.20,0.20,0.20,0.20])
with view1:
    expander = st.expander("Retailer wise Sales")
    data = df[["Retailer","TotalSales"]].groupby(by="Retailer")["TotalSales"].sum()
    expander.write(data)
with dwn1:
    st.download_button("Get Data", data = data.to_csv().encode("utf-8"),
                       file_name="RetailerSales.csv", mime="text/csv")

df["Month_Year"] = df["InvoiceDate"].dt.strftime("%b'%y")
result = df.groupby(by = df["Month_Year"])["TotalSales"].sum().reset_index()

with col5:
    fig1 = px.line(result, x = "Month_Year", y = "TotalSales", title="Total Sales Over Time",
                   template="gridon")
    st.plotly_chart(fig1,use_container_width=True)

with view2:
    expander = st.expander("Monthly Sales")
    data = result
    expander.write(data)
with dwn2:
    st.download_button("Get Data", data = result.to_csv().encode("utf-8"),
                       file_name="Monthly Sales.csv", mime="text/csv")
    
st.divider()

result1 = df.groupby(by="State")[["TotalSales","UnitsSold"]].sum().reset_index()

# add the units sold as a line chart on a secondary y-axis
fig3 = go.Figure()
fig3.add_trace(go.Bar(x = result1["State"], y = result1["TotalSales"], name = "Total Sales"))
fig3.add_trace(go.Scatter(x=result1["State"], y = result1["UnitsSold"], mode = "lines",
                          name ="Units Sold", yaxis="y2"))
fig3.update_layout(
    title = "Total Sales and Units Sold by State",
    xaxis = dict(title="State"),
    yaxis = dict(title="Total Sales", showgrid = False),
    yaxis2 = dict(title="Units Sold", overlaying = "y", side = "right"),
    template = "gridon",
    legend = dict(x=1,y=1.1)
)
_, col6 = st.columns([0.1,1])
with col6:
    st.plotly_chart(fig3,use_container_width=True)

_, view3, dwn3 = st.columns([0.5,0.45,0.45])
with view3:
    expander = st.expander("View Data for Sales by Units Sold")
    expander.write(result1)
with dwn3:
    st.download_button("Get Data", data = result1.to_csv().encode("utf-8"), 
                       file_name = "Sales_by_UnitsSold.csv", mime="text/csv")
st.divider()

_, col7 = st.columns([0.1,1])
treemap = df[["Region","City","TotalSales"]].groupby(by = ["Region","City"])["TotalSales"].sum().reset_index()

def format_sales(value):
    if value >= 0:
        return '{:.2f} Lakh'.format(value / 1_000_00)

treemap["TotalSales (Formatted)"] = treemap["TotalSales"].apply(format_sales)

fig4 = px.treemap(treemap, path = ["Region","City"], values = "TotalSales",
                  hover_name = "TotalSales (Formatted)",
                  hover_data = ["TotalSales (Formatted)"],
                  color = "City", height = 700, width = 600)
fig4.update_traces(textinfo="label+value")

with col7:
    st.subheader(":point_right: Total Sales by Region and City in Treemap")
    st.plotly_chart(fig4,use_container_width=True)

_, view4, dwn4 = st.columns([0.5,0.45,0.45])
with view4:
    result2 = df[["Region","City","TotalSales"]].groupby(by=["Region","City"])["TotalSales"].sum()
    expander = st.expander("View data for Total Sales by Region and City")
    expander.write(result2)
with dwn4:
    st.download_button("Get Data", data = result2.to_csv().encode("utf-8"),
                                        file_name="Sales_by_Region.csv", mime="text.csv")

_,view5, dwn5 = st.columns([0.5,0.45,0.45])
with view5:
    expander = st.expander("View Sales Raw Data")
    expander.write(df)
with dwn5:
    st.download_button("Get Raw Data", data = df.to_csv().encode("utf-8"),
                       file_name = "SalesRawData.csv", mime="text/csv")
st.divider()

# Preprocessing
st.sidebar.header("Preprocessing Options")
remove_outliers = st.sidebar.checkbox("Remove Outliers", value=True)
scale_data = st.sidebar.checkbox("Scale Data", value=True)

if remove_outliers:
    def remove_outlier(column):
        Q1, Q3 = column.quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower_bound = Q1 - (1.5 * IQR)
        upper_bound = Q3 + (1.5 * IQR)
        return column.clip(lower_bound, upper_bound)

    for col in df.select_dtypes(include=np.number).columns:
        df[col] = remove_outlier(df[col])

# Encoding categorical variables
categorical_cols = ['Retailer', 'Region', 'State', 'City', 'Product', 'SalesMethod']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Feature scaling
if scale_data:
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    numeric_cols = ['PriceperUnit', 'UnitsSold', 'OperatingProfit', 'OperatingMargin']
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Feature selection
features = st.sidebar.multiselect(
    "Select Features", options=df.columns, default=['PriceperUnit', 'UnitsSold', 'OperatingMargin']
)
target = st.sidebar.selectbox("Select Target", options=['TotalSales', 'OperatingProfit'])

# Splitting data
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)


# Visualization
st.title("Adidas Sales Prediction Dashboard")

# Display data
if st.checkbox("Show Raw Data"):
    st.subheader("Raw Data")
    st.write(df)

# Scatter plot for predictions
scatter_fig = px.scatter(
    x=y_test, y=y_pred, labels={'x': 'Actual', 'y': 'Predicted'},
    title="Actual vs Predicted Values"
)
scatter_fig.add_shape(
    type="line", x0=y_test.min(), x1=y_test.max(), y0=y_test.min(), y1=y_test.max(),
    line=dict(color="red", dash="dash")
)
st.plotly_chart(scatter_fig, use_container_width=True)

# Feature importance (coefficients)
coefficients = pd.DataFrame({"Feature": features, "Coefficient": model.coef_})
coeff_fig = px.bar(
    coefficients, x="Feature", y="Coefficient", title="Feature Importance",
    labels={"Coefficient": "Impact on Target"}
)
st.plotly_chart(coeff_fig, use_container_width=True)

# Prediction form
st.subheader("Make a Prediction")
input_data = {}
for feature in features:
    input_data[feature] = st.number_input(f"{feature}", value=float(df[feature].mean()))
input_df = pd.DataFrame([input_data])

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    st.write(f"### Predicted {target}: {prediction:.2f}")
