# Adidas Interactive Sales Dashboard

This repository contains a **Streamlit-based interactive sales dashboard** and integrates a **machine learning pipeline** for predicting key sales metrics for Adidas. The application provides rich data visualizations, preprocessing functionalities, and a predictive model to explore and analyze sales performance effectively.

## Features

### **Dashboard Functionalities**
- **Dashboard Header:**
  - Displays the Adidas logo, the current date, and a custom-styled title for the dashboard.
- **Interactive Data Visualizations:**
  - **Total Sales by Retailer (Bar Chart):** Visualizes sales distribution across retailers.
  - **Monthly Sales Trend (Line Chart):** Highlights sales trends over time.
  - **Total Sales and Units Sold by State:** Combines bar and line charts for comparative analysis.
  - **Treemap of Sales by Region and City:** Visualizes hierarchical sales data.
- **Data Downloads:**
  - Allows users to download processed datasets (e.g., retailer-wise sales, monthly sales, etc.) as CSV files.

### **Preprocessing and Data Cleaning**
- **Outlier Removal:** Option to remove outliers using the **Interquartile Range (IQR) method**.
- **Feature Scaling:** Option to standardize numeric data using `StandardScaler` from Scikit-learn.

### **Feature Engineering**
- Categorical variables (e.g., retailer, region, state, etc.) are encoded using **one-hot encoding**.
- Numeric features are standardized if scaling is enabled.

### **Machine Learning Pipeline**
- **Model Selection:** Implements a **Linear Regression model** to predict continuous target variables (e.g., total sales or operating profit).
- **Data Splitting:** Divides the dataset into training and testing sets (80/20 split).
- **Prediction and Analysis:**
  - **Scatter Plot:** Compares actual vs. predicted values.
  - **Feature Importance:** Visualizes model coefficients to identify the impact of each feature on the target variable.
  - **Custom Predictions:** Allows users to input feature values for real-time predictions.

---

## Technologies Used

- **[Streamlit](https://streamlit.io/):** For creating interactive web applications.
- **[Pandas](https://pandas.pydata.org/):** For data manipulation and analysis.
- **[NumPy](https://numpy.org/):** For numerical computations.
- **[Plotly](https://plotly.com/python/):** For interactive data visualizations.
- **[Scikit-learn](https://scikit-learn.org/):** For machine learning model implementation and data preprocessing.
- **[Pillow](https://python-pillow.org/):** For image processing (Adidas logo).
- **Datetime:** For handling date and time.

---

## Machine Learning Model

### **Model Used: Linear Regression**
- A supervised learning model for predicting continuous variables.
- Learns the relationship between selected features and the target variable.

---

## How to Use the Application

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/adidas-dashboard.git
   ```
2. Navigate to the project directory:
   ```bash
   cd adidas-dashboard
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
5. Open the application in your browser to interact with the dashboard.

---

## Purpose of the Application

1. **Data Exploration:** Gain insights into Adidas sales performance across various dimensions (e.g., retailer, region, time).
2. **Trend Analysis:** Identify patterns in sales over time and geographic locations.
3. **Prediction:** Enable forecasting of total sales or operating profit using selected features.

---

## Example Prediction

- **Target Variable:** Total Sales.
- **Input Features:**
  - `Price per Unit`: 50
  - `Units Sold`: 1000
  - `Operating Margin`: 20
- **Predicted Total Sales:** $52,000

Note: Actual predictions depend on the trained model and feature values provided during runtime.

---

## Contributions
Contributions are welcome! Please fork the repository and submit a pull request with your improvements.








