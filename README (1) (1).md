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

---

## Screenshots

![newplot](https://github.com/user-attachments/assets/355f0a7a-887c-49df-86c4-6ec537ffba9e)
### Total Sales by Retailer
The bar chart **"Total Sales by Retailer"** illustrates Adidas' total sales (in USD) across various retail partners. It provides a clear comparison of performance among key retailers, including **Foot Locker**, **Walmart**, **Sports Direct**, **West Gear**, **Kohl's**, and **Amazon**. 

#### Key Insights:
- **Top Performers:** 
  - **West Gear** and **Foot Locker** emerge as the leading retail partners, each generating over $200M in sales.
  - **Sports Direct** ranks third, with sales nearing $175M.
- **Lower Performers:** 
  - Retailers such as **Walmart**, **Kohl's**, and **Amazon** report comparatively lower sales volumes, ranging between $50M and $100M.

#### Business Relevance:
This visualization highlights the disparities in sales performance across different retail channels. Adidas can use these insights to:
1. **Strengthen High-Performing Partnerships:** Invest in strengthening collaborations with top retailers like West Gear and Foot Locker.
2. **Enhance Underperforming Channels:** Develop strategies to improve sales in channels such as Walmart, Kohl's, and Amazon.
3. **Strategic Resource Allocation:** Prioritize marketing and inventory allocation based on retailer-specific performance.

This data-driven analysis supports Adidas in optimizing its sales strategy and enhancing retail channel effectiveness.


![newplot (1)](https://github.com/user-attachments/assets/f2c5071c-c1b4-4691-9faf-ec9bba60f9b4)
### Total Sales Over Time
The chart titled **"Total Sales Over Time"** illustrates Adidas' monthly sales trends over the analyzed time period. The sales data is plotted with **total sales (in USD)** on the y-axis and time (month-year) on the x-axis.

#### Key Insights:
1. **Seasonal Trends:**
   - Sales exhibit a recurring pattern with **peaks observed during the holiday season (November-December)**, likely driven by holiday shopping and seasonal promotions.
   - **Dips in sales** are seen in the months following the holiday season, reflecting post-holiday slowdown.

2. **Periodic Fluctuations:**
   - There is consistent variability in sales across months, indicating external factors such as promotional campaigns, product launches, or seasonal demand influencing sales.

3. **Stability in Long-Term Performance:**
   - Despite monthly fluctuations, the overall sales trend appears stable with repeated high points around key periods.

#### Business Relevance:
This analysis helps Adidas to:
- **Capitalize on Peak Periods:** Align inventory and marketing efforts with high-demand periods (e.g., holiday season).
- **Address Off-Peak Periods:** Develop strategies to boost sales during slower months through targeted promotions or product diversification.
- **Long-Term Forecasting:** Utilize historical patterns to improve the accuracy of future sales predictions.

This visualization is essential for identifying sales seasonality and guiding decision-making to optimize revenue throughout the year.


![newplot (2)](https://github.com/user-attachments/assets/56ee8583-5a29-40e7-81ac-a2397a911a66)
### Total Sales and Units Sold by State
This chart titled **"Total Sales and Units Sold by State"** showcases the relationship between total revenue (blue bars) and the number of units sold (orange line) for Adidas products across various U.S. states.

#### Key Insights:
1. **High-Performing States:**
   - States like **California**, **Texas**, and **New York** demonstrate significantly higher sales and units sold compared to others, with California leading both metrics.
   - These states are likely major markets for Adidas, benefiting from larger populations and higher demand.

2. **Moderate Performers:**
   - States such as **Florida**, **Georgia**, and **Illinois** show balanced sales figures with moderate units sold, indicating consistent consumer demand.

3. **Low-Performing States:**
   - States like **Wyoming**, **Montana**, and **Vermont** have lower sales and units sold, reflecting limited market penetration or smaller populations.

#### Business Relevance:
This analysis provides valuable insights into regional performance, helping Adidas to:
- **Focus on High-Performing Regions:** Strengthen marketing campaigns and product availability in high-performing states like California, Texas, and New York.
- **Identify Growth Opportunities:** Investigate strategies to improve sales in low-performing states.
- **Optimize Supply Chain:** Align inventory and distribution efforts with regional demand trends.

This chart aids Adidas in tailoring state-specific strategies to maximize sales and improve market reach.


![newplot (3)](https://github.com/user-attachments/assets/8a89caa6-9e15-415d-a6ae-7816d6f6065e)
### Sales Distribution Treemap
The chart visualizes Adidas sales distribution across various cities and regions in the United States. Each region (West, Northeast, South, Southeast, and Midwest) is represented by a group of cities, with the size of each rectangle indicating the relative sales volume for that city. 

- **Top-performing cities**: Cities like **New York (39.58M)**, **San Francisco (34.53M)**, and **Miami (31.60M)** dominate their respective regions in sales.
- **Regional highlights**: The **Northeast** leads with significant contributions from New York and Albany, while the **Southeast** shows strong sales from Miami and Charleston.
- **Diverse performance**: The sales are well-distributed, showcasing how different regions contribute uniquely to the overall Adidas sales.

This chart provides a clear breakdown of sales distribution, helping identify high-performing markets and potential areas for growth.


![newplot (4)](https://github.com/user-attachments/assets/41c61d45-be59-4b22-bee6-c0fcb08862f4)
### Actual vs. Predicted Sales Plot
This scatter plot visualizes the **Actual vs. Predicted sales values** for Adidas, highlighting the performance of the predictive model used in this project. 

- **Insights from the plot**:
  - The red dashed line represents the ideal scenario where predictions perfectly match actual sales values.
  - Most data points are closely aligned with the red line, indicating that the model predictions are fairly accurate.
  - There is a slight dispersion at higher sales values, suggesting minor deviations in predictions for high-performing regions or products.

- **Significance**:
  This chart demonstrates the model's ability to generalize well across various sales ranges, supporting reliable forecasting for inventory planning, marketing, and strategic decisions.

This plot serves as an evaluation metric for the predictive model and confirms its utility in providing actionable insights for Adidas sales trends.


![newplot (5)](https://github.com/user-attachments/assets/68b330fa-6c5e-44c9-ad54-cbb291cd69cf)
### Feature Importance Analysis
This bar chart illustrates the relative importance of key features used in the predictive model for Adidas sales forecasting. The impact of each feature on the target variable (sales) is measured, highlighting the most influential factors.

- **Key Findings**:
  - **UnitsSold** has the highest impact on sales predictions, indicating that the number of units sold is the most critical factor in determining revenue.
  - **PricePerUnit** contributes significantly, showing that pricing strategies play a vital role in influencing overall sales.
  - **OperatingMargin** has a smaller but noticeable impact, suggesting its relevance in financial performance and sales outcomes.

- **Significance**:
  This feature importance analysis provides actionable insights for decision-makers to prioritize key drivers like optimizing pricing strategies and increasing unit sales to maximize revenue.

This chart reinforces the model's interpretability, ensuring its predictions are rooted in relevant business factors.



