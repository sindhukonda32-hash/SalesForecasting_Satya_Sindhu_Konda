import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Sales Forecasting & Demand Intelligence",
    page_icon="📈",
    layout="wide"
)
#Add this here
st.set_page_config(
    page_title="Sales Forecasting & Demand Intelligence",
    page_icon="📈",
    layout="wide"
)

# ADD THIS HERE
st.markdown("""
<style>
.main {
    background-color: #f8f9fa;
}

div[data-testid="metric-container"] {
    background-color:#ffffff;
    padding:18px;
    border-radius:12px;
    box-shadow:0px 2px 10px rgba(0,0,0,0.12);
}

h1,h2,h3{
    color:#0f172a;
}
</style>
""", unsafe_allow_html=True)

st.title("📈 End-to-End Sales Forecasting & Demand Intelligence System")
st.markdown("### AI Powered Business Dashboard")
st.markdown("---")

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
@st.cache_data
def load_data():

    print("Reading train.csv...")
    sales = pd.read_csv("train.csv")
    print("Reading vgsales.csv...")
    games = pd.read_csv("vgsales.csv")

    sales["Order Date"] = pd.to_datetime(
        sales["Order Date"],
        dayfirst=True,
        format="mixed"
    )

    sales["Ship Date"] = pd.to_datetime(
        sales["Ship Date"],
        dayfirst=True,
        format="mixed"
    )

    sales["Year"] = sales["Order Date"].dt.year
    sales["Month"] = sales["Order Date"].dt.month
    sales["Quarter"] = sales["Order Date"].dt.quarter
    sales["Week"] = sales["Order Date"].dt.isocalendar().week

    return sales, games


sales_df, games_df = load_data()

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title("📊 Navigation")

page = st.sidebar.radio(
    "Choose Dashboard",
    [
        "🏠 Home",
        "📊 Sales Dashboard",
        "📈 Forecast Explorer",
        "🚨 Anomaly Detection",
        "📦 Demand Segmentation",
        "ℹ️ About"
    ]
)

st.sidebar.markdown("---")

region = st.sidebar.selectbox(
    "Select Region",
    ["All"] + list(sales_df["Region"].unique())
)

category = st.sidebar.selectbox(
    "Select Category",
    ["All"] + list(sales_df["Category"].unique())
)

# ---------------------------------------------------
# FILTER DATA
# ---------------------------------------------------

filtered_df = sales_df.copy()

if region != "All":
    filtered_df = filtered_df[
        filtered_df["Region"] == region
    ]

if category != "All":
    filtered_df = filtered_df[
        filtered_df["Category"] == category
    ]

# ---------------------------------------------------
# HOME PAGE
# ---------------------------------------------------

if page == "🏠 Home":

    st.header("Business Overview Dashboard")

    total_sales = filtered_df["Sales"].sum()

    total_orders = filtered_df["Order ID"].nunique()

    avg_sales = filtered_df["Sales"].mean()

    avg_shipping = (
        filtered_df["Ship Date"] -
        filtered_df["Order Date"]
    ).dt.days.mean()

    c1,c2,c3,c4 = st.columns(4)

    c1.metric(
        "💰 Total Sales",
        f"${total_sales:,.0f}"
    )

    c2.metric(
        "📦 Orders",
        total_orders
    )

    c3.metric(
        "💵 Average Sale",
        f"${avg_sales:,.2f}"
    )

    c4.metric(
        "🚚 Avg Shipping Days",
        round(avg_shipping,2)
    )

    st.markdown("---")

    yearly_sales = (
        filtered_df
        .groupby("Year")["Sales"]
        .sum()
        .reset_index()
    )

    fig = px.bar(
        yearly_sales,
        x="Year",
        y="Sales",
        color="Sales",
        title="Yearly Sales"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    monthly_sales = (
        filtered_df
        .groupby("Month")["Sales"]
        .sum()
        .reset_index()
    )

    fig2 = px.line(
        monthly_sales,
        x="Month",
        y="Sales",
        markers=True,
        title="Monthly Sales Trend"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    col1,col2 = st.columns(2)

    with col1:

        category_sales = (
            filtered_df
            .groupby("Category")["Sales"]
            .sum()
            .reset_index()
        )

        fig3 = px.pie(
            category_sales,
            values="Sales",
            names="Category",
            title="Sales by Category"
        )

        st.plotly_chart(
            fig3,
            use_container_width=True
        )

    with col2:

        region_sales = (
            filtered_df
            .groupby("Region")["Sales"]
            .sum()
            .reset_index()
        )

        fig4 = px.bar(
            region_sales,
            x="Region",
            y="Sales",
            color="Region",
            title="Sales by Region"
        )

        st.plotly_chart(
            fig4,
            use_container_width=True
        )
        # ---------------------------------------------------
# SALES DASHBOARD
# ---------------------------------------------------

elif page == "📊 Sales Dashboard":

    st.header("📊 Sales Dashboard")

    st.subheader("Dataset Preview")

    st.dataframe(filtered_df.head())

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        yearly = (
            filtered_df.groupby("Year")["Sales"]
            .sum()
            .reset_index()
        )

        fig = px.bar(
            yearly,
            x="Year",
            y="Sales",
            color="Sales",
            title="Yearly Sales"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        monthly = (
            filtered_df.groupby("Month")["Sales"]
            .sum()
            .reset_index()
        )

        fig = px.line(
            monthly,
            x="Month",
            y="Sales",
            markers=True,
            title="Monthly Sales Trend"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        category = (
            filtered_df.groupby("Category")["Sales"]
            .sum()
            .reset_index()
        )

        fig = px.pie(
            category,
            names="Category",
            values="Sales",
            hole=0.4,
            title="Category Contribution"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        region = (
            filtered_df.groupby("Region")["Sales"]
            .sum()
            .reset_index()
        )

        fig = px.bar(
            region,
            x="Region",
            y="Sales",
            color="Region",
            title="Region Wise Sales"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("Top 10 Products")

    top_products = (
        filtered_df.groupby("Product Name")["Sales"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig = px.bar(
        top_products,
        x="Sales",
        y="Product Name",
        orientation="h",
        color="Sales",
        title="Top Selling Products"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("Sales Summary")

    st.dataframe(filtered_df.describe())

# ---------------------------------------------------
# FORECAST PAGE
# ---------------------------------------------------

elif page == "📈 Forecast Explorer":

    st.header("📈 Forecast Explorer")

    st.info(
        "This dashboard displays the forecasting models "
        "developed in the notebook."
    )

    model = st.selectbox(
        "Choose Forecasting Model",
        [
            "SARIMA",
            "Prophet",
            "XGBoost"
        ]
    )

    horizon = st.slider(
        "Forecast Horizon (Months)",
        1,
        3,
        3
    )

    monthly_sales = (
        filtered_df
        .groupby("Order Date")["Sales"]
        .sum()
        .resample("ME")
        .sum()
        .reset_index()
    )

    monthly_sales.columns = ["Date", "Sales"]

    fig = px.line(
        monthly_sales,
        x="Date",
        y="Sales",
        title=f"{model} Forecast Preview"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    c1, c2, c3 = st.columns(3)

    if model == "SARIMA":

        c1.metric("MAE", "16210")
        c2.metric("RMSE", "18866")
        c3.metric("MAPE", "N/A")

    elif model == "Prophet":

        c1.metric("MAE", "10641")
        c2.metric("RMSE", "11136")
        c3.metric("MAPE", "19.11%")

    else:

        c1.metric("MAE", "8695")
        c2.metric("RMSE", "11614")
        c3.metric("MAPE", "12.35%")

    st.success(f"Recommended Model: XGBoost")

    future = pd.DataFrame({
        "Month": [
            "Month 1",
            "Month 2",
            "Month 3"
        ],
        "Forecast Sales": [
            "Prediction 1",
            "Prediction 2",
            "Prediction 3"
        ]
    })

    st.subheader("3-Month Forecast")

    st.table(future)
    # ---------------------------------------------------
# ANOMALY DETECTION
# ---------------------------------------------------

elif page == "🚨 Anomaly Detection":

    st.header("🚨 Sales Anomaly Detection")

    st.write(
        "Detect unusual weekly sales using Isolation Forest."
    )

    weekly_sales = (
        filtered_df
        .groupby("Order Date")["Sales"]
        .sum()
        .resample("W")
        .sum()
        .reset_index()
    )

    weekly_sales.columns = ["Date", "Sales"]

    iso = IsolationForest(
        contamination=0.05,
        random_state=42
    )

    weekly_sales["Anomaly"] = iso.fit_predict(
        weekly_sales[["Sales"]]
    )

    anomalies = weekly_sales[
        weekly_sales["Anomaly"] == -1
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=weekly_sales["Date"],
            y=weekly_sales["Sales"],
            mode="lines",
            name="Weekly Sales"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=anomalies["Date"],
            y=anomalies["Sales"],
            mode="markers",
            marker=dict(
                color="red",
                size=10
            ),
            name="Anomalies"
        )
    )

    fig.update_layout(
        title="Isolation Forest Anomaly Detection"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.subheader("Detected Anomalies")

    st.dataframe(anomalies)

    st.download_button(
        "Download Anomaly Report",
        anomalies.to_csv(index=False),
        file_name="anomalies.csv"
    )

# ---------------------------------------------------
# DEMAND SEGMENTATION
# ---------------------------------------------------

elif page == "📦 Demand Segmentation":

    st.header("📦 Product Demand Segmentation")

    product = filtered_df.groupby(
        "Sub-Category"
    ).agg(
        Total_Sales=("Sales","sum"),
        Average_Order_Value=("Sales","mean"),
        Sales_Volatility=("Sales","std")
    ).reset_index()

    product = product.fillna(0)

    scaler = StandardScaler()

    X = scaler.fit_transform(
        product[
            [
                "Total_Sales",
                "Average_Order_Value",
                "Sales_Volatility"
            ]
        ]
    )

    model = KMeans(
        n_clusters=4,
        random_state=42,
        n_init=10
    )

    product["Cluster"] = model.fit_predict(X)

    pca = PCA(
        n_components=2
    )

    pca_result = pca.fit_transform(X)

    product["PCA1"] = pca_result[:,0]
    product["PCA2"] = pca_result[:,1]

    fig = px.scatter(
        product,
        x="PCA1",
        y="PCA2",
        color=product["Cluster"].astype(str),
        hover_name="Sub-Category",
        title="Demand Segments"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    cluster_map = {
        0:"High Volume",
        1:"Growing",
        2:"Stable",
        3:"Low Demand"
    }

    product["Demand Segment"] = (
        product["Cluster"]
        .map(cluster_map)
    )

    st.subheader("Segment Summary")

    st.dataframe(
        product[
            [
                "Sub-Category",
                "Demand Segment",
                "Total_Sales"
            ]
        ]
    )

    st.download_button(
        "Download Segment Report",
        product.to_csv(index=False),
        file_name="segments.csv"
    )
    # ---------------------------------------------------
# ABOUT PAGE
# ---------------------------------------------------

elif page == "ℹ️ About":

    st.header("ℹ️ About This Project")

    st.markdown("""
    ## End-to-End Sales Forecasting & Demand Intelligence System

    This project was developed as part of a Data Science Internship Final Assessment.

    ### Objectives
    - Forecast future product demand.
    - Detect unusual sales anomalies.
    - Segment products based on demand behavior.
    - Build an interactive dashboard for business users.

    ### Technologies Used
    - Python
    - Streamlit
    - Pandas
    - NumPy
    - Plotly
    - Scikit-learn
    - Statsmodels
    - Prophet
    - XGBoost

    ### Forecasting Models
    - SARIMA
    - Prophet
    - XGBoost

    ### Machine Learning Techniques
    - Isolation Forest
    - K-Means Clustering
    - Principal Component Analysis (PCA)

    ### Dashboard Features
    - Sales Overview
    - Forecast Explorer
    - Anomaly Detection
    - Product Demand Segmentation
    """)

    st.markdown("---")

    st.subheader("Project Statistics")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Dataset Records",
        f"{len(sales_df):,}"
    )

    col2.metric(
        "Regions",
        sales_df["Region"].nunique()
    )

    col3.metric(
        "Categories",
        sales_df["Category"].nunique()
    )

    st.markdown("---")

    st.subheader("Developer")

    st.success("""
    Name : Satya Sindhu Konda

    Project :
    End-to-End Sales Forecasting &
    Demand Intelligence System

    Internship Final Assessment
    """)

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.markdown("---")

st.markdown(
"""
<div style='text-align:center'>

### 📈 Sales Forecasting & Demand Intelligence Dashboard

Developed using

Python • Streamlit • Plotly • Pandas • Prophet • XGBoost • Scikit-learn

© 2026 Satya Sindhu Konda

</div>
""",
unsafe_allow_html=True
)