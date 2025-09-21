import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
import base64
from datetime import datetime

# Import visualization module
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.visualizations import create_dashboard, load_transaction_data

# Set page configuration
st.set_page_config(
    page_title="Banking Fraud Detection Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for banking theme
def load_css():
    css = """
    <style>
        .main {
            background-color: #f5f7fa;
        }
        .stApp {
            font-family: 'Arial', sans-serif;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #e6f0ff;
            border-radius: 4px 4px 0px 0px;
            padding: 10px 20px;
            font-weight: 600;
        }
        .stTabs [aria-selected="true"] {
            background-color: #1e5bb0 !important;
            color: white !important;
        }
        h1, h2, h3 {
            color: #1e5bb0;
        }
        .fraud-alert {
            background-color: #ffebee;
            border-left: 5px solid #e53935;
            padding: 15px;
            border-radius: 4px;
            margin-bottom: 20px;
        }
        .safe-alert {
            background-color: #e8f5e9;
            border-left: 5px solid #43a047;
            padding: 15px;
            border-radius: 4px;
            margin-bottom: 20px;
        }
        .metric-card {
            background-color: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            text-align: center;
            margin-bottom: 20px;
        }
        .metric-value {
            font-size: 36px;
            font-weight: bold;
            margin: 10px 0;
        }
        .metric-label {
            font-size: 14px;
            color: #666;
            text-transform: uppercase;
        }
        .bank-logo {
            max-width: 200px;
            margin-bottom: 20px;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Function to create a bank logo
def get_bank_logo():
    # Create a simple SVG logo
    svg_logo = '''
    <svg width="200" height="60" xmlns="http://www.w3.org/2000/svg">
        <rect width="200" height="60" fill="#1e5bb0" rx="5" ry="5"/>
        <text x="20" y="40" font-family="Arial" font-size="24" fill="white" font-weight="bold">SecureBank</text>
        <path d="M170 15 L180 30 L160 30 Z" fill="#ffd700"/>
        <circle cx="170" cy="35" r="10" fill="#ffd700"/>
    </svg>
    '''
    b64 = base64.b64encode(svg_logo.encode('utf-8')).decode('utf-8')
    return f'<img src="data:image/svg+xml;base64,{b64}" class="bank-logo">'    

# Function to load the trained model
def load_model():
    try:
        # Initialize Spark session
        spark = SparkSession.builder \
            .appName("BankingFraudDetection") \
            .config("spark.driver.memory", "2g") \
            .config("spark.executor.memory", "2g") \
            .getOrCreate()
        
        # Load the model
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, 'data', 'fraud_detection_model')
        
        if os.path.exists(model_path):
            model = PipelineModel.load(model_path)
            return spark, model
        else:
            return spark, None
    except Exception as e:
        st.warning(f"⚠️ Running in visualization-only mode. PySpark model could not be loaded due to Java compatibility issues.")
        st.info("The dashboard will show all visualizations, but real-time fraud detection is disabled.")
        return None, None

# Function to make predictions on new data
def predict_fraud(spark, model, data):
    if spark is None or model is None:
        return None, None
    
    try:
        # Convert input data to Spark DataFrame
        input_df = spark.createDataFrame([data])
        
        # Make prediction
        prediction = model.transform(input_df)
        
        # Extract prediction and probability
        result = prediction.select("prediction", "probability").collect()[0]
        is_fraud = int(result["prediction"])
        probability = float(result["probability"][1])  # Probability of fraud
        
        return is_fraud, probability
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None

# Function to make rule-based predictions when PySpark is not available
def rule_based_fraud_detection(data):
    """A simple rule-based approach to detect potential fraud when the ML model is unavailable"""
    # Define some simple rules based on known fraud patterns
    is_fraud = False
    confidence = 0.5  # Default confidence
    reasons = []
    
    # Rule 1: Large transaction amounts
    if data['amount'] > 1000:
        is_fraud = True
        confidence += 0.1
        reasons.append("High transaction amount")
    
    # Rule 2: Unusual transaction time
    if data['hour'] < 6 or data['hour'] > 22:
        is_fraud = True
        confidence += 0.1
        reasons.append("Unusual transaction time")
    
    # Rule 3: High-risk transaction types
    if data['transaction_type'] in ["CASH_OUT", "TRANSFER"]:
        is_fraud = True
        confidence += 0.1
        reasons.append("High-risk transaction type")
    
    # Rule 4: Unusual merchant categories for large amounts
    if data['amount'] > 500 and data['merchant_category'] in ["electronics", "jewelry"]:
        is_fraud = True
        confidence += 0.1
        reasons.append("Unusual merchant category for large amount")
    
    # Cap confidence at 0.9 since this is just a rule-based system
    confidence = min(confidence, 0.9)
    
    return (1 if is_fraud else 0), confidence, reasons

# Main application
def main():
    # Load CSS
    load_css()
    
    # Display bank logo
    st.markdown(get_bank_logo(), unsafe_allow_html=True)
    
    # App title
    st.title("Banking Fraud Detection System")
    st.markdown("### Advanced Analytics for Secure Banking Operations")
    
    # Sidebar
    st.sidebar.image("https://img.icons8.com/color/96/000000/bank-building.png")
    st.sidebar.title("Navigation")
    
    # Navigation
    page = st.sidebar.radio(
        "Select a page:",
        ["Dashboard", "Transaction Analysis", "Fraud Detection", "Real-time Monitoring", "About"]
    )
    
    # Load data and visualizations
    df = load_transaction_data()
    visualizations = create_dashboard()
    
    # Load model
    spark, model = load_model()
    
    # Dashboard page
    if page == "Dashboard":
        st.header("Fraud Detection Dashboard")
        st.markdown("#### Key metrics and insights from our fraud detection system")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_transactions = len(df)
        fraud_transactions = df['is_fraud'].sum()
        fraud_percentage = (fraud_transactions / total_transactions) * 100
        avg_transaction = df['amount'].mean()
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{total_transactions:,}</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Total Transactions</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{fraud_transactions:,}</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Fraudulent Transactions</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{fraud_percentage:.2f}%</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Fraud Rate</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">${avg_transaction:.2f}</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Avg Transaction</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Main visualizations
        st.subheader("Fraud Distribution")
        st.plotly_chart(visualizations['fraud_distribution'], use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Transaction Amount Distribution")
            st.plotly_chart(visualizations['amount_distribution'], use_container_width=True)
            
        with col2:
            st.subheader("Transaction Types")
            st.plotly_chart(visualizations['transaction_type'], use_container_width=True)
        
        # Model performance if available
        if visualizations['model_metrics'] is not None:
            st.subheader("Model Performance")
            st.plotly_chart(visualizations['model_metrics'], use_container_width=True)
    
    # Transaction Analysis page
    elif page == "Transaction Analysis":
        st.header("Transaction Analysis")
        st.markdown("#### Detailed analysis of transaction patterns and fraud indicators")
        
        # Create tabs for different analyses
        tab1, tab2, tab3 = st.tabs(["Time Analysis", "Location Analysis", "Category Analysis"])
        
        with tab1:
            st.subheader("Transaction Patterns by Time")
            st.plotly_chart(visualizations['hour_analysis'], use_container_width=True)
            st.plotly_chart(visualizations['day_analysis'], use_container_width=True)
            st.plotly_chart(visualizations['month_analysis'], use_container_width=True)
        
        with tab2:
            st.subheader("Transaction Patterns by Location")
            st.plotly_chart(visualizations['location'], use_container_width=True)
            
            # Map visualization (placeholder - would need actual coordinates)
            st.markdown("#### Geographic Distribution of Fraudulent Transactions")
            st.info("This feature requires actual geographic coordinates which are not available in the current dataset.")
        
        with tab3:
            st.subheader("Transaction Patterns by Category")
            st.plotly_chart(visualizations['merchant_category'], use_container_width=True)
            st.plotly_chart(visualizations['fraud_amount_by_type'], use_container_width=True)
            
            if visualizations['feature_importance'] is not None:
                st.subheader("Feature Importance")
                st.plotly_chart(visualizations['feature_importance'], use_container_width=True)
    
    # Fraud Detection page
    elif page == "Fraud Detection":
        st.header("Fraud Detection Tool")
        
        # Check if model is available
        model_available = (spark is not None and model is not None)
        
        if not model_available:
            st.markdown("#### Rule-based Fraud Detection (ML Model Unavailable)")
            st.info("The PySpark ML model could not be loaded due to Java compatibility issues. Using rule-based detection instead.")
        else:
            st.markdown("#### Test our fraud detection model with transaction data")
        
        # Input form for transaction details
        with st.form("transaction_form"):
            st.subheader("Enter Transaction Details")
            
            col1, col2 = st.columns(2)
            
            with col1:
                amount = st.number_input("Transaction Amount ($)", min_value=1.0, max_value=10000.0, value=100.0)
                transaction_type = st.selectbox(
                    "Transaction Type",
                    options=["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
                )
                merchant_category = st.selectbox(
                    "Merchant Category",
                    options=["grocery_pos", "entertainment", "gas_transport", "misc_pos", "shopping", "food_dining", "health_fitness", "travel", "home", "electronics"]
                )
            
            with col2:
                location = st.selectbox(
                    "Location",
                    options=["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose"]
                )
                time_of_day = st.slider("Hour of Day", 0, 23, 12)
                day_of_week = st.selectbox(
                    "Day of Week",
                    options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                )
            
            submit_button = st.form_submit_button("Check Transaction")
        
        # Process form submission
        if submit_button:
            # Map day of week to numeric
            day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
            day_numeric = day_map[day_of_week]
            
            # Create transaction data
            transaction_data = {
                "amount": amount,
                "transaction_type": transaction_type,
                "merchant_category": merchant_category,
                "location": location,
                "hour": time_of_day,
                "day_of_week": day_numeric,
                "date": datetime.now().strftime("%Y-%m-%d"),
                "time": f"{time_of_day:02d}:00:00"
            }
            
            st.subheader("Fraud Detection Result")
            
            # Use ML model if available, otherwise use rule-based approach
            if model_available:
                is_fraud, probability = predict_fraud(spark, model, transaction_data)
                reasons = []
                
                if amount > 1000:
                    reasons.append("High transaction amount")
                if time_of_day < 6 or time_of_day > 22:
                    reasons.append("Unusual transaction time")
                if transaction_type in ["CASH_OUT", "TRANSFER"]:
                    reasons.append("High-risk transaction type")
            else:
                is_fraud, probability, reasons = rule_based_fraud_detection(transaction_data)
            
            if is_fraud is not None:
                if is_fraud == 1:
                    st.markdown('<div class="fraud-alert">', unsafe_allow_html=True)
                    st.markdown(f"### ⚠️ Potential Fraud Detected")
                    st.markdown(f"This transaction has been flagged as potentially fraudulent with {probability*100:.2f}% confidence.")
                    if not model_available:
                        st.markdown("*Using rule-based detection (ML model unavailable)*")
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Show fraud indicators
                    st.subheader("Fraud Indicators")
                    for indicator in reasons:
                        st.markdown(f"- {indicator}")
                else:
                    st.markdown('<div class="safe-alert">', unsafe_allow_html=True)
                    st.markdown(f"### ✅ Transaction Appears Legitimate")
                    st.markdown(f"Our {'model' if model_available else 'system'} indicates this is likely a legitimate transaction with {(1-probability)*100:.2f}% confidence.")
                    if not model_available:
                        st.markdown("*Using rule-based detection (ML model unavailable)*")
                    st.markdown("</div>", unsafe_allow_html=True)
    
    # Real-time Monitoring page
    elif page == "Real-time Monitoring":
        st.header("Real-time Transaction Monitoring")
        st.markdown("#### Monitor transactions as they occur in the system")
        
        # This would typically connect to a real-time data source
        # For demonstration, we'll use the existing data with a filter for recent transactions
        
        st.info("This is a simulation of real-time monitoring using the synthetic dataset.")
        
        # Add a date filter
        unique_dates = sorted(df['date'].unique())
        selected_date = st.selectbox("Select Date to Monitor", unique_dates)
        
        # Filter data for selected date
        filtered_df = df[df['date'] == selected_date].copy()
        
        # Add a simulated "time ago" column
        filtered_df['time_ago'] = np.random.randint(1, 60, size=len(filtered_df))
        filtered_df = filtered_df.sort_values('time_ago')
        
        # Display recent transactions
        st.subheader("Recent Transactions")
        
        # Display transactions in a more visual way
        for i, row in filtered_df.head(10).iterrows():
            col1, col2, col3 = st.columns([1, 3, 1])
            
            with col1:
                if row['is_fraud'] == 1:
                    st.markdown("🚨")
                else:
                    st.markdown("✅")
            
            with col2:
                transaction_type = row['transaction_type']
                merchant = row['merchant_category']
                amount = row['amount']
                location = row['location']
                
                st.markdown(f"**{transaction_type}** - ${amount:.2f}")
                st.markdown(f"{merchant.title()} | {location}")
            
            with col3:
                st.markdown(f"{row['time_ago']} min ago")
            
            st.markdown("---")
        
        # Alert section
        st.subheader("Fraud Alerts")
        
        fraud_today = filtered_df[filtered_df['is_fraud'] == 1]
        
        if len(fraud_today) > 0:
            st.warning(f"{len(fraud_today)} potential fraud cases detected today")
            
            for i, row in fraud_today.head(5).iterrows():
                st.markdown(f"**Alert #{i}**: ${row['amount']:.2f} {row['transaction_type']} transaction in {row['location']}")
        else:
            st.success("No fraud alerts for the selected date")
    
    # About page
    elif page == "About":
        st.header("About Banking Fraud Detection System")
        
        st.markdown("""
        ### Project Overview
        
        This Banking Fraud Detection System uses advanced machine learning techniques to identify potentially fraudulent transactions in banking data. The system analyzes transaction patterns, amounts, locations, and other features to flag suspicious activities.
        
        ### Key Features
        
        - **Real-time Fraud Detection**: Analyze transactions as they occur to identify potential fraud
        - **Advanced Visualization**: Interactive dashboards to explore transaction patterns
        - **Machine Learning Model**: Powered by PySpark ML for scalable fraud detection
        - **Customizable Alerts**: Configure the system to alert on specific fraud patterns
        
        ### Technology Stack
        
        - **PySpark**: For large-scale data processing and machine learning
        - **Streamlit**: For interactive web application development
        - **Plotly**: For advanced data visualizations
        - **Pandas**: For data manipulation and analysis
        
        ### How It Works
        
        1. Transaction data is processed through our PySpark pipeline
        2. The machine learning model evaluates each transaction for fraud indicators
        3. Transactions are scored based on fraud probability
        4. High-risk transactions are flagged for review
        5. Visualizations help analysts identify patterns and trends
        
        ### Contact
        
        For more information, please contact the development team at securebank@example.com
        """)

if __name__ == "__main__":
    main()