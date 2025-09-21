# Banking Fraud Detection System

![SecureBank](https://img.icons8.com/color/96/000000/bank-building.png)

A comprehensive PySpark-based banking fraud detection system with interactive Streamlit visualizations and a professional banking theme.

## Overview

This project implements a machine learning-based fraud detection system for banking transactions. It uses PySpark for data processing and model training, and Streamlit for creating an interactive dashboard with a banking theme.

## Features

- **Synthetic Data Generation**: Creates realistic banking transaction data with embedded fraud patterns
- **PySpark ML Pipeline**: Implements a scalable machine learning pipeline for fraud detection
- **Interactive Dashboard**: Provides comprehensive visualizations of transaction patterns and fraud indicators
- **Real-time Fraud Detection**: Simulates real-time transaction monitoring and fraud alerts
- **Banking Theme UI**: Professional user interface designed with a banking aesthetic

## Project Structure

```
BANKING_FRAUD_DETECTION/
├── data/                  # Contains transaction datasets and trained models
├── src/                   # Source code
│   ├── generate_data.py   # Script to generate synthetic transaction data
│   ├── fraud_detection_model.py  # PySpark ML model implementation
│   ├── visualizations.py  # Data visualization functions
│   └── app.py             # Streamlit web application
├── static/                # Static files for the web application
├── templates/             # HTML templates (if needed)
└── requirements.txt       # Project dependencies
```

## Installation

1. Clone the repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Generate Synthetic Data

Run the data generation script to create synthetic banking transaction data:

```bash
python src/generate_data.py
```

This will create CSV files in the `data` directory with synthetic banking transactions, including both legitimate and fraudulent patterns.

### 2. Train the Fraud Detection Model

Train the PySpark machine learning model:

```bash
python src/fraud_detection_model.py
```

This will process the data, train a Random Forest classifier, and save the model to the `data` directory.

### 3. Generate Visualizations

Create static visualizations for the dashboard:

```bash
python src/visualizations.py
```

This will generate interactive HTML visualizations in the `static` directory.

### 4. Run the Streamlit Application

Launch the Streamlit web application:

```bash
streamlit run src/app.py
```

This will start the web server and open the dashboard in your default web browser.

### Fallback Mode

The application includes a fallback mode that works even when PySpark or Java compatibility issues are encountered:

- If PySpark cannot be initialized due to Java compatibility issues, the application will run in visualization-only mode
- The fraud detection feature will use a rule-based approach instead of the machine learning model
- All visualizations and dashboard features will still be available
- A notification will appear in the application indicating that it's running in fallback mode

## Dashboard Features

### Main Dashboard
- Overview of key metrics (total transactions, fraud rate, etc.)
- Distribution of fraudulent vs. legitimate transactions
- Transaction amount distribution by fraud status
- Transaction types analysis
- Model performance metrics

### Transaction Analysis
- Time-based analysis (hour of day, day of week, month)
- Location-based analysis
- Merchant category analysis
- Feature importance visualization

### Fraud Detection Tool
- Interactive form to test transactions against the model
- Fraud probability scoring
- Fraud indicators explanation

### Real-time Monitoring
- Simulated real-time transaction feed
- Fraud alerts dashboard

## Technologies Used

- **PySpark**: For large-scale data processing and machine learning
- **Streamlit**: For interactive web application development
- **Plotly**: For advanced data visualizations
- **Pandas**: For data manipulation and analysis
- **Matplotlib/Seaborn**: For additional visualizations

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project is for educational purposes only
- The data generated is synthetic and does not represent real banking transactions