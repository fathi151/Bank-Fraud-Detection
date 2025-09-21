import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Set style for matplotlib
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Function to load transaction data
def load_transaction_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'banking_transactions.csv')
    return pd.read_csv(data_path)

# Function to load feature importance data
def load_feature_importance():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'feature_importance.csv')
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return None

# Function to load model metrics
def load_model_metrics():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'model_metrics.csv')
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return None

# Create fraud distribution visualization
def plot_fraud_distribution(df):
    fraud_counts = df['is_fraud'].value_counts().reset_index()
    fraud_counts.columns = ['Fraud Status', 'Count']
    fraud_counts['Fraud Status'] = fraud_counts['Fraud Status'].map({0: 'Legitimate', 1: 'Fraudulent'})
    fraud_counts['Percentage'] = fraud_counts['Count'] / fraud_counts['Count'].sum() * 100
    
    fig = px.pie(
        fraud_counts, 
        values='Count', 
        names='Fraud Status',
        title='Distribution of Fraudulent vs. Legitimate Transactions',
        color='Fraud Status',
        color_discrete_map={'Legitimate': '#3498db', 'Fraudulent': '#e74c3c'},
        hole=0.4
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        title_font_size=24,
        legend_title_font_size=18,
        legend_font_size=16,
        template='plotly_white'
    )
    
    return fig

# Create transaction amount distribution by fraud status
def plot_amount_distribution(df):
    fig = px.histogram(
        df, 
        x='amount', 
        color='is_fraud',
        color_discrete_map={0: '#3498db', 1: '#e74c3c'},
        marginal='box',
        opacity=0.7,
        barmode='overlay',
        labels={'amount': 'Transaction Amount', 'is_fraud': 'Fraud Status'},
        title='Distribution of Transaction Amounts by Fraud Status',
        category_orders={'is_fraud': [0, 1]},
        hover_data=['transaction_type', 'merchant_category']
    )
    
    fig.update_layout(
        title_font_size=24,
        xaxis_title_font_size=18,
        yaxis_title_font_size=18,
        legend_title_text='Fraud Status',
        legend_title_font_size=18,
        legend_font_size=16,
        template='plotly_white',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
            itemsizing='constant'
        )
    )
    
    # Update legend labels
    newnames = {'0': 'Legitimate', '1': 'Fraudulent'}
    fig.for_each_trace(lambda t: t.update(name=newnames[t.name]))
    
    return fig

# Create transaction type distribution by fraud status
def plot_transaction_type_distribution(df):
    # Group by transaction type and fraud status
    type_fraud = df.groupby(['transaction_type', 'is_fraud']).size().reset_index(name='count')
    type_fraud['is_fraud'] = type_fraud['is_fraud'].map({0: 'Legitimate', 1: 'Fraudulent'})
    
    fig = px.bar(
        type_fraud,
        x='transaction_type',
        y='count',
        color='is_fraud',
        barmode='group',
        color_discrete_map={'Legitimate': '#3498db', 'Fraudulent': '#e74c3c'},
        labels={'transaction_type': 'Transaction Type', 'count': 'Number of Transactions', 'is_fraud': 'Fraud Status'},
        title='Transaction Types by Fraud Status'
    )
    
    fig.update_layout(
        title_font_size=24,
        xaxis_title_font_size=18,
        yaxis_title_font_size=18,
        legend_title_font_size=18,
        legend_font_size=16,
        template='plotly_white'
    )
    
    return fig

# Create merchant category distribution by fraud status
def plot_merchant_category_distribution(df):
    # Group by merchant category and fraud status
    merchant_fraud = df.groupby(['merchant_category', 'is_fraud']).size().reset_index(name='count')
    merchant_fraud['is_fraud'] = merchant_fraud['is_fraud'].map({0: 'Legitimate', 1: 'Fraudulent'})
    
    fig = px.bar(
        merchant_fraud,
        x='merchant_category',
        y='count',
        color='is_fraud',
        barmode='group',
        color_discrete_map={'Legitimate': '#3498db', 'Fraudulent': '#e74c3c'},
        labels={'merchant_category': 'Merchant Category', 'count': 'Number of Transactions', 'is_fraud': 'Fraud Status'},
        title='Merchant Categories by Fraud Status'
    )
    
    fig.update_layout(
        title_font_size=24,
        xaxis_title_font_size=18,
        yaxis_title_font_size=18,
        legend_title_font_size=18,
        legend_font_size=16,
        template='plotly_white'
    )
    
    return fig

# Create location distribution by fraud status
def plot_location_distribution(df):
    # Group by location and fraud status
    location_fraud = df.groupby(['location', 'is_fraud']).size().reset_index(name='count')
    location_fraud['is_fraud'] = location_fraud['is_fraud'].map({0: 'Legitimate', 1: 'Fraudulent'})
    
    fig = px.bar(
        location_fraud,
        x='location',
        y='count',
        color='is_fraud',
        barmode='group',
        color_discrete_map={'Legitimate': '#3498db', 'Fraudulent': '#e74c3c'},
        labels={'location': 'Location', 'count': 'Number of Transactions', 'is_fraud': 'Fraud Status'},
        title='Transaction Locations by Fraud Status'
    )
    
    fig.update_layout(
        title_font_size=24,
        xaxis_title_font_size=18,
        yaxis_title_font_size=18,
        legend_title_font_size=18,
        legend_font_size=16,
        template='plotly_white'
    )
    
    return fig

# Create time-based analysis
def plot_time_analysis(df):
    # Convert date and time to datetime
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    
    # Hour of day analysis
    hour_fraud = df.groupby(['hour', 'is_fraud']).size().reset_index(name='count')
    hour_fraud['is_fraud'] = hour_fraud['is_fraud'].map({0: 'Legitimate', 1: 'Fraudulent'})
    
    fig1 = px.line(
        hour_fraud,
        x='hour',
        y='count',
        color='is_fraud',
        color_discrete_map={'Legitimate': '#3498db', 'Fraudulent': '#e74c3c'},
        labels={'hour': 'Hour of Day', 'count': 'Number of Transactions', 'is_fraud': 'Fraud Status'},
        title='Transaction Volume by Hour of Day'
    )
    
    # Day of week analysis
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    df['day_name'] = df['day_of_week'].apply(lambda x: day_names[x])
    day_fraud = df.groupby(['day_name', 'is_fraud']).size().reset_index(name='count')
    day_fraud['is_fraud'] = day_fraud['is_fraud'].map({0: 'Legitimate', 1: 'Fraudulent'})
    
    fig2 = px.bar(
        day_fraud,
        x='day_name',
        y='count',
        color='is_fraud',
        barmode='group',
        color_discrete_map={'Legitimate': '#3498db', 'Fraudulent': '#e74c3c'},
        labels={'day_name': 'Day of Week', 'count': 'Number of Transactions', 'is_fraud': 'Fraud Status'},
        title='Transaction Volume by Day of Week',
        category_orders={'day_name': day_names}
    )
    
    # Month analysis
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    df['month_name'] = df['month'].apply(lambda x: month_names[x-1])
    month_fraud = df.groupby(['month_name', 'is_fraud']).size().reset_index(name='count')
    month_fraud['is_fraud'] = month_fraud['is_fraud'].map({0: 'Legitimate', 1: 'Fraudulent'})
    
    fig3 = px.bar(
        month_fraud,
        x='month_name',
        y='count',
        color='is_fraud',
        barmode='group',
        color_discrete_map={'Legitimate': '#3498db', 'Fraudulent': '#e74c3c'},
        labels={'month_name': 'Month', 'count': 'Number of Transactions', 'is_fraud': 'Fraud Status'},
        title='Transaction Volume by Month',
        category_orders={'month_name': month_names}
    )
    
    return fig1, fig2, fig3

# Create feature importance visualization
def plot_feature_importance(feature_importance_df):
    if feature_importance_df is None:
        return None
    
    # Sort by importance
    df = feature_importance_df.sort_values('importance', ascending=True).tail(20)
    
    fig = px.bar(
        df,
        y='feature',
        x='importance',
        orientation='h',
        title='Top 20 Feature Importance for Fraud Detection',
        color='importance',
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        title_font_size=24,
        xaxis_title='Importance',
        xaxis_title_font_size=18,
        yaxis_title='Feature',
        yaxis_title_font_size=18,
        template='plotly_white'
    )
    
    return fig

# Create model metrics visualization
def plot_model_metrics(metrics_df):
    if metrics_df is None:
        return None
    
    # Create a radar chart for model metrics
    metrics = metrics_df.iloc[0].to_dict()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=[metrics['auc'], metrics['accuracy'], metrics['f1']],
        theta=['AUC', 'Accuracy', 'F1 Score'],
        fill='toself',
        name='Model Performance',
        line_color='#3498db'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False,
        title='Model Performance Metrics',
        title_font_size=24,
        template='plotly_white'
    )
    
    return fig

# Create fraud amount by transaction type
def plot_fraud_amount_by_type(df):
    # Filter for fraudulent transactions
    fraud_df = df[df['is_fraud'] == 1]
    
    # Group by transaction type and calculate statistics
    type_stats = fraud_df.groupby('transaction_type')['amount'].agg(['mean', 'median', 'count']).reset_index()
    type_stats.columns = ['Transaction Type', 'Mean Amount', 'Median Amount', 'Count']
    
    # Create bubble chart
    fig = px.scatter(
        type_stats,
        x='Mean Amount',
        y='Median Amount',
        size='Count',
        color='Transaction Type',
        hover_name='Transaction Type',
        text='Transaction Type',
        title='Fraudulent Transaction Amounts by Type',
        size_max=60
    )
    
    fig.update_traces(textposition='top center')
    fig.update_layout(
        title_font_size=24,
        xaxis_title_font_size=18,
        yaxis_title_font_size=18,
        template='plotly_white'
    )
    
    return fig

# Create fraud detection dashboard
def create_dashboard():
    # Load data
    df = load_transaction_data()
    feature_importance_df = load_feature_importance()
    metrics_df = load_model_metrics()
    
    # Create visualizations
    fraud_dist_fig = plot_fraud_distribution(df)
    amount_dist_fig = plot_amount_distribution(df)
    transaction_type_fig = plot_transaction_type_distribution(df)
    merchant_category_fig = plot_merchant_category_distribution(df)
    location_fig = plot_location_distribution(df)
    hour_fig, day_fig, month_fig = plot_time_analysis(df)
    fraud_amount_type_fig = plot_fraud_amount_by_type(df)
    
    # Create feature importance and model metrics visualizations if available
    feature_importance_fig = None
    model_metrics_fig = None
    
    if feature_importance_df is not None:
        feature_importance_fig = plot_feature_importance(feature_importance_df)
    
    if metrics_df is not None:
        model_metrics_fig = plot_model_metrics(metrics_df)
    
    return {
        'fraud_distribution': fraud_dist_fig,
        'amount_distribution': amount_dist_fig,
        'transaction_type': transaction_type_fig,
        'merchant_category': merchant_category_fig,
        'location': location_fig,
        'hour_analysis': hour_fig,
        'day_analysis': day_fig,
        'month_analysis': month_fig,
        'fraud_amount_by_type': fraud_amount_type_fig,
        'feature_importance': feature_importance_fig,
        'model_metrics': model_metrics_fig
    }

# Main function
def main():
    # Create visualizations
    visualizations = create_dashboard()
    
    # Save visualizations to HTML files
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    static_dir = os.path.join(base_dir, 'static')
    os.makedirs(static_dir, exist_ok=True)
    
    for name, fig in visualizations.items():
        if fig is not None:
            fig.write_html(os.path.join(static_dir, f'{name}.html'))
    
    print("Visualizations created and saved to static directory.")

if __name__ == "__main__":
    main()