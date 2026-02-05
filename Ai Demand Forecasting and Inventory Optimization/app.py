"""
AI Demand Forecasting and Inventory Optimization - WEB DASHBOARD
=================================================================
Interactive web interface using Streamlit
Run with: streamlit run app.py
Access at: http://localhost:8501

FULLY FIXED VERSION - Updated for latest Streamlit & no jinja2 required
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy import stats
import json

# Page configuration
st.set_page_config(
    page_title="AI Demand Forecasting & Inventory Optimization",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">📊 AI Demand Forecasting & Inventory Optimization Dashboard</div>', unsafe_allow_html=True)

# Helper functions
@st.cache_data
def generate_synthetic_data(n_days=730, n_products=5):
    """Generate realistic synthetic retail sales data"""
    np.random.seed(42)
    date_range = pd.date_range(start='2022-01-01', periods=n_days, freq='D')
    
    data_list = []
    
    # Updated product information with realistic retail categories
    product_info = {
        'Fashion': {
            'base_demand': 70, 
            'seasonality': 45, 
            'trend': 0.04, 
            'price': 999.0,
            'description': 'Clothing and apparel items',
            'emoji': '👗'
        },
        'Air Cooler': {
            'base_demand': 40, 
            'seasonality': 80,  # High seasonality (summer demand)
            'trend': 0.02, 
            'price': 6500.0,
            'description': 'Seasonal cooling appliances',
            'emoji': '❄️'
        },
        'Electronic Devices': {
            'base_demand': 55, 
            'seasonality': 25, 
            'trend': 0.06,  # Growing trend
            'price': 18000.0,
            'description': 'Smartphones, tablets, gadgets',
            'emoji': '📱'
        },
        'Beauty Products': {
            'base_demand': 90, 
            'seasonality': 30, 
            'trend': 0.05, 
            'price': 499.0,
            'description': 'Cosmetics and personal care',
            'emoji': '💄'
        },
        'Grocery': {
            'base_demand': 160, 
            'seasonality': 15,  # Low seasonality (consistent demand)
            'trend': 0.03, 
            'price': 120.0,
            'description': 'Daily essentials and food items',
            'emoji': '🛒'
        }
    }
    
    for product_name, info in product_info.items():
        for i, date in enumerate(date_range):
            # Base demand with trend
            base = info['base_demand'] + (i * info['trend'])
            
            # Seasonality (weekly and yearly)
            weekly_seasonality = info['seasonality'] * np.sin(2 * np.pi * i / 7)
            yearly_seasonality = info['seasonality'] * 0.5 * np.sin(2 * np.pi * i / 365)
            
            # Special patterns for specific products
            if product_name == 'Air Cooler':
                # Peak in summer months (May-August)
                summer_boost = 60 if date.month in [5, 6, 7, 8] else -30
                yearly_seasonality += summer_boost
            
            if product_name == 'Fashion':
                # Higher demand during festive seasons
                festive_boost = 40 if date.month in [10, 11, 12] else 0
                yearly_seasonality += festive_boost
            
            if product_name == 'Beauty Products':
                # Higher demand on Valentine's Day, Mother's Day, etc.
                special_days = (date.month == 2 and date.day in [13, 14, 15]) or \
                              (date.month == 5 and date.day in [8, 9, 10])
                if special_days:
                    yearly_seasonality += 50
            
            # Weekend boost (higher for Fashion and Beauty Products)
            if product_name in ['Fashion', 'Beauty Products']:
                weekend_boost = 30 if date.dayofweek >= 5 else 0
            else:
                weekend_boost = 15 if date.dayofweek >= 5 else 0
            
            # Holiday boost (Diwali, Christmas, New Year)
            holiday_boost = 50 if date.month == 12 or (date.month == 11 and date.day > 20) else 0
            
            # Random noise
            noise = np.random.normal(0, 10)
            
            # Calculate demand
            demand = max(0, base + weekly_seasonality + yearly_seasonality + 
                       weekend_boost + holiday_boost + noise)
            
            # Promotional events (random 10% of days, more impactful for certain products)
            is_promo = np.random.random() < 0.1
            if is_promo:
                if product_name in ['Electronic Devices', 'Fashion']:
                    demand *= 1.4  # Higher impact for expensive items
                else:
                    demand *= 1.25
            
            data_list.append({
                'date': date,
                'product': product_name,
                'demand': int(demand),
                'price': info['price'],
                'is_weekend': 1 if date.dayofweek >= 5 else 0,
                'is_promotion': 1 if is_promo else 0,
                'day_of_week': date.dayofweek,
                'month': date.month,
                'day_of_month': date.day,
                'quarter': date.quarter
            })
    
    return pd.DataFrame(data_list)

def create_features(df, product_name):
    """Create time series features"""
    product_df = df[df['product'] == product_name].copy()
    product_df = product_df.sort_values('date').reset_index(drop=True)
    
    for lag in [1, 7, 14, 30]:
        product_df[f'lag_{lag}'] = product_df['demand'].shift(lag)
    
    for window in [7, 14, 30]:
        product_df[f'rolling_mean_{window}'] = product_df['demand'].rolling(window=window).mean()
        product_df[f'rolling_std_{window}'] = product_df['demand'].rolling(window=window).std()
    
    product_df['year'] = product_df['date'].dt.year
    product_df['day_of_year'] = product_df['date'].dt.dayofyear
    
    product_df = product_df.dropna()
    return product_df

@st.cache_resource
def train_models(df, product_name, test_size=0.2):
    """Train forecasting models"""
    product_df = create_features(df, product_name)
    
    feature_cols = [col for col in product_df.columns 
                   if col not in ['date', 'product', 'demand', 'price']]
    
    X = product_df[feature_cols]
    y = product_df['demand']
    
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    dates_test = product_df['date'].iloc[split_idx:].values
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
        'Linear Regression': LinearRegression()
    }
    
    results = {}
    
    for model_name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        results[model_name] = {
            'predictions': y_pred,
            'actual': y_test.values,
            'dates': dates_test,
            'metrics': {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}
        }
    
    return results

def calculate_inventory_metrics(predictions, actual_demand, lead_time=7, service_level=0.95):
    """Calculate optimal inventory levels"""
    errors = actual_demand - predictions
    error_std = np.std(errors)
    avg_demand_lead_time = np.mean(predictions) * lead_time
    z_score = stats.norm.ppf(service_level)
    safety_stock = z_score * error_std * np.sqrt(lead_time)
    reorder_point = avg_demand_lead_time + safety_stock
    annual_demand = np.sum(predictions) * (365 / len(predictions))
    eoq = np.sqrt((2 * annual_demand * 50) / 2)
    
    return {
        'avg_daily_demand': np.mean(predictions),
        'safety_stock': safety_stock,
        'reorder_point': reorder_point,
        'eoq': eoq,
        'service_level': service_level,
        'error_std': error_std
    }

# Sidebar
st.sidebar.header("⚙️ Configuration")

# Load or generate data
if 'df' not in st.session_state:
    with st.spinner('Generating synthetic retail data...'):
        st.session_state.df = generate_synthetic_data(n_days=730, n_products=5)

df = st.session_state.df

# Product info for display
product_info_display = {
    'Fashion': {'emoji': '👗', 'price': '₹999'},
    'Air Cooler': {'emoji': '❄️', 'price': '₹6,500'},
    'Electronic Devices': {'emoji': '📱', 'price': '₹18,000'},
    'Beauty Products': {'emoji': '💄', 'price': '₹499'},
    'Grocery': {'emoji': '🛒', 'price': '₹120'}
}

# Sidebar inputs
st.sidebar.subheader("📦 Product Selection")
product_options = df['product'].unique()
product_labels = [f"{product_info_display[p]['emoji']} {p} ({product_info_display[p]['price']})" 
                  for p in product_options]
product_selection = st.sidebar.selectbox("Select Product", range(len(product_options)), 
                                         format_func=lambda x: product_labels[x])
product = product_options[product_selection]

st.sidebar.markdown(f"**Selected:** {product}")

st.sidebar.subheader("📊 Model Settings")
test_size = st.sidebar.slider("Test Size (%)", 10, 40, 20) / 100

st.sidebar.subheader("📦 Inventory Settings")
lead_time = st.sidebar.slider("Lead Time (days)", 1, 30, 7)
service_level = st.sidebar.slider("Service Level (%)", 85, 99, 95) / 100

st.sidebar.markdown("---")
st.sidebar.info(f"""
**Service Level: {service_level*100:.0f}%**

This means:
- ✅ {service_level*100:.0f}% orders fulfilled
- ⚠️ {(1-service_level)*100:.1f}% stockout risk
""")

# Run analysis button
if st.sidebar.button("🚀 Run Analysis", type="primary"):
    st.session_state.run_analysis = True

# Main content
if 'run_analysis' in st.session_state and st.session_state.run_analysis:
    
    # Train models
    with st.spinner('Training machine learning models...'):
        results = train_models(df, product, test_size)
    
    best_model = min(results.items(), key=lambda x: x[1]['metrics']['MAPE'])
    
    # Calculate inventory metrics
    inventory_metrics = calculate_inventory_metrics(
        best_model[1]['predictions'],
        best_model[1]['actual'],
        lead_time,
        service_level
    )
    
    # Display metrics
    st.header(f"📈 Analysis Results for {product_info_display[product]['emoji']} {product}")
    
    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "🏆 Best Model",
            best_model[0],
            f"{100-best_model[1]['metrics']['MAPE']:.1f}% Accurate"
        )
    
    with col2:
        st.metric(
            "📊 MAPE",
            f"{best_model[1]['metrics']['MAPE']:.2f}%",
            "Lower is better"
        )
    
    with col3:
        st.metric(
            "📦 Safety Stock",
            f"{inventory_metrics['safety_stock']:.0f} units"
        )
    
    with col4:
        st.metric(
            "🔄 Reorder Point",
            f"{inventory_metrics['reorder_point']:.0f} units"
        )
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Forecast", "📊 Model Comparison", "📦 Inventory", "📋 Data"])
    
    with tab1:
        st.subheader(f"Demand Forecast - {best_model[0]}")
        
        # Create interactive plot with Plotly
        fig = go.Figure()
        
        dates = best_model[1]['dates']
        actual = best_model[1]['actual']
        predicted = best_model[1]['predictions']
        
        fig.add_trace(go.Scatter(
            x=dates, y=actual,
            mode='lines+markers',
            name='Actual Demand',
            line=dict(color='#636EFA', width=2),
            marker=dict(size=5)
        ))
        
        fig.add_trace(go.Scatter(
            x=dates, y=predicted,
            mode='lines+markers',
            name='Forecasted Demand',
            line=dict(color='#EF553B', width=2, dash='dash'),
            marker=dict(size=5, symbol='square')
        ))
        
        # Add confidence interval
        upper_bound = predicted + inventory_metrics['error_std']
        lower_bound = predicted - inventory_metrics['error_std']
        
        fig.add_trace(go.Scatter(
            x=np.concatenate([dates, dates[::-1]]),
            y=np.concatenate([upper_bound, lower_bound[::-1]]),
            fill='toself',
            fillcolor='rgba(239,85,59,0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Interval',
            showlegend=True
        ))
        
        fig.update_layout(
            title=f"Actual vs Predicted Demand for {product}",
            xaxis_title="Date",
            yaxis_title="Demand (units)",
            hovermode='x unified',
            height=500,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show accuracy
        accuracy = 100 - best_model[1]['metrics']['MAPE']
        st.success(f"✅ Forecast Accuracy: **{accuracy:.2f}%** - The model predictions are highly reliable!")
    
    with tab2:
        st.subheader("Model Performance Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # MAPE comparison
            model_names = list(results.keys())
            mapes = [results[m]['metrics']['MAPE'] for m in model_names]
            
            fig_mape = go.Figure(data=[
                go.Bar(x=model_names, y=mapes, 
                      marker_color=['#636EFA', '#EF553B', '#00CC96'],
                      text=[f"{m:.2f}%" for m in mapes],
                      textposition='outside')
            ])
            fig_mape.update_layout(
                title="MAPE Comparison (Lower is Better)",
                yaxis_title="MAPE (%)",
                height=400,
                template='plotly_white'
            )
            st.plotly_chart(fig_mape, use_container_width=True)
        
        with col2:
            # R² comparison
            r2_scores = [results[m]['metrics']['R2'] for m in model_names]
            
            fig_r2 = go.Figure(data=[
                go.Bar(x=model_names, y=r2_scores,
                      marker_color=['#636EFA', '#EF553B', '#00CC96'],
                      text=[f"{r:.3f}" for r in r2_scores],
                      textposition='outside')
            ])
            fig_r2.update_layout(
                title="R² Score Comparison (Higher is Better)",
                yaxis_title="R² Score",
                height=400,
                template='plotly_white'
            )
            st.plotly_chart(fig_r2, use_container_width=True)
        
        # Metrics table
        st.subheader("Detailed Metrics")
        metrics_df = pd.DataFrame({
            'Model': model_names,
            'MAE': [f"{results[m]['metrics']['MAE']:.2f}" for m in model_names],
            'RMSE': [f"{results[m]['metrics']['RMSE']:.2f}" for m in model_names],
            'R²': [f"{results[m]['metrics']['R2']:.4f}" for m in model_names],
            'MAPE': [f"{results[m]['metrics']['MAPE']:.2f}%" for m in model_names]
        })
        
        # Display without styling (removed jinja2 dependency)
        st.dataframe(metrics_df, width='stretch')
        
        # Highlight best model manually
        best_idx = model_names.index(best_model[0])
        st.success(f"✅ **Best Model:** {best_model[0]} (Row {best_idx + 1})")
    
    with tab3:
        st.subheader("📦 Inventory Optimization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Inventory Metrics")
            
            metrics_display = {
                "📈 Average Daily Demand": f"{inventory_metrics['avg_daily_demand']:.0f} units",
                "🛡️ Safety Stock": f"{inventory_metrics['safety_stock']:.0f} units",
                "🔔 Reorder Point": f"{inventory_metrics['reorder_point']:.0f} units",
                "📦 Economic Order Quantity": f"{inventory_metrics['eoq']:.0f} units",
                "✅ Service Level": f"{service_level*100:.0f}%",
                "⚠️ Stockout Risk": f"{(1-service_level)*100:.1f}%"
            }
            
            for key, value in metrics_display.items():
                st.markdown(f"**{key}:** `{value}`")
        
        with col2:
            # Bar chart of inventory metrics
            inventory_data = pd.DataFrame({
                'Metric': ['Avg Daily\nDemand', 'Safety\nStock', 'Reorder\nPoint', 'EOQ'],
                'Value': [
                    inventory_metrics['avg_daily_demand'],
                    inventory_metrics['safety_stock'],
                    inventory_metrics['reorder_point'],
                    inventory_metrics['eoq']
                ],
                'Color': ['#3498db', '#e74c3c', '#f39c12', '#2ecc71']
            })
            
            fig_inv = px.bar(
                inventory_data,
                x='Metric',
                y='Value',
                color='Color',
                title="Inventory Policy Visualization",
                text='Value'
            )
            fig_inv.update_traces(texttemplate='%{text:.0f}', textposition='outside')
            fig_inv.update_layout(showlegend=False, height=400, template='plotly_white')
            st.plotly_chart(fig_inv, use_container_width=True)
        
        # Recommendations
        st.markdown("### 💡 Smart Inventory Recommendations")
        
        st.success(f"""
        **Optimal Inventory Policy for {product}:**
        
        ✅ **Step 1:** Maintain a minimum of **{inventory_metrics['safety_stock']:.0f} units** as safety stock  
        ✅ **Step 2:** Set up an alert to reorder when inventory reaches **{inventory_metrics['reorder_point']:.0f} units**  
        ✅ **Step 3:** Each order should be for **{inventory_metrics['eoq']:.0f} units** (optimizes costs)  
        ✅ **Step 4:** This ensures **{service_level*100:.0f}%** service level with only **{(1-service_level)*100:.1f}%** stockout risk  
        
        💰 **Cost Savings:** Optimized ordering reduces holding costs while maintaining high service levels!
        """)
        
        # Cost analysis
        st.markdown("### 💰 Cost Analysis")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            holding_cost = inventory_metrics['safety_stock'] * 2
            st.metric("Annual Holding Cost", f"₹{holding_cost:.0f}")
        
        with col2:
            orders_per_year = inventory_metrics['avg_daily_demand'] * 365 / inventory_metrics['eoq']
            st.metric("Orders per Year", f"{orders_per_year:.0f}")
        
        with col3:
            ordering_cost = orders_per_year * 50
            st.metric("Annual Ordering Cost", f"₹{ordering_cost:.0f}")
    
    with tab4:
        st.subheader("📋 Dataset Overview")
        
        # Dataset statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Date Range", f"{(df['date'].max() - df['date'].min()).days} days")
        with col3:
            st.metric("Products", df['product'].nunique())
        
        # Product summary
        st.markdown("### 📊 All Products Summary")
        summary_df = df.groupby('product').agg({
            'demand': ['mean', 'std', 'min', 'max'],
            'price': 'first'
        }).round(2)
        summary_df.columns = ['Avg Demand', 'Std Dev', 'Min', 'Max', 'Price (₹)']
        st.dataframe(summary_df, width='stretch')
        
        # Show product data
        st.markdown(f"### 📝 Recent Data for {product} (Last 50 Records)")
        product_data = df[df['product'] == product].tail(50)
        st.dataframe(product_data, width='stretch')
        
        # Download buttons
        col1, col2 = st.columns(2)
        with col1:
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Dataset (CSV)",
                data=csv,
                file_name="retail_sales_data.csv",
                mime="text/csv"
            )
        
        with col2:
            product_csv = product_data.to_csv(index=False)
            st.download_button(
                label=f"📥 Download {product} Data (CSV)",
                data=product_csv,
                file_name=f"{product.lower().replace(' ', '_')}_data.csv",
                mime="text/csv"
            )

else:
    # Welcome screen
    st.markdown("""
    ## 👋 Welcome to the AI Demand Forecasting Dashboard!
    
    This interactive tool helps you:
    - 📈 **Forecast demand** using machine learning models
    - 📊 **Compare models** to find the best performer
    - 📦 **Optimize inventory** with safety stock and reorder points
    - 💡 **Get recommendations** for inventory management
    
    ### 🛍️ Available Products:
    """)
    
    cols = st.columns(5)
    products_display = [
        ('👗', 'Fashion', '₹999'),
        ('❄️', 'Air Cooler', '₹6,500'),
        ('📱', 'Electronics', '₹18,000'),
        ('💄', 'Beauty', '₹499'),
        ('🛒', 'Grocery', '₹120')
    ]
    
    for col, (emoji, name, price) in zip(cols, products_display):
        with col:
            st.markdown(f"""
            <div style='text-align: center; padding: 1rem; background: #f0f2f6; border-radius: 10px;'>
                <div style='font-size: 3rem;'>{emoji}</div>
                <div style='font-weight: bold;'>{name}</div>
                <div style='color: #666;'>{price}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("""
    ### 🚀 Getting Started:
    1. Select a product from the sidebar 👈
    2. Adjust model and inventory settings
    3. Click **"🚀 Run Analysis"** button
    4. Explore results in different tabs
    
    ### 📊 Features:
    - **3 ML Models**: Random Forest, Gradient Boosting, Linear Regression
    - **Interactive Charts**: Zoom, pan, and explore data with Plotly
    - **Real-time Updates**: Change settings and see results instantly
    - **Export Data**: Download datasets and results as CSV
    - **Professional Analytics**: Industry-standard metrics (MAE, RMSE, R², MAPE)
    
    **Ready to start?** 👉 Configure settings in the sidebar and click "Run Analysis"!
    """)
    
    # Show sample data
    st.subheader("📋 Sample Dataset Preview")
    st.dataframe(df.head(20), width='stretch')
    
    # Show statistics
    st.subheader("📊 Product Statistics")
    stats_df = df.groupby('product')['demand'].agg(['mean', 'std', 'min', 'max']).round(2)
    stats_df.columns = ['Average', 'Std Dev', 'Minimum', 'Maximum']
    st.dataframe(stats_df, width='stretch')

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🤖 AI Demand Forecasting & Inventory Optimization System | Built with Streamlit & Python</p>
    <p style='font-size: 0.8rem;'>Fashion • Air Cooler • Electronic Devices • Beauty Products • Grocery</p>
</div>
""", unsafe_allow_html=True)
