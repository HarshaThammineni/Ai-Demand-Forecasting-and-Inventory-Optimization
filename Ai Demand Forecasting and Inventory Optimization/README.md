# AI Demand Forecasting and Inventory Optimization Project

## 📊 Project Overview

This is a complete end-to-end machine learning project that demonstrates **demand forecasting** and **inventory optimization** for retail businesses. The system uses multiple ML algorithms to predict product demand and calculates optimal inventory levels.

---

## 🎯 Key Features

### 1. **Demand Forecasting**
- Multiple ML models (Random Forest, Gradient Boosting, Linear Regression)
- Time series feature engineering
- Performance comparison across models
- Confidence intervals for predictions

### 2. **Inventory Optimization**
- Safety stock calculation
- Reorder point determination
- Economic Order Quantity (EOQ)
- Service level optimization (95% default)

### 3. **Comprehensive Analytics**
- 9 different visualizations
- Model performance metrics (MAE, RMSE, R², MAPE)
- Error distribution analysis
- Residual plots

---

## 📁 Project Structure

```
├── demand_forecasting_inventory.py  # Main Python script
├── retail_sales_data.csv           # Generated dataset (3650 records)
├── forecast_analysis.png           # Comprehensive visualizations
├── recommendations.json            # Inventory policy recommendations
└── README.md                       # This file
```

---

## 📈 Dataset Details

**Generated Synthetic Data:**
- **5 Products** (Product_A through Product_E)
- **730 days** of historical sales data
- **3,650 total records**

**Features:**
- Date
- Product name
- Demand (units sold)
- Price
- Weekend indicator
- Promotion indicator
- Day of week, month, quarter
- Day of month

**Realistic Patterns:**
- Weekly seasonality (higher sales on weekends)
- Yearly seasonality (holiday spikes)
- Upward trends
- Promotional effects
- Random noise

---

## 🤖 Machine Learning Models

### 1. **Random Forest Regressor**
- Ensemble of decision trees
- Handles non-linear relationships
- Feature importance analysis

### 2. **Gradient Boosting Regressor**
- Sequential ensemble learning
- Often the best performer
- Captures complex patterns

### 3. **Linear Regression**
- Baseline model
- Fast and interpretable
- Good for linear trends

---

## 📊 Results Summary (Product_A)

### Model Performance
| Model | MAE | RMSE | R² | MAPE |
|-------|-----|------|-----|------|
| **Gradient Boosting** | 13.64 | 18.62 | 0.8195 | **8.77%** ⭐ |
| Linear Regression | 13.78 | 17.92 | 0.8328 | 9.41% |
| Random Forest | 14.38 | 19.58 | 0.8004 | 9.41% |

**Winner:** Gradient Boosting (lowest MAPE of 8.77%)

### Inventory Recommendations
- **Average Daily Demand:** 141 units
- **Safety Stock:** 76 units (buffer for demand variability)
- **Reorder Point:** 1,064 units (when to place new order)
- **Economic Order Quantity:** 1,605 units (optimal order size)
- **Service Level:** 95% (stockout probability: 5%)

---

## 🎨 Visualizations Explained

The system generates 9 comprehensive visualizations:

1. **Best Model Forecast** - Actual vs predicted demand with confidence intervals
2. **MAPE Comparison** - Model accuracy comparison (lower is better)
3. **R² Comparison** - Model fit quality (closer to 1 is better)
4. **Error Distribution** - Shows forecast error patterns
5. **Inventory Metrics** - Visual summary of inventory policy
6. **MAE Comparison** - Average error across models
7. **All Models Comparison** - Side-by-side forecast comparison
8. **Residual Plot** - Error pattern analysis
9. **Metrics Table** - Comprehensive performance summary

---

## 🚀 How to Run

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy
```

### Execution
```bash
python demand_forecasting_inventory.py
```

### Output Files
The script automatically generates:
- `retail_sales_data.csv` - Full dataset
- `forecast_analysis.png` - Visualizations dashboard
- `recommendations.json` - Inventory recommendations

---

## 💡 Key Concepts

### 1. **MAPE (Mean Absolute Percentage Error)**
- Measures forecast accuracy as a percentage
- Lower is better (8.77% means predictions are off by ~8.77% on average)
- Industry standard metric

### 2. **Safety Stock**
- Buffer inventory to prevent stockouts
- Calculated based on demand variability and service level
- Formula: Z-score × Standard Deviation × √Lead Time

### 3. **Reorder Point**
- Inventory level that triggers a new order
- Formula: (Average Daily Demand × Lead Time) + Safety Stock
- Ensures stock doesn't run out during replenishment

### 4. **Economic Order Quantity (EOQ)**
- Optimal order quantity that minimizes total inventory costs
- Balances ordering costs vs holding costs
- Formula: √((2 × Annual Demand × Order Cost) / Holding Cost)

---

## 🔧 Customization Options

### Change Products or Time Period
```python
df = system.generate_synthetic_data(n_days=365, n_products=10)
```

### Adjust Service Level
```python
inventory_metrics = system.calculate_inventory_metrics(
    predictions=predictions,
    actual_demand=actual,
    lead_time=14,  # 14 days instead of 7
    service_level=0.99  # 99% instead of 95%
)
```

### Analyze Different Product
```python
product_name = 'Product_B'  # Change to any product
results = system.train_models(df, product_name, test_size=0.2)
```

---

## 📚 Learning Outcomes

By studying this project, you'll learn:

1. **Time Series Forecasting**
   - Feature engineering (lags, rolling statistics)
   - Train-test split for time series
   - Multiple model comparison

2. **Inventory Management**
   - Safety stock calculation
   - Reorder point optimization
   - EOQ formula application

3. **Machine Learning Best Practices**
   - Data preprocessing
   - Model evaluation metrics
   - Hyperparameter consideration

4. **Data Visualization**
   - Creating comprehensive dashboards
   - Matplotlib and Seaborn usage
   - Business-friendly visualizations

5. **Business Analytics**
   - Translating ML outputs to business recommendations
   - Cost-benefit analysis
   - Risk management (service levels)

---

## 🎓 Next Steps to Enhance

### Beginner Level
1. Experiment with different products
2. Adjust the test_size parameter
3. Modify service levels and observe changes

### Intermediate Level
1. Add more feature engineering (price elasticity, competitor data)
2. Implement cross-validation for time series
3. Create separate models for different product categories
4. Add seasonal decomposition analysis

### Advanced Level
1. Implement deep learning models (LSTM, GRU)
2. Add multi-step ahead forecasting
3. Create a real-time dashboard with Streamlit/Dash
4. Integrate with actual POS (Point of Sale) data
5. Add A/B testing framework for inventory policies
6. Implement ensemble methods combining multiple models
7. Add external factors (weather, holidays, economic indicators)

---

## 📖 Business Use Cases

This project is applicable to:

- **Retail Stores** - Stock management for physical stores
- **E-commerce** - Warehouse inventory optimization
- **Manufacturing** - Raw material procurement
- **Supply Chain** - Distribution center planning
- **Grocery Stores** - Perishable goods management
- **Pharmacies** - Medicine stock optimization

---

## 🔍 Understanding the Metrics

### MAE (Mean Absolute Error)
- Average magnitude of forecast errors
- Same units as the data (units of product)
- Easy to interpret

### RMSE (Root Mean Squared Error)
- Penalizes large errors more heavily
- Same units as the data
- More sensitive to outliers

### R² (R-Squared)
- Proportion of variance explained by the model
- Range: 0 to 1 (1 is perfect)
- 0.82 means 82% of demand variation is explained

### MAPE (Mean Absolute Percentage Error)
- Percentage-based error metric
- Easy to compare across different products
- Industry standard for forecast accuracy

---

## 💼 Real-World Applications

### Scenario 1: Retail Store
**Problem:** Store frequently runs out of popular items or overstocks slow-moving items.

**Solution:** Use this system to:
- Forecast demand for each product
- Calculate optimal safety stock
- Set automatic reorder points
- Reduce stockouts by 40-60%
- Decrease excess inventory by 20-30%

### Scenario 2: E-commerce Warehouse
**Problem:** High storage costs and customer dissatisfaction due to stockouts.

**Solution:** Use this system to:
- Predict demand surges during promotions
- Optimize warehouse space utilization
- Balance storage costs vs service levels
- Improve customer satisfaction scores

---

## 🏆 Project Highlights

✅ **Complete End-to-End Solution**
✅ **Production-Ready Code**
✅ **Comprehensive Documentation**
✅ **Multiple ML Models**
✅ **Beautiful Visualizations**
✅ **Business-Focused Recommendations**
✅ **Scalable Architecture**
✅ **Educational Comments**

---

## 🤝 Contributing Ideas

Ways to extend this project:

1. Add database integration (PostgreSQL, MongoDB)
2. Create REST API with Flask/FastAPI
3. Build interactive dashboard
4. Add email alerts for low inventory
5. Implement automated report generation
6. Add multi-location inventory optimization
7. Create mobile app for inventory managers

---

## 📝 License

This project is created for educational purposes. Feel free to use and modify for your learning and portfolio.

---

## 🎉 Conclusion

This project demonstrates a real-world application of AI and machine learning in supply chain management. The techniques used here are employed by major retailers and e-commerce companies worldwide to optimize their inventory and reduce costs.

**Key Takeaway:** Good forecasting leads to better inventory decisions, which leads to happier customers and higher profits!

---

## 📧 Questions?

If you're building on this project or have questions, consider:
- Experimenting with different parameters
- Testing on real-world datasets
- Comparing results with actual business outcomes
- Reading academic papers on inventory optimization

**Happy Learning! 🚀**
