"""
AI Demand Forecasting and Inventory Optimization System
========================================================
This project demonstrates:
1. Time series demand forecasting using multiple ML models
2. Inventory optimization with safety stock calculations
3. Comprehensive visualizations and performance metrics

WINDOWS VERSION - Fixed file paths for Windows OS
CORRECTED - Proper product names and analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import os
warnings.filterwarnings('ignore')

# Machine Learning Models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Statistical Models - using simple moving average as alternative
from scipy import stats

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class DemandForecastingSystem:
    """Complete demand forecasting and inventory optimization system"""
    
    def __init__(self):
        self.models = {}
        self.predictions = {}
        self.metrics = {}
        self.scaler = StandardScaler()
        
    def generate_synthetic_data(self, n_days=730, n_products=5):
        """Generate realistic synthetic retail sales data"""
        print("Generating synthetic retail dataset...")
        
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
                'description': 'Clothing and apparel items'
            },
            'Air Cooler': {
                'base_demand': 40, 
                'seasonality': 80,  # High seasonality (summer demand)
                'trend': 0.02, 
                'price': 6500.0,
                'description': 'Seasonal cooling appliances'
            },
            'Electronic Devices': {
                'base_demand': 55, 
                'seasonality': 25, 
                'trend': 0.06,  # Growing trend
                'price': 18000.0,
                'description': 'Smartphones, tablets, gadgets'
            },
            'Beauty Products': {
                'base_demand': 90, 
                'seasonality': 30, 
                'trend': 0.05, 
                'price': 499.0,
                'description': 'Cosmetics and personal care'
            },
            'Grocery': {
                'base_demand': 160, 
                'seasonality': 15,  # Low seasonality (consistent demand)
                'trend': 0.03, 
                'price': 120.0,
                'description': 'Daily essentials and food items'
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
        
        df = pd.DataFrame(data_list)
        print(f"Generated {len(df)} records for {n_products} products over {n_days} days")
        print(f"\nProduct Categories:")
        for product_name, info in product_info.items():
            print(f"  • {product_name}: {info['description']} (₹{info['price']:.0f})")
        return df
    
    def create_features(self, df, product_name):
        """Create time series features for ML models"""
        product_df = df[df['product'] == product_name].copy()
        product_df = product_df.sort_values('date').reset_index(drop=True)
        
        # Lag features
        for lag in [1, 7, 14, 30]:
            product_df[f'lag_{lag}'] = product_df['demand'].shift(lag)
        
        # Rolling statistics
        for window in [7, 14, 30]:
            product_df[f'rolling_mean_{window}'] = product_df['demand'].rolling(window=window).mean()
            product_df[f'rolling_std_{window}'] = product_df['demand'].rolling(window=window).std()
        
        # Date features
        product_df['year'] = product_df['date'].dt.year
        product_df['day_of_year'] = product_df['date'].dt.dayofyear
        
        # Drop NaN values created by lag and rolling features
        product_df = product_df.dropna()
        
        return product_df
    
    def train_models(self, df, product_name, test_size=0.2):
        """Train multiple forecasting models"""
        print(f"\n{'='*60}")
        print(f"Training models for {product_name}")
        print(f"{'='*60}")
        
        # Prepare data
        product_df = self.create_features(df, product_name)
        
        # Split features and target
        feature_cols = [col for col in product_df.columns 
                       if col not in ['date', 'product', 'demand', 'price']]
        
        X = product_df[feature_cols]
        y = product_df['demand']
        
        # Train-test split
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        dates_test = product_df['date'].iloc[split_idx:].values
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
            'Linear Regression': LinearRegression()
        }
        
        results = {}
        
        for model_name, model in models.items():
            print(f"\nTraining {model_name}...")
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            
            results[model_name] = {
                'model': model,
                'predictions': y_pred,
                'actual': y_test.values,
                'dates': dates_test,
                'metrics': {
                    'MAE': mae,
                    'RMSE': rmse,
                    'R2': r2,
                    'MAPE': mape
                }
            }
            
            print(f"  MAE: {mae:.2f}")
            print(f"  RMSE: {rmse:.2f}")
            print(f"  R²: {r2:.4f}")
            print(f"  MAPE: {mape:.2f}%")
        
        return results
    
    def calculate_inventory_metrics(self, predictions, actual_demand, lead_time=7, service_level=0.95):
        """Calculate optimal inventory levels"""
        
        # Calculate forecast error
        errors = actual_demand - predictions
        error_std = np.std(errors)
        
        # Average demand during lead time
        avg_demand_lead_time = np.mean(predictions) * lead_time
        
        # Safety stock calculation (using z-score for service level)
        z_score = stats.norm.ppf(service_level)
        safety_stock = z_score * error_std * np.sqrt(lead_time)
        
        # Reorder point
        reorder_point = avg_demand_lead_time + safety_stock
        
        # Economic Order Quantity (EOQ) - simplified
        annual_demand = np.sum(predictions) * (365 / len(predictions))
        holding_cost_per_unit = 2  # Example: ₹2 per unit per year
        order_cost = 50  # Example: ₹50 per order
        
        eoq = np.sqrt((2 * annual_demand * order_cost) / holding_cost_per_unit)
        
        return {
            'avg_daily_demand': np.mean(predictions),
            'safety_stock': safety_stock,
            'reorder_point': reorder_point,
            'economic_order_quantity': eoq,
            'service_level': service_level,
            'forecast_error_std': error_std,
            'avg_demand_lead_time': avg_demand_lead_time
        }
    
    def visualize_results(self, results, product_name, inventory_metrics, output_file='forecast_analysis.png'):
        """Create comprehensive visualizations"""
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Forecast vs Actual
        ax1 = plt.subplot(3, 3, 1)
        best_model = min(results.items(), key=lambda x: x[1]['metrics']['MAPE'])
        dates = best_model[1]['dates']
        
        plt.plot(dates, best_model[1]['actual'], label='Actual Demand', 
                linewidth=2, marker='o', markersize=3, alpha=0.7)
        plt.plot(dates, best_model[1]['predictions'], label=f'{best_model[0]} Forecast', 
                linewidth=2, marker='s', markersize=3, alpha=0.7)
        plt.fill_between(dates, 
                         best_model[1]['predictions'] - inventory_metrics['forecast_error_std'],
                         best_model[1]['predictions'] + inventory_metrics['forecast_error_std'],
                         alpha=0.2, label='Confidence Interval')
        plt.xlabel('Date')
        plt.ylabel('Demand')
        plt.title(f'{product_name}: Best Model Forecast ({best_model[0]})')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 2. Model Comparison - MAPE
        ax2 = plt.subplot(3, 3, 2)
        model_names = list(results.keys())
        mapes = [results[m]['metrics']['MAPE'] for m in model_names]
        colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))
        bars = plt.bar(model_names, mapes, color=colors, edgecolor='black', linewidth=1.5)
        plt.xlabel('Model')
        plt.ylabel('MAPE (%)')
        plt.title('Model Performance Comparison (MAPE)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 3. R² Score Comparison
        ax3 = plt.subplot(3, 3, 3)
        r2_scores = [results[m]['metrics']['R2'] for m in model_names]
        bars = plt.bar(model_names, r2_scores, color=colors, edgecolor='black', linewidth=1.5)
        plt.xlabel('Model')
        plt.ylabel('R² Score')
        plt.title('Model Performance Comparison (R²)')
        plt.xticks(rotation=45, ha='right')
        plt.ylim([0, 1])
        plt.grid(True, alpha=0.3, axis='y')
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Forecast Error Distribution
        ax4 = plt.subplot(3, 3, 4)
        errors = best_model[1]['actual'] - best_model[1]['predictions']
        plt.hist(errors, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        plt.axvline(x=np.mean(errors), color='green', linestyle='--', 
                   linewidth=2, label=f'Mean Error: {np.mean(errors):.2f}')
        plt.xlabel('Forecast Error')
        plt.ylabel('Frequency')
        plt.title('Forecast Error Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        
        # 5. Inventory Policy Visualization
        ax5 = plt.subplot(3, 3, 5)
        inventory_levels = ['Average\nDaily Demand', 'Safety\nStock', 'Reorder\nPoint', 'EOQ']
        values = [
            inventory_metrics['avg_daily_demand'],
            inventory_metrics['safety_stock'],
            inventory_metrics['reorder_point'],
            inventory_metrics['economic_order_quantity']
        ]
        colors_inv = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71']
        bars = plt.bar(inventory_levels, values, color=colors_inv, edgecolor='black', linewidth=1.5)
        plt.ylabel('Units')
        plt.title('Inventory Optimization Metrics')
        plt.grid(True, alpha=0.3, axis='y')
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0f}', ha='center', va='bottom', fontweight='bold')
        
        # 6. MAE Comparison
        ax6 = plt.subplot(3, 3, 6)
        maes = [results[m]['metrics']['MAE'] for m in model_names]
        bars = plt.bar(model_names, maes, color=colors, edgecolor='black', linewidth=1.5)
        plt.xlabel('Model')
        plt.ylabel('MAE')
        plt.title('Model Performance Comparison (MAE)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 7. All Models Forecast Comparison
        ax7 = plt.subplot(3, 3, 7)
        plt.plot(dates, best_model[1]['actual'], label='Actual', 
                linewidth=2.5, color='black', alpha=0.8)
        for i, (model_name, result) in enumerate(results.items()):
            plt.plot(dates[::5], result['predictions'][::5], label=model_name, 
                    linewidth=1.5, marker='o', markersize=4, alpha=0.6)
        plt.xlabel('Date')
        plt.ylabel('Demand')
        plt.title('All Models Forecast Comparison')
        plt.legend(loc='best', fontsize=8)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 8. Residual Plot
        ax8 = plt.subplot(3, 3, 8)
        plt.scatter(best_model[1]['predictions'], errors, alpha=0.6, edgecolors='black', s=50)
        plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
        plt.xlabel('Predicted Demand')
        plt.ylabel('Residual (Actual - Predicted)')
        plt.title('Residual Plot')
        plt.grid(True, alpha=0.3)
        
        # 9. Metrics Summary Table
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('tight')
        ax9.axis('off')
        
        metrics_data = []
        for model_name, result in results.items():
            metrics_data.append([
                model_name,
                f"{result['metrics']['MAE']:.2f}",
                f"{result['metrics']['RMSE']:.2f}",
                f"{result['metrics']['R2']:.3f}",
                f"{result['metrics']['MAPE']:.2f}%"
            ])
        
        table = ax9.table(cellText=metrics_data,
                         colLabels=['Model', 'MAE', 'RMSE', 'R²', 'MAPE'],
                         cellLoc='center',
                         loc='center',
                         colColours=['lightgray']*5)
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        ax9.set_title('Performance Metrics Summary', fontsize=12, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved to: {output_file}")
        plt.close()
    
    def generate_recommendations(self, product_name, inventory_metrics, results):
        """Generate actionable inventory recommendations"""
        best_model = min(results.items(), key=lambda x: x[1]['metrics']['MAPE'])
        
        recommendations = {
            'product': product_name,
            'best_model': best_model[0],
            'model_accuracy': f"{best_model[1]['metrics']['MAPE']:.2f}% MAPE",
            'inventory_recommendations': {
                'maintain_safety_stock': f"{inventory_metrics['safety_stock']:.0f} units",
                'reorder_when_inventory_reaches': f"{inventory_metrics['reorder_point']:.0f} units",
                'optimal_order_quantity': f"{inventory_metrics['economic_order_quantity']:.0f} units",
                'expected_daily_demand': f"{inventory_metrics['avg_daily_demand']:.0f} units",
                'service_level_target': f"{inventory_metrics['service_level']*100:.0f}%"
            },
            'cost_implications': {
                'estimated_holding_cost_per_order': f"₹{inventory_metrics['safety_stock'] * 2:.2f}",
                'stockout_risk': f"{(1 - inventory_metrics['service_level'])*100:.1f}%"
            }
        }
        
        return recommendations

def main():
    """Main execution function"""
    print("="*80)
    print("AI DEMAND FORECASTING AND INVENTORY OPTIMIZATION SYSTEM")
    print("="*80)
    
    # Initialize system
    system = DemandForecastingSystem()
    
    # Generate data
    df = system.generate_synthetic_data(n_days=730, n_products=5)
    
    # Save dataset to CURRENT DIRECTORY (Windows-compatible)
    df.to_csv('retail_sales_data.csv', index=False)
    print(f"\nDataset saved to: retail_sales_data.csv")
    print(f"Dataset shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head(10))
    
    print(f"\nDataset Statistics:")
    print(df.groupby('product')['demand'].agg(['mean', 'std', 'min', 'max']))
    
    # CORRECTED: Select a product that actually exists in the data
    # You can change this to any of: 'Fashion', 'Air Cooler', 'Electronic Devices', 'Beauty Products', 'Grocery'
    product_name = 'Fashion'  # Changed from 'Product_A' to actual product name
    
    print(f"\n{'='*60}")
    print(f"Analyzing product: {product_name}")
    print(f"{'='*60}")
    
    # Train models
    results = system.train_models(df, product_name, test_size=0.2)
    
    # Get best model for inventory calculations
    best_model = min(results.items(), key=lambda x: x[1]['metrics']['MAPE'])
    
    # Calculate inventory metrics
    inventory_metrics = system.calculate_inventory_metrics(
        predictions=best_model[1]['predictions'],
        actual_demand=best_model[1]['actual'],
        lead_time=7,
        service_level=0.95
    )
    
    print(f"\n{'='*60}")
    print(f"INVENTORY OPTIMIZATION METRICS FOR {product_name}")
    print(f"{'='*60}")
    for key, value in inventory_metrics.items():
        print(f"{key.replace('_', ' ').title()}: {value:.2f}")
    
    # Generate visualizations (save to current directory)
    system.visualize_results(results, product_name, inventory_metrics, 'forecast_analysis.png')
    
    # Generate recommendations
    recommendations = system.generate_recommendations(product_name, inventory_metrics, results)
    
    print(f"\n{'='*60}")
    print(f"RECOMMENDATIONS FOR {product_name}")
    print(f"{'='*60}")
    print(f"\nBest Model: {recommendations['best_model']}")
    print(f"Model Accuracy: {recommendations['model_accuracy']}")
    print(f"\nInventory Policy:")
    for key, value in recommendations['inventory_recommendations'].items():
        print(f"  • {key.replace('_', ' ').title()}: {value}")
    print(f"\nCost Implications:")
    for key, value in recommendations['cost_implications'].items():
        print(f"  • {key.replace('_', ' ').title()}: {value}")
    
    # Save recommendations to current directory
    import json
    with open('recommendations.json', 'w') as f:
        json.dump(recommendations, f, indent=4)
    print(f"\nRecommendations saved to: recommendations.json")
    
    print("\n" + "="*80)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\nGenerated Files (in current folder):")
    print("  1. retail_sales_data.csv - Complete dataset")
    print("  2. forecast_analysis.png - Comprehensive visualizations")
    print("  3. recommendations.json - Inventory recommendations")
    print(f"\nAll files saved in: {os.getcwd()}")
    
    print("\n" + "="*80)
    print("TIP: To analyze a different product, change line 435:")
    print("Available products: Fashion, Air Cooler, Electronic Devices, Beauty Products, Grocery")
    print("="*80)

if __name__ == "__main__":
    main()
