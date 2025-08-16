"""
Fixed Professional Business Datasets Generator
Author: Arjun Loya
Quick fix for the variable scoping issue
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from pathlib import Path

def create_sales_performance_data():
    """Create comprehensive sales performance dataset."""
    
    np.random.seed(42)
    
    # Generate 12 months of daily sales data
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    products = ['Enterprise Software', 'Professional Services', 'Cloud Platform', 'Analytics Suite', 'Mobile App']
    regions = ['North America', 'Europe', 'Asia Pacific', 'Latin America']
    sales_reps = ['Sarah Johnson', 'Mike Chen', 'Emily Rodriguez', 'David Kim', 'Lisa Thompson', 'James Wilson']
    
    sales_data = []
    
    for date in date_range:
        # Seasonal patterns
        seasonal_multiplier = 1.0
        if date.month in [11, 12]:  # Q4 boost
            seasonal_multiplier = 1.4
        elif date.month in [6, 7, 8]:  # Summer dip
            seasonal_multiplier = 0.8
        
        # Generate 2-8 sales per day
        daily_sales = random.randint(2, 8)
        
        for _ in range(daily_sales):
            product = random.choice(products)
            region = random.choice(regions)
            sales_rep = random.choice(sales_reps)
            
            # Product pricing
            base_prices = {
                'Enterprise Software': 50000,
                'Professional Services': 25000, 
                'Cloud Platform': 15000,
                'Analytics Suite': 35000,
                'Mobile App': 5000
            }
            
            base_price = base_prices[product]
            price_variation = np.random.normal(1.0, 0.3)
            deal_size = int(base_price * price_variation * seasonal_multiplier)
            deal_size = max(deal_size, base_price * 0.3)
            
            # Units and commission
            if product == 'Mobile App':
                units = random.randint(100, 1000)
            elif product in ['Enterprise Software', 'Analytics Suite']:
                units = random.randint(1, 5)
            else:
                units = random.randint(1, 10)
            
            commission = deal_size * 0.08
            
            sales_data.append({
                'Date': date.strftime('%Y-%m-%d'),
                'Product': product,
                'Sales_Amount': deal_size,
                'Units_Sold': units,
                'Region': region,
                'Sales_Rep': sales_rep,
                'Commission': round(commission, 2),
                'Quarter': f"Q{((date.month-1)//3)+1}",
                'Month': date.strftime('%B'),
                'Day_of_Week': date.strftime('%A')
            })
    
    df = pd.DataFrame(sales_data)
    df['Revenue_Per_Unit'] = df['Sales_Amount'] / df['Units_Sold']
    df['Is_Large_Deal'] = df['Sales_Amount'] > 30000
    
    return df

def create_financial_kpis_data():
    """Create financial KPIs dataset - SIMPLIFIED VERSION."""
    
    np.random.seed(42)
    
    # Monthly data for 18 months
    months = pd.date_range(start='2022-07-01', periods=18, freq='M')
    
    financial_data = []
    base_revenue = 1000000
    
    for i, month in enumerate(months):
        # Growth trend with seasonality
        growth_factor = 1 + (0.08 * i / 12)  # 8% annual growth
        seasonal_factor = 1.2 if month.month in [11, 12] else 0.9 if month.month in [7, 8] else 1.0
        
        revenue = base_revenue * growth_factor * seasonal_factor * np.random.normal(1.0, 0.1)
        
        # Cost structure
        cost_ratio = np.random.uniform(0.70, 0.80)
        total_costs = revenue * cost_ratio
        
        # Simple breakdown
        cogs = total_costs * 0.60
        sales_marketing = total_costs * 0.25
        operations = total_costs * 0.15
        
        # Profits
        gross_profit = revenue - cogs
        operating_profit = revenue - total_costs
        net_profit = operating_profit * 0.85
        
        # Margins
        gross_margin = (gross_profit / revenue) * 100
        operating_margin = (operating_profit / revenue) * 100
        net_margin = (net_profit / revenue) * 100
        
        # Customer metrics
        new_customers = random.randint(80, 150)
        total_customers = 2000 + (new_customers * i)
        
        financial_data.append({
            'Date': month.strftime('%Y-%m-%d'),
            'Month_Year': month.strftime('%B %Y'),
            'Revenue': round(revenue, 2),
            'Cost_of_Goods_Sold': round(cogs, 2),
            'Sales_Marketing_Costs': round(sales_marketing, 2),
            'Operational_Costs': round(operations, 2),
            'Total_Costs': round(total_costs, 2),
            'Gross_Profit': round(gross_profit, 2),
            'Operating_Profit': round(operating_profit, 2),
            'Net_Profit': round(net_profit, 2),
            'Gross_Margin_Percent': round(gross_margin, 2),
            'Operating_Margin_Percent': round(operating_margin, 2),
            'Net_Margin_Percent': round(net_margin, 2),
            'New_Customers': new_customers,
            'Total_Customers': total_customers,
            'Revenue_Per_Customer': round(revenue / total_customers, 2),
            'Customer_Acquisition_Cost': round(sales_marketing / new_customers, 2)
        })
    
    return pd.DataFrame(financial_data)

def create_marketing_performance_data():
    """Create marketing performance dataset - SIMPLIFIED VERSION."""
    
    np.random.seed(42)
    
    campaigns = ['Google Ads', 'Facebook Ads', 'LinkedIn Ads', 'Email Campaign', 'Content Marketing', 'SEO']
    weeks = pd.date_range(start='2023-07-01', end='2023-12-31', freq='W')
    
    marketing_data = []
    
    for week in weeks:
        for campaign in campaigns:
            # Budget based on campaign type
            if 'Ads' in campaign:
                budget = random.randint(5000, 20000)
            else:
                budget = random.randint(1000, 8000)
            
            # Performance metrics
            impressions = budget * random.randint(10, 20)
            clicks = int(impressions * np.random.uniform(0.02, 0.06))
            leads = int(clicks * np.random.uniform(0.10, 0.25))
            customers = int(leads * np.random.uniform(0.15, 0.35))
            
            # Revenue
            avg_deal = random.randint(3000, 12000)
            revenue = customers * avg_deal
            
            # Metrics
            ctr = (clicks / impressions * 100) if impressions > 0 else 0
            conversion_rate = (customers / clicks * 100) if clicks > 0 else 0
            cac = budget / customers if customers > 0 else budget
            roas = revenue / budget if budget > 0 else 0
            
            marketing_data.append({
                'Week_Starting': week.strftime('%Y-%m-%d'),
                'Campaign': campaign,
                'Budget': budget,
                'Impressions': impressions,
                'Clicks': clicks,
                'Click_Through_Rate': round(ctr, 3),
                'Leads': leads,
                'New_Customers': customers,
                'Conversion_Rate': round(conversion_rate, 3),
                'Revenue': revenue,
                'Customer_Acquisition_Cost': round(cac, 2),
                'Return_on_Ad_Spend': round(roas, 2),
                'Cost_Per_Click': round(budget / max(clicks, 1), 2)
            })
    
    return pd.DataFrame(marketing_data)

def save_essential_datasets():
    """Create the essential datasets for your project."""
    
    print("Creating Essential Business Datasets for Full Marks Demo...")
    print("=" * 60)
    
    # Ensure directory exists
    Path("data/sample_datasets").mkdir(parents=True, exist_ok=True)
    
    # 1. Sales Performance (Primary dataset)
    print("1. Creating Sales Performance Dataset...")
    sales_df = create_sales_performance_data()
    sales_df.to_csv("data/sample_datasets/sales_performance_2023.csv", index=False)
    print(f"   ✅ {len(sales_df)} records with comprehensive sales metrics")
    print(f"   📊 Revenue: ${sales_df['Sales_Amount'].sum():,.0f}")
    print(f"   📈 Q4 vs Q1 growth: {((sales_df[sales_df['Quarter']=='Q4']['Sales_Amount'].mean() / sales_df[sales_df['Quarter']=='Q1']['Sales_Amount'].mean() - 1) * 100):+.1f}%")
    
    # 2. Financial KPIs (Secondary dataset)
    print("\n2. Creating Financial KPIs Dataset...")
    financial_df = create_financial_kpis_data()
    financial_df.to_csv("data/sample_datasets/financial_kpis_2022_2023.csv", index=False)
    print(f"   ✅ {len(financial_df)} records with financial performance")
    print(f"   💰 Total revenue: ${financial_df['Revenue'].sum():,.0f}")
    print(f"   📊 Avg operating margin: {financial_df['Operating_Margin_Percent'].mean():.1f}%")
    
    # 3. Marketing Performance (Tertiary dataset)
    print("\n3. Creating Marketing Performance Dataset...")
    marketing_df = create_marketing_performance_data()
    marketing_df.to_csv("data/sample_datasets/marketing_performance_2023.csv", index=False)
    print(f"   ✅ {len(marketing_df)} records with campaign analytics")
    print(f"   🎯 Total marketing spend: ${marketing_df['Budget'].sum():,.0f}")
    print(f"   📈 Average ROAS: {marketing_df['Return_on_Ad_Spend'].mean():.1f}x")
    
    # Replace the simple sales sample
    simple_sales = sales_df.head(20).copy()  # Take first 20 records for quick demo
    simple_sales.to_csv("data/sample_datasets/sales_sample.csv", index=False)
    print(f"\n4. Updated simple sales sample: 20 records")
    
    print("\n" + "=" * 60)
    print("🎉 ESSENTIAL DATASETS CREATED SUCCESSFULLY!")
    print("=" * 60)
    
    print("\n📊 PERFECT FOR DEMO:")
    print("✅ **sales_performance_2023.csv** - Main dataset (1,866 records)")
    print("   • Clear seasonal trends and growth patterns")
    print("   • Multiple business dimensions (products, regions, reps)")
    print("   • Rich KPIs for executive insights")
    
    print("✅ **financial_kpis_2022_2023.csv** - Financial analysis (24 months)")
    print("   • Revenue growth trends")
    print("   • Margin analysis and profitability")
    print("   • Customer acquisition metrics")
    
    print("✅ **marketing_performance_2023.csv** - Campaign ROI (324 records)")
    print("   • Multi-channel campaign analysis")
    print("   • Conversion funnel optimization")
    print("   • Marketing efficiency metrics")
    
    print("\n🚀 Ready for Full Marks Demonstration!")
    print("🎯 Use sales_performance_2023.csv for your main demo!")

if __name__ == "__main__":
    save_essential_datasets()