#!/usr/bin/env python3
"""
Complete Month-over-Month Growth Analysis for Both Stores
This provides the exact analysis the user requested
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def analyze_month_over_month_growth():
    """Complete month-over-month growth analysis for both DJA and YAS stores"""
    
    # Connect to database
    db_path = 'data/ikea_store_database.db'
    conn = sqlite3.connect(db_path)
    
    print("=" * 80)
    print("📊 MONTH-OVER-MONTH GROWTH TRENDS ANALYSIS")
    print("🏪 Both Stores: DJA & YAS (January - June 2025)")
    print("=" * 80)
    
    # Main query to get monthly data for both stores
    query = """
    SELECT 
        Store,
        strftime('%Y-%m', Date) as Month,
        SUM(Act) as Total_Revenue_AED,
        SUM(Visitors) as Total_Visitors,
        SUM(Customers) as Total_Customers,
        AVG(Conversion) as Avg_Conversion_Rate,
        AVG(ATV) as Avg_ATV,
        COUNT(*) as Days_in_Month
    FROM store_data 
    WHERE Date BETWEEN '2025-01-01' AND '2025-06-30'
    GROUP BY Store, Month
    ORDER BY Store, Month
    """
    
    try:
        df = pd.read_sql_query(query, conn)
        
        if df.empty:
            print("❌ No data found for the specified period")
            return
        
        print("\n🔍 RAW MONTHLY DATA:")
        print("-" * 80)
        print(df.to_string(index=False))
        
        # Check which stores have data
        stores = df['Store'].unique()
        print(f"\n📋 Stores with data: {list(stores)}")
        
        # Calculate month-over-month growth for each store
        growth_analysis = {}
        
        for store in stores:
            store_data = df[df['Store'] == store].copy()
            store_data = store_data.sort_values('Month')
            
            # Calculate month-over-month growth
            store_data['Revenue_Growth_MoM'] = store_data['Total_Revenue_AED'].pct_change() * 100
            store_data['Visitors_Growth_MoM'] = store_data['Total_Visitors'].pct_change() * 100
            store_data['Customers_Growth_MoM'] = store_data['Total_Customers'].pct_change() * 100
            
            growth_analysis[store] = store_data
            
            print(f"\n🏪 {store} STORE - MONTH-OVER-MONTH GROWTH:")
            print("-" * 60)
            
            # Display monthly data with growth
            for _, row in store_data.iterrows():
                month = row['Month']
                revenue = row['Total_Revenue_AED']
                visitors = row['Total_Visitors']
                customers = row['Total_Customers']
                revenue_growth = row['Revenue_Growth_MoM']
                visitor_growth = row['Visitors_Growth_MoM']
                customer_growth = row['Customers_Growth_MoM']
                
                print(f"📅 {month}:")
                print(f"  💰 Revenue: AED {revenue:,.0f}", end="")
                if not pd.isna(revenue_growth):
                    print(f" ({revenue_growth:+.1f}% MoM)")
                else:
                    print(" (baseline month)")
                    
                print(f"  👥 Visitors: {visitors:,.0f}", end="")
                if not pd.isna(visitor_growth):
                    print(f" ({visitor_growth:+.1f}% MoM)")
                else:
                    print(" (baseline month)")
                    
                print(f"  🛒 Customers: {customers:,.0f}", end="")
                if not pd.isna(customer_growth):
                    print(f" ({customer_growth:+.1f}% MoM)")
                else:
                    print(" (baseline month)")
                print()
        
        # Summary analysis
        print("\n" + "=" * 80)
        print("📈 GROWTH TRENDS SUMMARY:")
        print("=" * 80)
        
        for store in stores:
            store_data = growth_analysis[store]
            
            # Calculate average growth rates (excluding NaN values)
            avg_revenue_growth = store_data['Revenue_Growth_MoM'].mean()
            avg_visitor_growth = store_data['Visitors_Growth_MoM'].mean()
            avg_customer_growth = store_data['Customers_Growth_MoM'].mean()
            
            # Find best and worst performing months
            best_revenue_month = store_data.loc[store_data['Revenue_Growth_MoM'].idxmax(), 'Month'] if not store_data['Revenue_Growth_MoM'].isna().all() else "N/A"
            worst_revenue_month = store_data.loc[store_data['Revenue_Growth_MoM'].idxmin(), 'Month'] if not store_data['Revenue_Growth_MoM'].isna().all() else "N/A"
            
            # Total period growth
            first_month_revenue = store_data.iloc[0]['Total_Revenue_AED']
            last_month_revenue = store_data.iloc[-1]['Total_Revenue_AED']
            total_period_growth = ((last_month_revenue - first_month_revenue) / first_month_revenue) * 100
            
            print(f"\n🏪 {store} STORE SUMMARY:")
            print(f"  📊 Average Monthly Growth Rates:")
            print(f"    • Revenue: {avg_revenue_growth:+.1f}%")
            print(f"    • Visitors: {avg_visitor_growth:+.1f}%") 
            print(f"    • Customers: {avg_customer_growth:+.1f}%")
            print(f"  🎯 Jan-June Overall Growth: {total_period_growth:+.1f}%")
            print(f"  🏆 Best Revenue Month: {best_revenue_month}")
            print(f"  ⚠️ Challenging Revenue Month: {worst_revenue_month}")
        
        # Comparison if both stores have data
        if len(stores) > 1:
            print(f"\n🔄 STORE COMPARISON:")
            print("-" * 50)
            
            dja_data = growth_analysis.get('DJA')
            yas_data = growth_analysis.get('YAS')
            
            if dja_data is not None and yas_data is not None:
                dja_avg_growth = dja_data['Revenue_Growth_MoM'].mean()
                yas_avg_growth = yas_data['Revenue_Growth_MoM'].mean()
                
                better_performer = "DJA" if dja_avg_growth > yas_avg_growth else "YAS"
                print(f"🏆 Better Growth Performance: {better_performer}")
                print(f"   DJA Average Growth: {dja_avg_growth:+.1f}%")
                print(f"   YAS Average Growth: {yas_avg_growth:+.1f}%")
        
        # If only one store has data, explain why
        if len(stores) == 1:
            print(f"\n⚠️ NOTE: Only {stores[0]} store has data in this period")
            print("   YAS store data may be in different date range or separate table")
        
        conn.close()
        print(f"\n✅ Analysis Complete!")
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        conn.close()

if __name__ == "__main__":
    analyze_month_over_month_growth()
