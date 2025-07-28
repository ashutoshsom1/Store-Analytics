"""
Executive Analytics Agent for IKEA Store Performance
Handles complex business intelligence queries for management reporting
"""

import os
import sqlite3
import pandas as pd
from typing import Dict, Any
from logger import logger

class IKEAExecutiveAnalytics:
    def __init__(self):
        self.base_dir = os.getcwd()
        self.db_path = os.path.join(self.base_dir, "data", "ikea_store_database.db")
    
    def get_connection(self):
        """Get database connection"""
        return sqlite3.connect(self.db_path)
    
    def store_performance_comparison(self) -> Dict[str, Any]:
        """Compare DJA vs YAS store performance across all KPIs"""
        try:
            conn = self.get_connection()
            
            query = """
            SELECT 
                Store,
                ROUND(SUM(Act), 2) as Total_Revenue_AED,
                ROUND(SUM(Ly), 2) as Last_Year_Revenue_AED,
                ROUND(AVG(vs_Ly_percent), 2) as Avg_YoY_Growth_Percent,
                SUM(Visitors) as Total_Visitors,
                SUM(Customers) as Total_Customers,
                ROUND(AVG(Conversion), 4) as Avg_Conversion_Rate,
                ROUND(AVG(ATV), 2) as Avg_Transaction_Value_AED,
                SUM(Item_Sold) as Total_Items_Sold,
                ROUND(AVG(Price_Item), 2) as Avg_Price_Per_Item_AED,
                ROUND(AVG(Item_Cust), 2) as Avg_Items_Per_Customer
            FROM store_data 
            GROUP BY Store 
            ORDER BY Total_Revenue_AED DESC
            """
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if len(df) >= 2:
                top_store = df.iloc[0]
                bottom_store = df.iloc[1]
                revenue_gap = top_store['Total_Revenue_AED'] - bottom_store['Total_Revenue_AED']
                
                analysis = {
                    "summary": f"{top_store['Store']} outperforms {bottom_store['Store']} by AED {revenue_gap:,.2f}",
                    "detailed_comparison": df.to_dict('records'),
                    "key_insights": [
                        f"Revenue leader: {top_store['Store']} (AED {top_store['Total_Revenue_AED']:,.2f})",
                        f"Conversion leader: {df.loc[df['Avg_Conversion_Rate'].idxmax(), 'Store']} ({df['Avg_Conversion_Rate'].max():.4f})",
                        f"ATV leader: {df.loc[df['Avg_Transaction_Value_AED'].idxmax(), 'Store']} (AED {df['Avg_Transaction_Value_AED'].max():.2f})"
                    ]
                }
                return analysis
            else:
                return {"error": "Insufficient store data for comparison"}
                
        except Exception as e:
            logger.error(f"Error in store performance comparison: {str(e)}")
            return {"error": f"Analysis failed: {str(e)}"}
    
    def channel_revenue_analysis(self) -> Dict[str, Any]:
        """Analyze revenue distribution between Store and IFB channels"""
        try:
            conn = self.get_connection()
            
            query = """
            SELECT 
                Channel,
                ROUND(SUM(Act), 2) as Total_Revenue_AED,
                ROUND(AVG(ATV), 2) as Avg_Transaction_Value_AED,
                ROUND(AVG(Conversion), 4) as Avg_Conversion_Rate,
                COUNT(*) as Total_Records
            FROM store_data 
            GROUP BY Channel 
            ORDER BY Total_Revenue_AED DESC
            """
            
            df = pd.read_sql_query(query, conn)
            
            # Calculate percentages
            total_revenue = df['Total_Revenue_AED'].sum()
            df['Revenue_Percentage'] = (df['Total_Revenue_AED'] / total_revenue * 100).round(2)
            
            conn.close()
            
            analysis = {
                "channel_breakdown": df.to_dict('records'),
                "total_revenue_aed": total_revenue,
                "key_insights": [
                    f"Primary channel: {df.iloc[0]['Channel']} ({df.iloc[0]['Revenue_Percentage']:.1f}% of revenue)",
                    f"Secondary channel: {df.iloc[1]['Channel']} ({df.iloc[1]['Revenue_Percentage']:.1f}% of revenue)",
                    f"ATV difference: AED {abs(df.iloc[0]['Avg_Transaction_Value_AED'] - df.iloc[1]['Avg_Transaction_Value_AED']):.2f}"
                ]
            }
            return analysis
            
        except Exception as e:
            logger.error(f"Error in channel revenue analysis: {str(e)}")
            return {"error": f"Analysis failed: {str(e)}"}
    
    def monthly_growth_trends(self) -> Dict[str, Any]:
        """Show month-over-month growth trends for both stores"""
        try:
            conn = self.get_connection()
            
            query = """
            SELECT 
                strftime('%Y-%m', Date) as Month,
                Store,
                ROUND(SUM(Act), 2) as Monthly_Revenue_AED,
                ROUND(AVG(vs_Ly_percent), 2) as Avg_YoY_Growth_Percent,
                SUM(Customers) as Monthly_Customers
            FROM store_data 
            GROUP BY strftime('%Y-%m', Date), Store
            ORDER BY Month, Store
            """
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            # Calculate month-over-month growth
            df_pivot = df.pivot(index='Month', columns='Store', values='Monthly_Revenue_AED')
            mom_growth = df_pivot.pct_change() * 100
            
            analysis = {
                "monthly_trends": df.to_dict('records'),
                "month_over_month_growth": mom_growth.fillna(0).round(2).to_dict(),
                "key_insights": [
                    f"Total months analyzed: {df['Month'].nunique()}",
                    f"Strongest month: {df.loc[df['Monthly_Revenue_AED'].idxmax(), 'Month']} ({df.loc[df['Monthly_Revenue_AED'].idxmax(), 'Store']})",
                    f"Average YoY growth: {df['Avg_YoY_Growth_Percent'].mean():.2f}%"
                ]
            }
            return analysis
            
        except Exception as e:
            logger.error(f"Error in monthly growth analysis: {str(e)}")
            return {"error": f"Analysis failed: {str(e)}"}
    
    def conversion_funnel_analysis(self) -> Dict[str, Any]:
        """Analyze visitor to customer conversion funnel"""
        try:
            conn = self.get_connection()
            
            query = """
            SELECT 
                Store,
                Channel,
                SUM(Visitors) as Total_Visitors,
                SUM(Customers) as Total_Customers,
                ROUND(AVG(Conversion), 4) as Avg_Conversion_Rate,
                ROUND((CAST(SUM(Customers) AS FLOAT) / SUM(Visitors)) * 100, 2) as Overall_Conversion_Percent
            FROM store_data 
            GROUP BY Store, Channel
            ORDER BY Overall_Conversion_Percent DESC
            """
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            # Overall metrics
            total_visitors = df['Total_Visitors'].sum()
            total_customers = df['Total_Customers'].sum()
            overall_conversion = (total_customers / total_visitors * 100) if total_visitors > 0 else 0
            
            analysis = {
                "funnel_breakdown": df.to_dict('records'),
                "overall_metrics": {
                    "total_visitors": total_visitors,
                    "total_customers": total_customers,
                    "overall_conversion_rate": round(overall_conversion, 2)
                },
                "key_insights": [
                    f"Best converting store/channel: {df.iloc[0]['Store']}-{df.iloc[0]['Channel']} ({df.iloc[0]['Overall_Conversion_Percent']:.2f}%)",
                    f"Conversion opportunity: {total_visitors - total_customers:,} visitors didn't convert",
                    f"Average conversion rate: {df['Overall_Conversion_Percent'].mean():.2f}%"
                ]
            }
            return analysis
            
        except Exception as e:
            logger.error(f"Error in conversion funnel analysis: {str(e)}")
            return {"error": f"Analysis failed: {str(e)}"}
    
    def anomaly_detection(self, kpi_column: str = 'Act') -> Dict[str, Any]:
        """Detect anomalies in specified KPI using IQR method"""
        try:
            conn = self.get_connection()
            
            # Calculate IQR bounds
            query_stats = f"""
            WITH ordered_data AS (
                SELECT {kpi_column}, ROW_NUMBER() OVER (ORDER BY {kpi_column}) as rn,
                       COUNT(*) OVER () as total_count
                FROM store_data
                WHERE {kpi_column} IS NOT NULL
            ),
            quartiles AS (
                SELECT 
                    (SELECT {kpi_column} FROM ordered_data WHERE rn = CAST(total_count * 0.25 AS INT)) as Q1,
                    (SELECT {kpi_column} FROM ordered_data WHERE rn = CAST(total_count * 0.75 AS INT)) as Q3
            )
            SELECT Q1, Q3, (Q3 - Q1) as IQR FROM quartiles
            """
            
            stats_df = pd.read_sql_query(query_stats, conn)
            Q1, Q3, IQR = stats_df.iloc[0]['Q1'], stats_df.iloc[0]['Q3'], stats_df.iloc[0]['IQR']
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Find anomalies
            query_anomalies = f"""
            SELECT Date, Store, Channel, {kpi_column}
            FROM store_data
            WHERE {kpi_column} < {lower_bound} OR {kpi_column} > {upper_bound}
            ORDER BY ABS({kpi_column} - {(Q1 + Q3) / 2}) DESC
            LIMIT 10
            """
            
            anomalies_df = pd.read_sql_query(query_anomalies, conn)
            conn.close()
            
            analysis = {
                "kpi_analyzed": kpi_column,
                "statistical_bounds": {
                    "Q1": Q1, "Q3": Q3, "IQR": IQR,
                    "lower_bound": lower_bound, "upper_bound": upper_bound
                },
                "anomalies_found": len(anomalies_df),
                "anomaly_records": anomalies_df.to_dict('records'),
                "key_insights": [
                    f"Total anomalies detected: {len(anomalies_df)}",
                    f"Most extreme value: {anomalies_df.iloc[0][kpi_column] if len(anomalies_df) > 0 else 'None'}",
                    f"Normal range: {lower_bound:.2f} to {upper_bound:.2f}"
                ] if len(anomalies_df) > 0 else ["No anomalies detected in the specified KPI"]
            }
            return analysis
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {str(e)}")
            return {"error": f"Analysis failed: {str(e)}"}

def get_executive_analytics_agent(question: str) -> Dict[str, Any]:
    """Main function to handle executive analytics queries"""
    analytics = IKEAExecutiveAnalytics()
    
    question_lower = question.lower()
    
    if any(keyword in question_lower for keyword in ['store performance', 'compare stores', 'dja vs yas', 'better store']):
        return {"analysis_type": "Store Performance Comparison", "results": analytics.store_performance_comparison()}
    
    elif any(keyword in question_lower for keyword in ['channel', 'ifb', 'food beverages', 'revenue mix']):
        return {"analysis_type": "Channel Revenue Analysis", "results": analytics.channel_revenue_analysis()}
    
    elif any(keyword in question_lower for keyword in ['monthly', 'month-over-month', 'growth trends', 'trend']):
        return {"analysis_type": "Monthly Growth Trends", "results": analytics.monthly_growth_trends()}
    
    elif any(keyword in question_lower for keyword in ['conversion', 'funnel', 'visitor', 'customer journey']):
        return {"analysis_type": "Conversion Funnel Analysis", "results": analytics.conversion_funnel_analysis()}
    
    elif any(keyword in question_lower for keyword in ['anomaly', 'anomalies', 'unusual', 'outlier']):
        # Determine which KPI to analyze
        kpi_column = 'Act'  # default
        if 'conversion' in question_lower:
            kpi_column = 'Conversion'
        elif 'atv' in question_lower or 'transaction value' in question_lower:
            kpi_column = 'ATV'
        elif 'visitor' in question_lower:
            kpi_column = 'Visitors'
        elif 'customer' in question_lower:
            kpi_column = 'Customers'
        
        return {"analysis_type": "Anomaly Detection", "results": analytics.anomaly_detection(kpi_column)}
    
    else:
        # Default comprehensive analysis
        return {
            "analysis_type": "Comprehensive Performance Dashboard",
            "results": {
                "store_comparison": analytics.store_performance_comparison(),
                "channel_analysis": analytics.channel_revenue_analysis(),
                "conversion_metrics": analytics.conversion_funnel_analysis()
            }
        }
