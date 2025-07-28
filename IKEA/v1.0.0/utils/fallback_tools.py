import os
import re
import sqlite3
import time
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import streamlit as st
import yaml
from pydantic import BaseModel, Field

import langchain
from langchain.chains import create_sql_query_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain_community.vectorstores import FAISS
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.example_selectors.semantic_similarity import SemanticSimilarityExampleSelector


from constants import *
from logger import logger
from data.dataloader import *
from utils.utils import *
from utils.executive_analytics import get_executive_analytics_agent

langchain.verbose = False
st.set_option("deprecation.showPyplotGlobalUse", False)
# matplotlib.use('TkAgg')  # Commented out as per previous response
matplotlib.use("agg")



def get_top_key_information_of_store(data:dict)->dict:
    """retrieves the store related information for a given store identifier"""
    try:
        store_name = data['store_name']  # DJA or YAS
        # Get the directory where the current .py file is located
        base_dir = os.getcwd()
        # Build the path to the SQLite database file (relative to this script)
        db_path = os.path.join(base_dir, "data", "ikea_store_database.db")
        # Create SQLDatabase instance with the full path        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        #Total Actuals (revenue) in the current period
        sql_query = """SELECT 
            round(SUM(Act)) AS total_actuals
        FROM 
            store_data
        WHERE 
            Store = ? 
            AND Date >= '2025-01-01' AND Date <= '2025-06-30';"""
        cursor.execute(sql_query, (store_name,))
        total_actuals_current_period = cursor.fetchall()
        total_actuals_current_period = [item for sublist in total_actuals_current_period for item in sublist][0]
        
        #Total Last Year comparison in the current period
        sql_query = """SELECT 
            round(SUM(Ly)) AS total_ly
        FROM 
            store_data
        WHERE 
            Store = ? 
            AND Date >= '2025-01-01' AND Date <= '2025-06-30';"""
        cursor.execute(sql_query, (store_name,))
        total_ly_current_period = cursor.fetchall()
        total_ly_current_period = [item for sublist in total_ly_current_period for item in sublist][0]

        #Total customers in the current period
        sql_query = """SELECT 
            SUM(Customers) AS total_customers
        FROM 
            store_data
        WHERE 
            Store = ? 
            AND Date >= '2025-01-01' AND Date <= '2025-06-30';"""
        cursor.execute(sql_query, (store_name,))
        total_customers_current_period = cursor.fetchall()
        total_customers_current_period = [item for sublist in total_customers_current_period for item in sublist][0]

        #Top 3 performing channels by revenue in the current period
        sql_query = """SELECT 
            Channel,
            round(SUM(Act)) AS total_actuals
        FROM 
            store_data
        WHERE Store = ? AND Date >= '2025-01-01' AND Date <= '2025-06-30'
        GROUP BY 
            Channel
        ORDER BY 
            total_actuals DESC
        LIMIT 3;"""
        cursor.execute(sql_query, (store_name,))
        top_channels = cursor.fetchall()
        top_performing_channels = [item[0] for item in top_channels]
        
        #Store ranking in terms of actuals in the current period
        sql_query = """WITH
        store_actuals AS (
            SELECT
                Store,
                SUM(Act) AS total_actuals
            FROM
                store_data
            WHERE
                Date >= '2025-01-01' AND Date <= '2025-06-30'
            GROUP BY
                Store
        ), ranked_stores AS (
            SELECT
                Store,
                total_actuals,
                RANK() OVER (ORDER BY total_actuals DESC) AS rank
            FROM
                store_actuals
        )SELECT
            rank
        FROM
            ranked_stores
        WHERE
            Store = ?;"""
        cursor.execute(sql_query, (store_name,))
        store_ranking_actuals_current_period = cursor.fetchall()
        store_ranking_actuals_current_period = [item for sublist in store_ranking_actuals_current_period for item in sublist][0]
        
       #Peak Performing Date
        sql_query = """SELECT 
            Date AS peak_date,
            Act AS daily_actuals
        FROM store_data
        WHERE Store = ?
        ORDER BY Act DESC
        LIMIT 1;
        """
        cursor.execute(sql_query, (store_name,))
        peak_performing_date = cursor.fetchall()
        peak_performing_date = [item for sublist in peak_performing_date for item in sublist][0]

        conn.close()

        data_req = {'Store Name': store_name ,\
                    'Total Customers Served in Jan-Jun 2025': total_customers_current_period,\
                    'Peak Performing Date' : peak_performing_date,\
                    'Total Actuals in Jan-Jun 2025' : total_actuals_current_period,\
                    'Total Last Year Comparison in Jan-Jun 2025' : total_ly_current_period,\
                    'Top Performing Channels in Jan-Jun 2025': top_performing_channels,\
                    'Current Period Performance Rank out of Total Stores':store_ranking_actuals_current_period
                    }
        return data_req
    except:
        return "Store Name not found!"



def get_recommendation_agent(question:str)->dict:
    """create recommendations based on store performance patterns and KPI trends"""
    # Get the directory where the current .py file is located
    base_dir = os.getcwd()
    # Build the path to the SQLite database file (relative to this script)
    db_path = os.path.join(base_dir, "data", "ikea_store_database.db")
    # Create SQLDatabase instance with the full path
    db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
    examples = [
    {
        "input": "What recommendations can you provide to improve DJA store performance?",
        "query": """
                    SELECT 
                        Channel,
                        AVG(Conversion) as avg_conversion,
                        AVG(ATV) as avg_atv,
                        AVG(Item_Cust) as avg_items_per_customer
                    FROM store_data 
                    WHERE Store = 'DJA'
                    GROUP BY Channel
                    ORDER BY avg_conversion DESC;
                """
        },
    {
        "input":"Which channel performs better for YAS store in terms of conversion?",

        "query":"""
            SELECT 
                Channel,
                AVG(Conversion) as avg_conversion,
                SUM(Act) as total_actuals
            FROM store_data 
            WHERE Store = 'YAS'
            GROUP BY Channel
            ORDER BY avg_conversion DESC;"""
    },
    {
        "input":"What is the total performance comparison between DJA and YAS stores?",
        "query":"""SELECT 
                Store,
                SUM(Act) as total_actuals,
                SUM(Customers) as total_customers,
                AVG(Conversion) as avg_conversion,
                AVG(ATV) as avg_atv
                FROM store_data
                GROUP BY Store
                ORDER BY total_actuals DESC;"""
    },
    {
        "input":"Which store has better performance in IKEA Food & Beverages channel?",
        "query":"""SELECT 
                Store,
                SUM(Act) as total_actuals,
                AVG(Conversion) as avg_conversion
                FROM store_data
                WHERE Channel = 'IFB'
                GROUP BY Store
                ORDER BY total_actuals DESC;
                """
    }]
    
    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples,
        AzureOpenAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL"),
            azure_endpoint=os.getenv("EMBEDDING_ENDPOINT"),
            api_key=os.getenv("EMBEDDING_KEY"),
            api_version=os.getenv("EMBEDDING_API_VERSION")
            ),
        FAISS,
        k=3,
        input_keys=["input"],
    )


    system_prefix = """You are an agent designed to interact with a SQL database.

    store_data: This table stores information about IKEA store performance including Actuals, Last Year, Visitors, Customers, 
    Conversion rates, ATV, Item_Sold, Price_Item, and Item_Cust for DJA and YAS stores in AE region across Store and IFB channels.
    Important Note: I will use appropriate logic (e.g., DISTINCT) when performing calculations on this table to account for any 
    duplicate values. This ensures accurate results for measures like average, count, and sum.
    When I need to provide recommendations, I need to always refer this table.

    Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
    Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
    You can order the results by a relevant column to return the most interesting examples in the database.
    Never query for all the columns from a specific table, only ask for the relevant columns given the question.
    You have access to tools for interacting with the database.
    Only use the given tools. Only use the information returned by the tools to construct your final answer.
    You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

    DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

    If the question does not seem related to the database, just return "I don't know" as the answer.
    If you cannot find the database then , just return "I could not find information"

    You must execute sql statements, do not return any sql query as an output.
    Add DISTINCT to your generated sql query

    You must be used when you will have recommendation related questions about store performance.    
    Here are some examples of user inputs and their corresponding SQL queries:"""

    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=PromptTemplate.from_template(
            "User input: {input}\nSQL query: {query}"
        ),
        input_variables=["input", "dialect", "top_k"],
        prefix=system_prefix,
        suffix="",
    )
    full_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate(prompt=few_shot_prompt),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
    agent_executor = create_sql_agent(
    llm=llm,
    db=db,
    prompt=full_prompt,
    verbose=True,
    agent_type="openai-tools",
    max_iterations=15,
    max_execution_time=90,
    early_stopping_method='force'
    )
    output=agent_executor.invoke(question)
    return {"data_req":output}

def get_forecast_details(question:dict)->dict:
    """provides year-over-year comparison and trend analysis for IKEA store data"""
    # Get the directory where the current .py file is located
    base_dir = os.getcwd()
    # Build the path to the SQLite database file (relative to this script)
    db_path = os.path.join(base_dir, "data", "ikea_store_database.db")
    # Create SQLDatabase instance with the full path
    db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    examples = [
    {"input": "What is the year-over-year growth for DJA store?", "query": "SELECT Store, AVG(vs_Ly_percent) as avg_yoy_growth, SUM(Act) as total_actuals, SUM(Ly) as total_ly FROM store_data WHERE Store='DJA' GROUP BY Store;"},
    {"input": "Show me the performance trend for YAS store in IFB channel?", "query": "SELECT Date, Act, Ly, vs_Ly_percent FROM store_data WHERE Store='YAS' AND Channel='IFB' ORDER BY Date;"}
    ]
    
    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples,
        AzureOpenAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL"),
            azure_endpoint=os.getenv("EMBEDDING_ENDPOINT"),
            api_key=os.getenv("EMBEDDING_KEY"),
            api_version=os.getenv("EMBEDDING_API_VERSION")
            ),
        FAISS,
        k=2,
        input_keys=["input"],
    )


    system_prefix = """You are an agent designed to interact with a SQL database.

    store_data: This table stores information about IKEA store performance including Actuals (Act), Last Year (Ly), year-over-year 
    percentage change (vs_Ly_percent), Visitors, Customers, Conversion rates, ATV, Item_Sold, Price_Item, and Item_Cust for DJA and YAS 
    stores in AE region across Store and IFB channels for Jan-Jun 2025 period.
    Important Note: I will use appropriate logic (e.g., DISTINCT) when performing calculations on this table to account for duplicate 
    values in certain columns. This ensures accurate results for measures like average, count, and sum.
    You are expert in answering questions related to year-over-year comparisons and trend analysis.

    Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
    Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
    You can order the results by a relevant column to return the most interesting examples in the database.

    Key Principle:
        - Never query for all the columns from a specific table, only ask for the relevant columns given the question.
        - You have access to tools for interacting with the database.
        - Only use the given tools. Only use the information returned by the tools to construct your final answer.
        - You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.
        - DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
        - If the question does not seem related to the database, just return "I don't know" as the answer.
        - If you cannot find the database then , just return "I could not find information"
        - You must execute sql statements, do not return any sql query as an output.
        - Add DISTINCT to your generated sql query whenever required
        - Focus on year-over-year analysis using Act, Ly, and vs_Ly_percent columns
        - The Date column follows the format YYYY-MM-DD covering Jan-Jun 2025
    
    Here are some examples of user inputs and their corresponding SQL queries:"""

    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=PromptTemplate.from_template(
            "User input: {input}\nSQL query: {query}"
        ),
        input_variables=["input", "dialect", "top_k"],
        prefix=system_prefix,
        suffix="",
    )

    full_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate(prompt=few_shot_prompt),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
    )
    agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    prompt=full_prompt,
    verbose=True,
    agent_type="openai-tools",
    max_iterations=15,
    max_execution_time=90,
    early_stopping_method='force'
    )
    output=agent_executor.invoke(question)
    return {'trend_analysis': output} 

def get_anomaly_agent(question:dict)->dict:
    examples = [
    {"input": "List all stores.", "query": "select DISTINCT Store from store_data;"},
    {"input":"Give me the channels available in the data",
     "query":"""
            SELECT DISTINCT Channel FROM store_data;"""},
    {
        "input": "Find all stores having anomalies in Actuals column.",
        "query": """WITH ordered AS (
                    SELECT Act, 
                            ROW_NUMBER() OVER (ORDER BY Act) AS row_num,
                            COUNT(*) OVER () AS total_rows
                    FROM store_data
                    ),
                    quartiles AS (
                    SELECT 
                        (SELECT Act FROM ordered WHERE row_num = CAST(total_rows * 0.25 AS INT)) AS Q1,
                        (SELECT Act FROM ordered WHERE row_num = CAST(total_rows * 0.75 AS INT)) AS Q3
                    )
                    SELECT DISTINCT s.Store
                    FROM store_data s
                    JOIN quartiles q
                    WHERE s.Act < (q.Q1 - 1.5 * (q.Q3 - q.Q1));"""
    },
    {
        "input": "Find all channels having anomalies in Actuals column.",
        "query": """
                WITH ordered AS (
                SELECT Act, 
                        ROW_NUMBER() OVER (ORDER BY Act) AS row_num,
                        COUNT(*) OVER () AS total_rows
                FROM store_data
                ),
                quartiles AS (
                SELECT 
                    (SELECT Act FROM ordered WHERE row_num = CAST(total_rows * 0.25 AS INT)) AS Q1,
                    (SELECT Act FROM ordered WHERE row_num = CAST(total_rows * 0.75 AS INT)) AS Q3
                )
                SELECT DISTINCT s.Channel
                FROM store_data s
                JOIN quartiles q
                WHERE s.Act < (q.Q1 - 1.5 * (q.Q3 - q.Q1))""",
                },
    {
        "input": "Find all stores having anomalies",
        "query": """WITH ordered AS (
                    SELECT Act, 
                            ROW_NUMBER() OVER (ORDER BY Act) AS row_num,
                            COUNT(*) OVER () AS total_rows
                    FROM store_data
                    ),
                    quartiles AS (
                    SELECT 
                        (SELECT Act FROM ordered WHERE row_num = CAST(total_rows * 0.25 AS INT)) AS Q1,
                        (SELECT Act FROM ordered WHERE row_num = CAST(total_rows * 0.75 AS INT)) AS Q3
                    )
                    SELECT DISTINCT s.Store
                    FROM store_data s
                    JOIN quartiles q
                    WHERE s.Act < (q.Q1 - 1.5 * (q.Q3 - q.Q1));"""
    },
    {
        "input":"Give me date information in the store data",
        "query":"""SELECT DISTINCT Date FROM store_data ORDER BY Date;"""
    },
    {
        "input":"Give me store performance details",
        "query":"SELECT Store, Channel, SUM(Act) as total_actuals, AVG(Conversion) as avg_conversion FROM store_data GROUP BY Store, Channel;"
    },
    {
        "input":"I need you to give me the most anomalous store performance",
        "query":"""
                WITH ordered AS 
                    (
                        SELECT Store, Act, ROW_NUMBER() OVER (ORDER BY Act) AS row_num, COUNT(*) OVER () AS total_rows 
                        FROM store_data
                    ), 
                quartiles AS 
                    (SELECT 
                        (
                            SELECT Act 
                            FROM ordered 
                            WHERE row_num = CAST(total_rows * 0.25 AS INT)) AS Q1, 
                            (
                                SELECT Act 
                                FROM ordered 
                                WHERE row_num = CAST(total_rows * 0.75 AS INT)) AS Q3
                            ) 
                SELECT DISTINCT s.Store, s.Date, s.Act 
                FROM store_data s 
                JOIN quartiles q 
                WHERE s.Act < (q.Q1 - 1.5 * (q.Q3 - q.Q1)) 
                ORDER BY s.Act ASC LIMIT 1
                """
    }]
    
    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples,
        AzureOpenAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL"),
            azure_endpoint=os.getenv("EMBEDDING_ENDPOINT"),
            api_key=os.getenv("EMBEDDING_KEY"),
            api_version=os.getenv("EMBEDDING_API_VERSION")
            ),
        FAISS,
        k=10,
        input_keys=["input"],
    )
    
    system_prefix = """You are an agent designed to interact with a SQL database.
    store_data: This table stores information about IKEA store performance including Actuals (Act), Last Year (Ly), 
    year-over-year percentage (vs_Ly_percent), Visitors, Customers, Conversion, ATV, Item_Sold, Price_Item, and Item_Cust 
    for DJA and YAS stores in AE region across Store and IFB channels for Jan-Jun 2025.
    Important Note: I will use appropriate logic (e.g., DISTINCT) when performing calculations on this table to account for duplicate 
    values in certain columns. This ensures accurate results for measures like average, count, and sum.
    When I need to know about any anomaly i need to always refer this table.

    Given an input question, create a syntactically correct SQLite query to run.
    Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
    You can order the results by a relevant column to return the most interesting examples in the database.
    Never query for all the columns from a specific table, only ask for the relevant columns given the question.
    
    DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

    If the question does not seem related to the database, just return "I don't know" as the answer.
    If you cannot find the database then , just return "I could not find information"

    
    Add DISTINCT to your generated sql query
    nHere is the relevant table info: {table_info}Below are a number of examples of questions and their corresponding SQL queries:"""
    example_prompt = PromptTemplate.from_template(
    "Question: {input} \n"
    "SQL query: {query}"
    )

    few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix=system_prefix,
    suffix=(
        "Question: {input} \n"
        "SQL query: "
    ),
    input_variables=["input", "table_info" ,"top_k"]
    )
    full_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate(prompt=few_shot_prompt),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
    )
    agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    prompt=full_prompt,
    verbose=True,
    agent_type="openai-tools",
    max_iterations=15,
    max_execution_time=90,
    early_stopping_method="force"
    )
    answer = agent_executor.invoke(question)
    return answer

def get_kpi_analysis_agent(question:dict)->dict:
    """Analyzes IKEA store KPIs including Conversion, ATV, Item metrics, etc."""
    examples = [
    {"input": "Which store has better conversion rates?", "query": "SELECT Store, AVG(Conversion) as avg_conversion FROM store_data GROUP BY Store ORDER BY avg_conversion DESC;"},
    {"input": "What is the average transaction value for DJA store?", "query": "SELECT Store, AVG(ATV) as avg_transaction_value FROM store_data WHERE Store='DJA';"},
    {"input": "Compare ATV performance between Store and IFB channels", "query": "SELECT Channel, AVG(ATV) as avg_atv, AVG(Item_Cust) as avg_items_per_customer FROM store_data GROUP BY Channel ORDER BY avg_atv DESC;"}
    ]
    
    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples,
        AzureOpenAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL"),
            azure_endpoint=os.getenv("EMBEDDING_ENDPOINT"),
            api_key=os.getenv("EMBEDDING_KEY"),
            api_version=os.getenv("EMBEDDING_API_VERSION")
            ),
        FAISS,
        k=3,
        input_keys=["input"],
    )

    system_prefix = """You are an agent designed to interact with a SQL database.
    store_data: This table stores information about IKEA store KPIs including Actuals, Conversion rates, ATV (Average Transaction Value), 
    Item_Sold, Price_Item, Item_Cust, and other performance metrics for DJA and YAS stores.
    Important Note: I will use appropriate logic (e.g., DISTINCT) when performing calculations on this table to account for duplicate 
    values in certain columns. This ensures accurate results for measures like average, count, and sum.
    You are expert in analyzing store KPIs and performance metrics.
    
    Given an input question, create a syntactically correct SQLite query to run.
    Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
    You can order the results by a relevant column to return the most interesting examples in the database.
    Never query for all the columns from a specific table, only ask for the relevant columns given the question.
    
    DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

    If the question does not seem related to the database, just return "I don't know" as the answer.
    If you cannot find the database then , just return "I could not find information"

    
    Add DISTINCT to your generated sql query
    nHere is the relevant table info: {table_info}
    
    Here are some examples of user inputs and their corresponding SQL queries:"""

    example_prompt = PromptTemplate.from_template(
    "Question: {input}\n"
    "SQL query: {query}"
    )
    few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix=system_prefix,
    suffix=(
        "Question: {input}\n"
        "SQL query: "
    ),
    input_variables=["input", "table_info" ,"top_k"]
    )
    full_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate(prompt=few_shot_prompt),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
    )
    agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    prompt=full_prompt,
    verbose=True,
    agent_type="openai-tools",
    max_iterations=15,
    max_execution_time=90,
    early_stopping_method="force"
    )
    answer = agent_executor.invoke(question)
    return {'kpi_analysis': answer}


def get_performance_summary_agent(question:dict)->dict:
    """Provides summary and comparative analysis of IKEA store performance"""
    # Get the directory where the current .py file is located
    base_dir = os.getcwd()
    # Build the path to the SQLite database file (relative to this script)
    db_path = os.path.join(base_dir, "data", "ikea_store_database.db")
    # Create SQLDatabase instance with the full path
    db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
    agent_executor = create_sql_agent(
    llm=llm,
    db=db,
    prompt="""
                You are an expert assistant tasked with providing summary and comparative analysis of IKEA store performance
                based on the store_data table. Focus on KPIs like Actuals, Conversion rates, ATV, Item metrics, and 
                year-over-year comparisons. Provide insights based on data covering Jan-Jun 2025 for DJA and YAS stores 
                across Store and IFB channels in AE region. If the data does not contain a clear answer, 
                respond with "The answer is not available in the store performance data."
            """,
    verbose=True,
    agent_type="openai-tools",
    max_iterations=15,
    max_execution_time=90,
    early_stopping_method="force"
    )
    output=agent_executor.invoke(question)
    return output

def get_executive_dashboard(question: str) -> dict:
    """Handle executive-level analytical queries using the executive analytics engine"""
    try:
        result = get_executive_analytics_agent(question)
        return {"executive_analysis": result}
    except Exception as e:
        logger.error(f"Error in executive dashboard: {str(e)}")
        return {"error": f"Executive analysis failed: {str(e)}"}
    return {'performance_summary': output}