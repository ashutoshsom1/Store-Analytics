import re
from sentence_transformers import SentenceTransformer
import numpy as np
import streamlit as st


import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('agg')
import matplotlib.pyplot as plt
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)

import pandas as pd
from typing import List
from pydantic import BaseModel, Field

import langchain
langchain.verbose = False

from dotenv import load_dotenv
load_dotenv(".env")

from constants.constant import *
import yaml
from data.dataloader import *
from utils.utils import extract_sql

import os
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

from logger import logger


# Updated create_retriver_for_sql that uses LangChain chain instead of raw LLM calls
def retrieve_examples(user_query, table_schema,model,index,df, top_k=5):
    """retrieve similar QA pairs from the vector store based on distance score"""
    query_text = str(user_query) + " | " + str(table_schema)
    logger.info(f"Query Text: {query_text}")
    query_vec = model.encode([query_text])
    # search the faiss index (euclidean distance) to fetch similar few shot examples
    D, I = index.search(np.array(query_vec), k=top_k)
    return df.iloc[I[0]]


def build_prompt(retrieved_df:pd.DataFrame, user_query:str, db)->str:
    """builds a semantically similar few shot SQL code generation prompt"""
    prompt =f'''
        You are a SQL expert who can create syntactically correct SQLite sql queries for a given question provided with table information and schemas.
        ### Tables Info:
        * `store_data`: Records detailed IKEA store performance data including KPIs like Actuals, Last Year, Visitors, Customers, Conversion rates, ATV, and item metrics for DJA and YAS stores in AE region across Store and IFB (IKEA Food & Beverages) channels.
        ### Always adhere to these principles when writing SQLite queries against the schema below:
        {db.get_context()['table_info']}
        ### Below are examples of questions and their corresponding SQL queries:
    '''
    
    # Use the correct column names: 'question' and 'sql' instead of 'question' and 'answer'
    for _, row in retrieved_df.iterrows():
        prompt += f"\nQ: {row['question']}\nA: {row['sql']}\n"
 
    prompt += f"\n### New Question:\nQ: {user_query}\nA: <PROVIDE THE NECESSARY SQL QUERY HERE TO ANSWER THE QUESTION>"
    
    end='''  
        ###Table Relationships and Key Information:
           *In the `store_data` table, the following relationships and constraints apply:
            - Region is fixed as 'AE' (United Arab Emirates)
            - Store values are limited to 'DJA' and 'YAS'
            - Channel values are limited to 'Store' and 'IFB' (IKEA Food & Beverages)
            - Date format is YYYY-MM-DD covering Jan to Jun 2025
            - All monetary values (Act, Ly, ATV, Price_Item) are in AED currency
            - Conversion = Customers / Visitors
            - ATV = Act / Customers  
            - Price_Item = Act / Item_Sold
            - Item_Cust = Item_Sold / Customers
            - vs_Ly_percent = ((Act - Ly) / Ly) * 100
            
        Important Notes:
        - Use strftime('%w', Date) for day of week: 0=Sunday, 6=Saturday
        - For weekend analysis: strftime('%w', Date) IN ('0', '6') 
        - For conversion rates, multiply by 100 for percentage
        - Always filter by Store and appropriate date ranges
        - Use appropriate GROUP BY clauses for aggregations
        '''
    prompt += f"\n{end}"
    return prompt
 
def get_sql_from_llm(prompt:str, agent=llm)->str:
    """generates executable sql code from the llm"""
    sysyem_prompt='''
            You are a helpful assistant that writes syntactically correct SQLite SQL queries.
            **### SQL Rules**
            1. **Syntax & Efficiency**
                * Write **syntactically correct** SQLite queries that accomplish the user's request.
                * Minimize scanned rows—leverage proper filtering, indexing, `DISTINCT` and `WHERE` clauses whenever necessary
                * Use `LIMIT 5` statement by default if the number is not specifically mentioned by the user or the user specifically
                  asks for all the data. If the user requests to see the whole data, use aggregation in your query to minimize the data
                  volumne in the view that the database will returns.
                * Avoid functions or constructs not supported by SQLite (e.g., no `FIELD()` in `ORDER BY`).
                * Never issue DML statements (`INSERT`, `UPDATE`, `DELETE`, `DROP`) or any query likely to overload the database.

            2.If the question asked can be answered using the existing columns on the tables, you must use those columns instead of deriving them.

            3. **Anomaly Related Questions Handling**
                * If the user asks about anomaly or anomaly related question; use the following framework to solve this question:
                    - break down the user's question analytically to understand the intent and requirement; think step by step
                    - understand on which column of a table you need to use to answer the user question
                    - if the column is of numerical type, use the inter quartile range on that column to fetch the anomalous rows
                    - if the column is of a categorical type, take either Actuals (Act) or revenue as reference to calculate the anomaly
                    - always use the Act (Actuals) column of the store data by default if a column is not mentioned by the user 
            
            5. **Output & Presentation**
                * **Never** return raw SQL or unformatted result sets directly. Always wrap insights in explanatory text or visualizations if required.
                * Use `ORDER BY` judiciously to make results deterministic when needed.
            
            6. **Out-of-Scope Handling**
                * If the question cannot be answered from the available tables or falls outside these rules, respond:
                 > “This question is out of my current scope.”
     
            Follow these guidelines to ensure every query is correct, performant, and user-friendly.
        '''
    messages=[
        {"role": "system", "content":sysyem_prompt},
        {"role": "user", "content": prompt}
        ]
    # invoke the client for query generation
    response=agent.invoke(messages)   
    return response.content

def create_retriver_for_sql(user_query:str)->str:
    """
        fetches semantically similar sql question->code examples from the knowledge store
        using the euclidean distance metric and FAISS indexes
    """
    # retrieve the similar examples from the vector database
    retrieved_df = retrieve_examples(user_query, table_schema, model, index, df)
    # use the retrieved examples to build a few shot prompt for llm invocation
    prompt=build_prompt(retrieved_df, user_query,db)
    # get the SQL code output from the model using the few-shot prompt
    output=get_sql_from_llm(prompt,llm)
    logger.info(f"✅ Generated SQL query =>\n {output}")
    # parse the sql code
    res = extract_sql(output)
    logger.info(f"Extracted SQL query :\n {res}")
    try:
        answer = db.run(res)
        if not answer or (isinstance(answer, str) and answer.strip() == ""):
            raise ValueError(f"Empty result: {answer}; likely due to incorrect SQL code: {res}")
        
        # Format the answer as a meaningful response with better structure
        logger.info(f"✅ SQL Query Results: {answer}")
        return f"""SQL Query executed successfully:
{res}

Results:
{answer}"""
    except Exception as e:
        error_msg = str(e)
        logger.info(f"⚠️ SQL execution failed. Error: {error_msg}")
        repair_prompt = f"""
            The following SQL query generated no data and/or returned an error on a sqlite database provided below

            Original Question: {user_query}
            Error Message: {error_msg}
            Check the table information: {metadata_content}

            ###Table Relationships and Key Information:
            *In the `store_data` table, the following relationships and constraints apply:
                - Region is fixed as 'AE' (United Arab Emirates)
                - Store values are limited to 'DJA' and 'YAS'
                - Channel values are limited to 'Store' and 'IFB' (IKEA Food & Beverages)
                - Date format is YYYY-MM-DD covering Jan to Jun 2025
                - All monetary values (Act, Ly, ATV, Price_Item) are in AED currency
                - Conversion = Customers / Visitors
                - ATV = Act / Customers  
                - Price_Item = Act / Item_Sold
                - Item_Cust = Item_Sold / Customers
                - vs_Ly_percent = ((Act - Ly) / Ly) * 100

            Based on information above, fix the sql query to make it syntactically correct that aligns 
            with the user's question and mitigates the error message when ran on a database.
            you must provide only the corrected SQLite query and nothing more or less. 
            you don't need to explain how you've written the sql query. 
        """
        # generate a repaired SQL code based on error message and previous code
        repaired_output = get_sql_from_llm(repair_prompt, llm)
        # parse the generated code
        repaired_sql = extract_sql(repaired_output)
        logger.info(f"⛏️ Repaired SQL: {repaired_sql}")
        try:
            answer = db.run(repaired_sql)
            if not answer or (isinstance(answer, str) and answer.strip() == ""):
                raise ValueError("Second attempt also returned empty result.")
            logger.info(f"✅ Repaired SQL Results: {answer}")
            return f"""SQL Query (Repaired) executed successfully:
{repaired_sql}

Results:
{answer}"""
        except Exception as e2:
            error_message = f"SQL execution failed after repair attempt. Original error: {error_msg}. Repair error: {str(e2)}"
            logger.error(error_message)
            return f"Unable to execute query. Error: {error_message}"